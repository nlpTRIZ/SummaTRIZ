import os
import pickle
import numpy as np
import torch
from tensorboardX import SummaryWriter
import sys 


import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.init_loss = 100

        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct, test_iter_fct, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """

        logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0

        train_iter = train_iter_fct()
        test_iter = test_iter_fct()
        valid_iter = valid_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        list_save_representations=[[] for i in range(2)]

        while step <= train_steps:

            reduce_counter = 0
            

            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        list_save_representations = self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats,list_save_representations)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)
                            self.validate(valid_iter, step)
                            self.test(test_iter, step)
                            test_iter = test_iter_fct()
                            valid_iter = valid_iter_fct()
                            self.model.train()

                        step += 1
                        if step > train_steps:
                            break
            
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        list_save_representations=[[] for i in range(2)]

        true_positives = 0
        true_positives_margin =0
        true_positives_margin2 =0
        total_positives = 0

        with torch.no_grad():

            probas_tp = np.empty((0))
            probas_fp = np.empty((0))
            probas_fn = np.empty((0))

            sent_tp = []
            sent_fp = []
            sent_fn = []

            pos_labels = []
            count = 0

            # with open("list1.txt", "rb") as fp:   # Unpickling
            #     list1 = pickle.load(fp)

            # list1 = [min(list1[i]) for i in range(len(list1))]


            for batch in valid_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                src_str = batch.src_str
                
            
                sent_scores, mask, sentences_rep= self.model(src, segs, clss, mask, mask_cls)

    
                # Computation true positive
                #####################################################
                scores = (sent_scores*mask.float()).cpu().data.numpy()
                

                for i,label in enumerate(labels.cpu().data.numpy()):
                    pos_label = np.where(label==1)
                    # pos_labels.append(list(pos_label[0]))
                    pos_score = np.argsort(scores[i])

                    src = src_str[i]
                    
                    for pos in range(len(pos_label[0])+2):
                        if (pos_score[-(pos+1)] in pos_label[0]):
                            true_positives_margin+=1

                    # for pos in range(len(pos_label[0])+4):
                    #     if (pos_score[-(pos+1)] in pos_label[0]):
                    #         true_positives_margin2+=1

                    pos_positives = pos_score[-len(pos_label[0]):]
                    pos_groundtruth = pos_label[0]
                    probas_fn = np.append(probas_fn,scores[i][np.setdiff1d(pos_groundtruth, pos_positives)],axis=0)
                    
                    
                    for index in list(np.setdiff1d(pos_groundtruth, pos_positives)):
                        sent_fn.append(src[index])
                
                    for pos in range(len(pos_label[0])):
                        # if (pos_score[-(pos+1)] < list1[i]):
                        #     count += 1
                        if (pos_score[-(pos+1)] in pos_label[0]):
                            true_positives+=1
                            probas_tp = np.append(probas_tp,[scores[i][pos_score[-(pos+1)]]],axis=0)
                            sent_tp.append(src[pos_score[-(pos+1)]])
                        else:
                            probas_fp = np.append(probas_fp,[scores[i][pos_score[-(pos+1)]]],axis=0)
                            sent_fp.append(src[pos_score[-(pos+1)]])

                    total_positives+=len(pos_label[0])
                
                #####################################################
                
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)

            # print(pos_labels)
            # print(count)

            # with open("list1.txt", "wb") as fp:
            #     pickle.dump(pos_labels, fp)

            print("true positives",true_positives)
            print("true positives_margin2",true_positives_margin)
            # print("true positives_margin4",true_positives_margin2)
            print("total positives",total_positives)
            # print("probas tp",probas_tp)
            # print("probas fp",probas_fp)
            # print("probas fn",probas_fn)
            # print("sent_tp",len(sent_tp))
            # print("sent_fp",len(sent_fp))
            # print("sent_fn",len(sent_fn))

            dict_data = {
            "probas_tp": probas_tp,
            "probas_fp": probas_fp,
            "probas_fn": probas_fn,
            "sent_tp": sent_tp,
            "sent_fp": sent_fp,
            "sent_fn": sent_fn
            }
            
            loss = stats.xent()
            if loss < self.init_loss:
                # np.save('../logs/dict_data_'+str(loss)+'.npy', dict_data)
                # if self.init_loss!= 100:
                #     os.remove('../dict_data_'+str(self.init_loss)+'.npy') 
                self.init_loss = loss

            self._report_step(0, step, valid_stats=stats)

            return stats, list_save_representations



    def test(self, test_iter, step=0):
        """ Validate model.
            test_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()
        list_save_representations=[[] for i in range(2)]

        true_positives = 0
        true_positives_margin =0
        true_positives_margin2 =0
        total_positives = 0

        with torch.no_grad():

            probas_tp = np.empty((0))
            probas_fp = np.empty((0))
            probas_fn = np.empty((0))

            sent_tp = []
            sent_fp = []
            sent_fn = []

            pos_labels = []
            count = 0

            # with open("list1.txt", "rb") as fp:   # Unpickling
            #     list1 = pickle.load(fp)

            # list1 = [min(list1[i]) for i in range(len(list1))]


            for batch in test_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                src_str = batch.src_str
                
            
                sent_scores, mask, sentences_rep= self.model(src, segs, clss, mask, mask_cls)

    
                # Computation true positive
                #####################################################
                scores = (sent_scores*mask.float()).cpu().data.numpy()
                

                for i,label in enumerate(labels.cpu().data.numpy()):
                    pos_label = np.where(label==1)
                    # pos_labels.append(list(pos_label[0]))
                    pos_score = np.argsort(scores[i])

                    src = src_str[i]
                    
                    for pos in range(len(pos_label[0])+2):
                        if (pos_score[-(pos+1)] in pos_label[0]):
                            true_positives_margin+=1

                    # for pos in range(len(pos_label[0])+4):
                    #     if (pos_score[-(pos+1)] in pos_label[0]):
                    #         true_positives_margin2+=1

                    pos_positives = pos_score[-len(pos_label[0]):]
                    pos_groundtruth = pos_label[0]
                    probas_fn = np.append(probas_fn,scores[i][np.setdiff1d(pos_groundtruth, pos_positives)],axis=0)
                    
                    
                    for index in list(np.setdiff1d(pos_groundtruth, pos_positives)):
                        sent_fn.append(src[index])
                
                    for pos in range(len(pos_label[0])):
                        # if (pos_score[-(pos+1)] < list1[i]):
                        #     count += 1
                        if (pos_score[-(pos+1)] in pos_label[0]):
                            true_positives+=1
                            probas_tp = np.append(probas_tp,[scores[i][pos_score[-(pos+1)]]],axis=0)
                            sent_tp.append(src[pos_score[-(pos+1)]])
                        else:
                            probas_fp = np.append(probas_fp,[scores[i][pos_score[-(pos+1)]]],axis=0)
                            sent_fp.append(src[pos_score[-(pos+1)]])

                    total_positives+=len(pos_label[0])
                
                #####################################################
                
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)

            # print(pos_labels)
            # print(count)

            # with open("list1.txt", "wb") as fp:
            #     pickle.dump(pos_labels, fp)

            print("true positives",true_positives)
            print("true positives_margin2",true_positives_margin)
            # print("true positives_margin4",true_positives_margin2)
            print("total positives",total_positives)
            # print("probas tp",probas_tp)
            # print("probas fp",probas_fp)
            # print("probas fn",probas_fn)
            # print("sent_tp",len(sent_tp))
            # print("sent_fp",len(sent_fp))
            # print("sent_fn",len(sent_fn))

            dict_data = {
            "probas_tp": probas_tp,
            "probas_fp": probas_fp,
            "probas_fn": probas_fn,
            "sent_tp": sent_tp,
            "sent_fp": sent_fp,
            "sent_fn": sent_fn
            }
            
            loss = stats.xent()
            if loss < self.init_loss:
                # np.save('../logs/dict_data_'+str(loss)+'.npy', dict_data)
                # if self.init_loss!= 100:
                #     os.remove('../dict_data_'+str(self.init_loss)+'.npy') 
                self.init_loss = loss

            self._report_step(0, step, valid_stats=stats)

            return stats, list_save_representations



    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats, list_save_representations=None):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.src_sent_labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls


            sent_scores, mask, sentences_rep = self.model(src, segs, clss, mask, mask_cls)

            loss = self.loss(sent_scores, labels.float())
            loss = (loss * mask.float()).sum()
            (loss / loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

        return list_save_representations

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # optimizer = self.optim
        # optim_state = optimizer.state_dict()
        # generator_state_dict = real_generator.state_dict()
        # checkpoint = {
        #     'model': model_state_dict,
        #     # 'generator': generator_state_dict,
        #     'opt': self.args,
        #     'optim': self.optim,
        # }
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
