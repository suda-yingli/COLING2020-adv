import torch
import torch.nn.functional as F
import numpy as np
import pickle
import random
from config import *
from optimizer import *
#from MST import *
from nn_modules import *
import time
from instance import *
from classifier import *
from vocab import *
from dataset import *
import shutil
import os

class Parser(object):
    def __init__(self, conf):
        self._conf = conf
        self._torch_device = torch.device(self._conf.device)#'cuda:0':is GPU 0??? yli
        self._use_cuda, self._cuda_device = ('cuda' == self._torch_device.type, self._torch_device.index)
        if self._use_cuda:
            assert 0 <= self._cuda_device < 8
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._cuda_device)
            self._cuda_device = 0
        self._optimizer = None
        self._use_bucket = (self._conf.max_bucket_num > 1)
        self._train_datasets = []
        self._dev_datasets = []
        self._test_datasets = []
        self._word_dict = VocabDict('words')
        self._tag_dict = VocabDict('postags')
        # there may be more than one label dictionaries in the multi-task learning scenario
        self._label_dict = VocabDict('labels')
        self._ext_word_dict = VocabDict('ext_words')
        self._ext_word_emb_np = None

        self._all_params_requires_grad = []
        self._all_params_requires_grad_adv = []
        self._all_params_adv = []
        self._all_layers_adv = []
        self._all_params = []
        self._all_layers = []
        self._input_layer = None
        if self._conf.is_shared_lstm:
            self._lstm_layer = []
            if self._conf.is_gate_lstm:
                self._gate_lstm=[]
        else:
            self._lstm_layer = None
        if self._conf.is_adversary:
            self._classficationD = []
        self._mlp_layer = []
        self._bi_affine_layer_arc = []
        self._bi_affine_layer_label = []

        self._eval_metrics = EvalMetrics()

    # create and init all the models needed according to config
    def init_models(self):
        assert self._ext_word_dict.size() > 0 and self._ext_word_emb_np is not None and self._word_dict.size() > 0
        self._input_layer = InputLayer('input', self._conf, self._word_dict.size(),
                                       self._ext_word_dict.size(), self._tag_dict.size(), self._ext_word_emb_np)

        lstm_input_size_sh = self._conf.word_emb_dim + self._conf.tag_emb_dim
        lstm_input_size_pr = self._conf.word_emb_dim + self._conf.tag_emb_dim + self._conf.domain_emb_dim
        self._lstm_layer.append(MyLSTM('lstm_shared', \
                input_size=lstm_input_size_sh, hidden_size=self._conf.lstm_hidden_dim, \
                num_layers=self._conf.lstm_layer_num, bidirectional=True,\
                dropout_in=self._conf.lstm_input_dropout_ratio,\
                dropout_out=self._conf.lstm_hidden_dropout_ratio_for_next_timestamp, is_fine_tune=True))
        self._lstm_layer.append(MyLSTM('lstm_private', \
                input_size=lstm_input_size_pr, hidden_size=self._conf.lstm_hidden_dim, \
                num_layers=self._conf.lstm_layer_num, bidirectional=True,\
                dropout_in=self._conf.lstm_input_dropout_ratio,\
                dropout_out=self._conf.lstm_hidden_dropout_ratio_for_next_timestamp, is_fine_tune=True))

        mlp_input_size = 2 * self._conf.lstm_hidden_dim

        
        if self._conf.is_multi:
            for i in range(self._conf.domain_size):
                self._mlp_layer.append(MLPLayer('mlp'+str(i+1), activation=nn.LeakyReLU(0.1), input_size= mlp_input_size,\
                        hidden_size=2 * (self._conf.mlp_output_dim_arc + self._conf.mlp_output_dim_rel)))

                self._bi_affine_layer_arc.append(BiAffineLayer('biaffine-arc'+str(i+1), self._conf.mlp_output_dim_arc,\
                        self._conf.mlp_output_dim_arc, 1, bias_dim=(1, 0)))
                self._bi_affine_layer_label.append(BiAffineLayer('biaffine-label'+str(i+1), self._conf.mlp_output_dim_rel,\
                        self._conf.mlp_output_dim_rel, self._label_dict.size(), bias_dim=(2, 2)))
        else:
            self._mlp_layer.append(MLPLayer('mlp', activation=nn.LeakyReLU(0.1), input_size= mlp_input_size,\
                    hidden_size=2 * (self._conf.mlp_output_dim_arc + self._conf.mlp_output_dim_rel)))

            self._bi_affine_layer_arc.append(BiAffineLayer('biaffine-arc', self._conf.mlp_output_dim_arc,\
                    self._conf.mlp_output_dim_arc, 1, bias_dim=(1, 0)))
            self._bi_affine_layer_label.append(BiAffineLayer('biaffine-label', self._conf.mlp_output_dim_rel,\
                    self._conf.mlp_output_dim_rel, self._label_dict.size(), bias_dim=(2, 2)))

        assert ([] == self._all_layers)
        if self._conf.is_shared_lstm:
            for one_layer in [self._input_layer]+self._lstm_layer \
                    + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label:
                self._all_layers.append(one_layer)
            if self._conf.is_gate_lstm:
                self._gate_lstm.append(GateLSTMs('gate-lstm', activation=nn.Sigmoid(), \
                        input_size= 4 * self._conf.lstm_hidden_dim, hidden_size=2 * self._conf.lstm_hidden_dim))#input1600;output800
                self._all_layers.append(self._gate_lstm[0])
        else:
            for one_layer in [self._input_layer, self._lstm_layer] \
                    + self._mlp_layer + self._bi_affine_layer_arc + self._bi_affine_layer_label:
                self._all_layers.append(one_layer)
        if self._conf.is_adversary:
            self._classficationD.append(ClassificationD('classficationd', activation=nn.ReLU(),\
                    input_size= 2 *self._conf.lstm_hidden_dim, hidden_size=self._conf.domain_size+1))
            self._all_layers_adv.append(self._classficationD[0])
 
    # This function is useless, and will probably never be used
    def put_models_on_cpu_if_need(self):
        if not self._use_cuda:
            return
        # If the nnModule is on GPU, then .to(torch.device('cpu')) will lead to the unnecessary use of gpu:0
        for one_layer in self._all_layers:
            one_layer.to(self._cpu_device)

    def put_models_on_gpu_if_need(self):
        if not self._use_cuda:
            return
        for one_layer in self._all_layers:
            one_layer.cuda(self._cuda_device) # the argument can be removed
        self._all_layers_adv[0].cuda(self._cuda_device) # the argument can be removed

    def collect_all_params(self):
        assert([] == self._all_params)
        for one_layer in self._all_layers:
            for one_param in one_layer.parameters():
                self._all_params.append(one_param)
        assert([] == self._all_params_requires_grad)
        self._all_params_requires_grad = [param for param in self._all_params if param.requires_grad]
        for one_param in self._all_layers_adv[0].parameters():
            self._all_params_adv.append(one_param)
        self._all_params_requires_grad_adv = [param for param in self._all_params_adv if param.requires_grad]




    def run(self):
        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.train_files, self._train_datasets,
                                        inst_num_max=self._conf.inst_num_max)#trainfilename,[],-1
            if self._conf.is_dictionary_exist is False:
                print("create dict...")
                for dataset in self._train_datasets:
                    self.create_dictionaries(dataset, self._label_dict)
                self.save_dictionaries(self._conf.dict_dir)
                print("create dict done")
                return

        self.load_dictionaries(self._conf.dict_dir)

        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.dev_files, self._dev_datasets,
                                        inst_num_max=self._conf.inst_num_max)

        self.open_and_load_datasets(self._conf.test_files, self._test_datasets,
                                    inst_num_max=self._conf.inst_num_max)

        print('numeralizing [and pad if use-bucket] all instances in all datasets', flush=True)
        for dataset in self._train_datasets + self._dev_datasets + self._test_datasets:#all datasets in one [].
            self.numeralize_all_instances(dataset, self._label_dict)#将所有的instence都转换成数字形式
            if self._use_bucket:
                self.pad_all_inst(dataset)#对桶中的实例进行padding

        print('init models', flush=True)
        self.init_models()

        if self._conf.is_train: 
            self.put_models_on_gpu_if_need()   
            self.collect_all_params()
            assert self._optimizer is None
            self._optimizer = Optimizer(self._all_params_requires_grad, self._conf)
            self._optimizer_adv = Optimizer(self._all_params_requires_grad_adv, self._conf,is_adv=True)
            self.train()
            return

        assert self._conf.is_test
        self.load_model(self._conf.model_dir, self._conf.model_eval_num)
        self.put_models_on_gpu_if_need()   
        for dataset in self._test_datasets:
            self.evaluate(dataset, output_file_name='./'+dataset.file_name_short+'.out')
            self._eval_metrics.compute_and_output(self._test_datasets[0], self._conf.model_eval_num)
            self._eval_metrics.clear()

    def forward(self, words, ext_words, tags, lstm_masks, domains, domain_id):
        is_training = self._input_layer.training

        input_out_sh, input_out_pr = self._input_layer(words, ext_words, tags, domains)

        # -> length batch dim 
        input_out_sh = input_out_sh.transpose(0, 1)
        input_out_pr = input_out_pr.transpose(0, 1)

        #print("init lstm_masks",lstm_masks.size())#[544,10]
        masks = lstm_masks
        lstm_masks = torch.unsqueeze(lstm_masks.transpose(0, 1), dim=2)#[10,544,1]
        #print("input_out",input_out.size())
        shared_lstm_out=self._lstm_layer[0](input_out_sh, lstm_masks, initial=None, is_training=is_training)#[10,544,400]
        private_lstm_out=self._lstm_layer[1](input_out_pr, lstm_masks, initial=None, is_training=is_training)#[10,544,400]
        diff = self.diff_module(lstm_masks, shared_lstm_out, private_lstm_out) 
        if self._conf.is_gate_lstm:
            lstm_out = self._gate_lstm[0](shared_lstm_out, private_lstm_out)
        else:
            #lstm_out = torch.cat((shared_lstm_out, private_lstm_out), dim=2)
            lstm_out = shared_lstm_out+ private_lstm_out

        if is_training:
            lstm_out = drop_sequence_shared_mask(lstm_out, self._conf.mlp_input_dropout_ratio)

        if self._conf.is_multi:
            arc_scores, label_scores = self.mlp_biaffine_module(domain_id, lstm_out, is_training)
        else:
            arc_scores, label_scores = self.mlp_biaffine_module(1, lstm_out, is_training)
        
        classficationd = self.classfication_module(shared_lstm_out, masks,is_training)
        return arc_scores, label_scores, classficationd, diff

    def classfication_module(self,lstm_out,lstm_masks,is_training):
        classficationd = self._classficationD[0](lstm_out,lstm_masks,is_training)
        classficationd = classficationd.transpose(0, 1)
        return classficationd

    def mlp_biaffine_module(self, domain_id, lstm_out, is_training):
        mlp_out = self._mlp_layer[domain_id-1](lstm_out)
        if is_training:
            mlp_out = drop_sequence_shared_mask(mlp_out, self._conf.mlp_output_dropout_ratio)
        mlp_out = mlp_out.transpose(0, 1)
        mlp_arc_dep, mlp_arc_head, mlp_label_dep, mlp_label_head = \
                torch.split(mlp_out, [self._conf.mlp_output_dim_arc, self._conf.mlp_output_dim_arc,\
                self._conf.mlp_output_dim_rel, self._conf.mlp_output_dim_rel], dim=2)
        arc_scores = self._bi_affine_layer_arc[domain_id-1](mlp_arc_dep, mlp_arc_head)
        arc_scores = torch.squeeze(arc_scores, dim=3)
        label_scores = self._bi_affine_layer_label[domain_id-1](mlp_label_dep, mlp_label_head)
        return arc_scores, label_scores
        
    def diff_module(self,lstm_masks,shared_lstm_out,private_lstm_out):
        length,batch,dim=shared_lstm_out.size()
        lstm_mask1 = lstm_masks.expand(length, batch, dim)
        b = torch.bmm(torch.mul(shared_lstm_out,lstm_mask1).transpose(1,2),torch.mul(private_lstm_out,lstm_mask1))
        diff = torch.mul(b,b)
        diff1 = torch.sum(diff,dim=2)
        diff2 = torch.sum(diff1)
        return diff2

    @staticmethod
    def compute_loss(arc_scores, label_scores, gold_arcs, gold_labels, total_word_num, one_batch):
        batch_size, len1, len2 = arc_scores.size()
        assert(len1 == len2)

        # gold_arcs, gold_labels: batch_size max-len
        penalty_on_ignored = []  # so that certain scores are ignored in computing cross-entropy loss
        for inst in one_batch:
            length = inst.size()
            penalty = arc_scores.new_tensor([0.] * length + [-1e10] * (len1 - length))
            penalty_on_ignored.append(penalty.unsqueeze(dim=0))
        penalty_on_ignored = torch.stack(penalty_on_ignored, 0)
        arc_scores = arc_scores + penalty_on_ignored

        arc_loss = F.cross_entropy(
            arc_scores.view(batch_size * len1, len2), gold_arcs.view(batch_size * len1),
            ignore_index=ignore_id_head_or_label, size_average=False)

        batch_size2, len12, len22, label_num = label_scores.size()
        assert batch_size2 == batch_size and len12 == len2 and len22 == len2

        # Discard len2 dim: batch len1 L
        label_scores_of_concern = arc_scores.new_full((batch_size, len1, label_num), 0)  # discard len2 dim

        scores_one_sent = [label_scores[0][0][0]] * len1
        for i_batch, (scores, arcs) in enumerate(zip(label_scores, gold_arcs)):
            for i in range(one_batch[i_batch].size()):
                scores_one_sent[i] = scores[i, arcs[i]]  # [mod][gold-head]: L * float
            label_scores_of_concern[i_batch] = torch.stack(scores_one_sent, dim=0)

        rel_loss = F.cross_entropy(label_scores_of_concern.view(batch_size * len1, label_num),
                                   gold_labels.view(batch_size * len1),
                                   ignore_index=ignore_id_head_or_label, size_average=False)

        loss = (arc_loss + rel_loss) / total_word_num
        return loss

    '''
    @staticmethod
    def adversary_loss(classficationd, domains,total_word_num):
        batch_size, len1, len2 = classficationd.size()
        #classficationd = F.softmax(classficationd)
        adv_loss = F.cross_entropy(classficationd.contiguous().view(batch_size * len1, len2),\
                domains.view(batch_size * len1), ignore_index = 0)
        adv_loss = adv_loss / total_word_num
        return adv_loss
    '''

    def train(self):

        update_step_cnt, eval_cnt, best_eval_cnt, best_accuracy = 0, 0, 0, 0.
        self._eval_metrics.clear()
        self.set_training_mode(is_training=True)
        domain, domain_src, domain_tgt = 0, 0, 0
        while True:
            #print("len(self._train_datasets)",len(self._train_datasets))
            if(len(self._train_datasets) > 1):
                #print("self._train_datasets[domain].file_name_short",self._train_datasets[domain].file_name_short)
                #print("self._train_datasets[domain].domain_id",self._train_datasets[domain].domain_id)
                #print("self._train_datasets[domain].batch_num",self._train_datasets[domain].batch_num)
                if (domain == 0 and domain_tgt < self._train_datasets[0].batch_num):
                #if (domain == 0 and domain_src < self._train_datasets[1].batch_num):
                    inst_num, loss = self.train_or_eval_one_batch(self._train_datasets[domain], is_training=True)
                    domain_tgt += 1
                    domain = 1
                else:
                    inst_num, loss = self.train_or_eval_one_batch(self._train_datasets[domain], is_training=True)
                    domain_src += 1
                    #domain = 0
                    if domain_tgt == self._train_datasets[0].batch_num:
                        domain = 1
                    else:
                        domain = 0
            else:
                inst_num, loss = self.train_or_eval_one_batch(self._train_datasets[0], is_training=True)
            assert inst_num > 0
            assert loss is not None
            loss.backward()
            nn.utils.clip_grad_norm_(self._all_params_requires_grad, max_norm=self._conf.clip)
            nn.utils.clip_grad_norm_(self._all_params_requires_grad_adv, max_norm=self._conf.clip)
            self._optimizer.step()
            self._optimizer_adv.step()
            self._optimizer.zero_grad()
            self._optimizer_adv.zero_grad()
            self.zero_grad()
 
            update_step_cnt += 1
            #print("update_step_cnt ",update_step_cnt)

            if 0 == update_step_cnt % self._conf.eval_every_update_step_num:
                eval_cnt += 1
                domain_src, domain_tgt, domain, train_size = 0, 0, 0, 0
                self._eval_metrics.compute_and_output(self._train_datasets[0], eval_cnt)
                self._eval_metrics.clear()

                print("begin evaluate")
                self.evaluate(self._dev_datasets[0])
                self._eval_metrics.compute_and_output(self._dev_datasets[0], eval_cnt)
                current_las = self._eval_metrics.las
                self._eval_metrics.clear()

                if best_accuracy < current_las - 1e-3:
                    if eval_cnt > self._conf.save_model_after_eval_num:
                        if best_eval_cnt > self._conf.save_model_after_eval_num:
                            self.del_model(self._conf.model_dir, best_eval_cnt)
                        self.save_model(self._conf.model_dir, eval_cnt)
                        self.evaluate(self._test_datasets[0], output_file_name=None)
                        self._eval_metrics.compute_and_output(self._test_datasets[0], eval_cnt)
                        self._eval_metrics.clear()

                    best_eval_cnt = eval_cnt
                    best_accuracy = current_las

                self.set_training_mode(is_training=True)

            if (best_eval_cnt + self._conf.train_stop_after_eval_num_no_improve < eval_cnt) or \
                    (eval_cnt > self._conf.train_max_eval_num):
                break

    def train_or_eval_one_batch(self, dataset, is_training):
        one_batch, total_word_num, max_len = dataset.get_one_batch(rewind=is_training)
        # NOTICE: total_word_num does not include w_0
        if len(one_batch) == 0:
            print("one_batch is none" + dataset.file_name_short)
            return 0, None
        words, ext_words, tags, gold_heads, gold_labels, lstm_masks, domains = \
            self.compose_batch_data_variable(one_batch, max_len)
        if self._conf.is_adversary:
            arc_scores, label_scores, classficationd, diff = self.forward(words, ext_words, tags, lstm_masks, domains, dataset.domain_id)
        else:
            arc_scores, label_scores, diff = self.forward(words, ext_words, tags, lstm_masks, domains, dataset.domain_id)
        #print("onebatch len:",len(one_batch))
        self.decode(arc_scores, label_scores, one_batch, self._label_dict)
        final_loss = Parser.compute_loss(arc_scores, label_scores, gold_heads, gold_labels,
                                   total_word_num, one_batch)
        self.compute_accuracy(one_batch, self._eval_metrics)
        self._eval_metrics.loss_accumulated += final_loss.item()
        if self._conf.is_adversary:
            #adv_loss = Parser.adversary_loss(classficationd, domains,total_word_num)
            adv_loss = ClassificationD.adversary_loss(classficationd, domains,total_word_num)
            adv_loss = self._conf.adversary_lambda_loss * adv_loss
            print("parser loss:",final_loss)
            print("adversary loss:",adv_loss)
            self._eval_metrics.loss_accumulated += adv_loss.item()
            ClassificationD.compute_accuray(classficationd, domains)
            final_loss += adv_loss
        if self._conf.is_diff_loss:
            diff_loss = self._conf.diff_bate_loss*diff/total_word_num
            self._eval_metrics.loss_accumulated += diff_loss.item()
            final_loss += diff_loss
            print("diff_loss",diff_loss)
        return len(one_batch), final_loss

    def evaluate(self, dataset, output_file_name=None):
        self.set_training_mode(is_training=False)
        while True:
            inst_num, loss = self.train_or_eval_one_batch(dataset, is_training=False)
            if 0 == inst_num:
                break
            assert loss is not None

        if output_file_name is not None:
            with open(output_file_name, 'w', encoding='utf-8') as out_file:
                all_inst = dataset.all_inst
                for inst in all_inst:
                    inst.write(out_file)

    @staticmethod
    def decode(arc_scores, label_scores, one_batch, label_dict):
        # detach(): Returns a new Tensor, detached from the current graph.
        arc_scores = arc_scores.detach().cpu().numpy()
        label_scores = label_scores.detach().cpu().numpy()

        for (arc_score, label_score, inst) in zip(arc_scores, label_scores, one_batch):
            arc_pred = np.argmax(arc_score, axis=1)   # mod-head order issue. BE CAREFUL
            label_score_of_concern = label_score[np.arange(inst.size()), arc_pred[:inst.size()]]
            label_pred = np.argmax(label_score_of_concern, axis=1)
            Parser.set_predict_result(inst, arc_pred, label_pred, label_dict)

    def create_dictionaries(self, dataset, label_dict):
        all_inst = dataset.all_inst
        for inst in all_inst:
            for i in range(1, inst.size()):
                self._word_dict.add_key_into_counter(inst.words_s[i])
                self._tag_dict.add_key_into_counter(inst.tags_s[i])
                if inst.heads_i[i] != ignore_id_head_or_label:
                    label_dict.add_key_into_counter(inst.labels_s[i])

    def numeralize_all_instances(self, dataset, label_dict):
        all_inst = dataset.all_inst
        for inst in all_inst:
            for i in range(0, inst.size()):
                inst.words_i[i] = self._word_dict.get_id(inst.words_s[i])
                inst.ext_words_i[i] = self._ext_word_dict.get_id(inst.words_s[i])
                inst.tags_i[i] = self._tag_dict.get_id(inst.tags_s[i])
                if inst.heads_i[i] != ignore_id_head_or_label:
                    inst.labels_i[i] = label_dict.get_id(inst.labels_s[i])

    def load_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path)
        self._word_dict.load(path + self._word_dict.name, cutoff_freq=self._conf.word_freq_cutoff,  
                             default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self._tag_dict.load(path + self._tag_dict.name,
                            default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self._label_dict.load(path + self._label_dict.name, default_keys_ids=())

        self._ext_word_dict.load(self._conf.ext_word_dict_full_path,
                                 default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self.load_ext_word_emb(self._conf.ext_word_emb_full_path,
                               default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))

    def save_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path) is False
        os.mkdir(path)
        self._word_dict.save(path + self._word_dict.name)
        self._tag_dict.save(path + self._tag_dict.name)
        self._label_dict.save(path + self._label_dict.name)

    def load_ext_word_emb(self, full_file_name, default_keys_ids=()):
        assert os.path.exists(full_file_name)
        with open(full_file_name, 'rb') as f:
            self._ext_word_emb_np = pickle.load(f)
        dim = self._ext_word_emb_np.shape[1]
        assert dim == self._conf.word_emb_dim
        for i, (k, v) in enumerate(default_keys_ids):
            assert(i == v)
        pad_and_unk_embedding = np.zeros((len(default_keys_ids), dim), dtype=data_type)
        self._ext_word_emb_np = np.concatenate([pad_and_unk_embedding, self._ext_word_emb_np])
        self._ext_word_emb_np = self._ext_word_emb_np / np.std(self._ext_word_emb_np)

    @staticmethod
    def del_model(path, eval_num):
        path = os.path.join(path, 'models-%d/' % eval_num)
        assert os.path.exists(path)
        # os.rmdir(path)
        shutil.rmtree(path)
        print('Delete model %s done.' % path)

    def load_model(self, path, eval_num):
        path = os.path.join(path, 'models-%d/' % eval_num)
        assert os.path.exists(path)
        for layer in self._all_layers:
            # Without 'map_location='cpu', you may find the unnecessary use of gpu:0, unless CUDA_VISIBLE_DEVICES=6 python $exe ... 
            layer.load_state_dict(torch.load(path + layer.name, map_location='cpu'))
            # layer.load_state_dict(torch.load(path + layer.name)) 
        print('Load model %s done.' % path)

    def save_model(self, path, eval_num):
        path = os.path.join(path, 'models-%d/' % eval_num)
        # assert os.path.exists(path) is False
        if os.path.exists(path) is False:
            os.mkdir(path)
        for layer in self._all_layers:
            torch.save(layer.state_dict(), path + layer.name)
        print('Save model %s done.' % path)

    def open_and_load_datasets(self, file_names, datasets, inst_num_max):
        assert len(datasets) == 0
        names = file_names.strip().split(':')
        assert len(names) > 0
        for name in names:
            datasets.append(Dataset(name, max_bucket_num=self._conf.max_bucket_num,
                                    word_num_one_batch=self._conf.word_num_one_batch,
                                    sent_num_one_batch=self._conf.sent_num_one_batch,
                                    inst_num_max=inst_num_max))#80,5000,200,-1

    @staticmethod
    def set_predict_result(inst, arc_pred, label_pred, label_dict):
        # assert arc_pred.size(0) == inst.size()
        for i in np.arange(1, inst.size()):
            inst.heads_i_predict[i] = arc_pred[i]
            inst.labels_i_predict[i] = label_pred[i]
            inst.labels_s_predict[i] = label_dict.get_str(inst.labels_i_predict[i])

    @staticmethod
    def compute_accuracy_one_inst(inst, eval_metrics):
        (a, b, c) = inst.eval()
        eval_metrics.word_num += inst.word_num()
        eval_metrics.word_num_to_eval += a
        eval_metrics.word_num_correct_arc += b
        eval_metrics.word_num_correct_label += c

    @staticmethod
    def compute_accuracy(one_batch, eval_metrics):
        eval_metrics.sent_num += len(one_batch)
        for inst in one_batch:
            Parser.compute_accuracy_one_inst(inst, eval_metrics)

    def set_training_mode(self, is_training=True):
        for one_layer in self._all_layers:
            one_layer.train(mode=is_training)

    def zero_grad(self):
        for one_layer in self._all_layers:
            one_layer.zero_grad()


    def pad_all_inst(self, dataset):
        for (max_len, inst_num_one_batch, this_bucket) in dataset.all_buckets:
            for inst in this_bucket:
                assert inst.lstm_mask is None
                inst.words_i, inst.ext_words_i, inst.tags_i, inst.heads_i, inst.labels_i, inst.lstm_mask, inst.domains_i = \
                    self.pad_one_inst(inst, max_len)

    def pad_one_inst(self, inst, max_sz):
        sz = inst.size()
        assert len(inst.words_i) == sz
        assert max_sz >= sz
        pad_sz = (0, max_sz - sz)
        return np.pad(inst.words_i, pad_sz, 'constant', constant_values=0), \
               np.pad(inst.ext_words_i, pad_sz, 'constant', constant_values=0), \
               np.pad(inst.tags_i, pad_sz, 'constant', constant_values=0), \
               np.pad(inst.heads_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label), \
               np.pad(inst.labels_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label), \
               np.pad(np.ones(sz, dtype=data_type), pad_sz, 'constant', constant_values=0), \
               np.pad(inst.domains_i, pad_sz, 'constant', constant_values=0)

    def compose_batch_data_variable(self, one_batch, max_len):
        words, ext_words, tags, heads, labels, lstm_masks, domains = [], [], [], [], [], [], []
        for inst in one_batch:
            if self._use_bucket:
                words.append(inst.words_i)
                ext_words.append(inst.ext_words_i)
                tags.append(inst.tags_i)
                heads.append(inst.heads_i)
                labels.append(inst.labels_i)
                lstm_masks.append(inst.lstm_mask)
                domains.append(inst.domains_i)
            else:
                ret = self.pad_one_inst(inst, max_len)
                words.append(ret[0])
                ext_words.append(ret[1])
                tags.append(ret[2])
                heads.append(ret[3])
                labels.append(ret[4])
                lstm_masks.append(ret[5])
                domains.append(ret[6])
        # dim: batch max-len
        words, ext_words, tags, heads, labels, lstm_masks, domains = \
            torch.from_numpy(np.stack(words, axis=0)), torch.from_numpy(np.stack(ext_words, axis=0)), \
            torch.from_numpy(np.stack(tags, axis=0)), torch.from_numpy(np.stack(heads, axis=0)), \
            torch.from_numpy(np.stack(labels, axis=0)), torch.from_numpy(np.stack(lstm_masks, axis=0)), \
            torch.from_numpy(np.stack(domains, axis=0))

        # MUST assign for Tensor.cuda() unlike nn.Module
        if self._use_cuda:
            words, ext_words, tags, heads, labels, lstm_masks, domains = \
               words.cuda(self._cuda_device), ext_words.cuda(self._cuda_device), \
               tags.cuda(self._cuda_device), heads.cuda(self._cuda_device), \
               labels.cuda(self._cuda_device), lstm_masks.cuda(self._cuda_device), \
               domains.cuda(self._cuda_device)
        return words, ext_words, tags, heads, labels, lstm_masks, domains


class EvalMetrics(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.sent_num = 0
        self.word_num = 0
        self.word_num_to_eval = 0
        self.word_num_correct_arc = 0
        self.word_num_correct_label = 0
        self.uas = 0.
        self.las = 0.
        self.loss_accumulated = 0.
        self.start_time = time.time()
        self.time_gap = 0.

    def compute_and_output(self, dataset, eval_cnt):
        assert self.word_num > 0
        self.uas = 100. * self.word_num_correct_arc / self.word_num_to_eval
        self.las = 100. * self.word_num_correct_label / self.word_num_to_eval
        self.time_gap = float(time.time() - self.start_time)
        print("%30s(%5d): loss=%.3f las=%.3f, uas=%.3f, %d (%d) words, %d sentences, time=%.3f [%s]" %
              (dataset.file_name_short, eval_cnt, self.loss_accumulated, self.las, self.uas,
               self.word_num_to_eval, self.word_num, self.sent_num, self.time_gap, get_time_str()), flush=True)
