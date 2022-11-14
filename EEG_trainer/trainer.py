# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:36:15 2021

@author: phantom
"""

from torch.autograd import Variable
from torch import optim
import torch
import shutil
import copy

class Trainer:
    def __init__(self, model, loss, source_train_loader, target_test_loader, args):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.source_train_loader = source_train_loader
        self.target_test_loader = target_test_loader
        
        # Loss function and Optimizer
        self.loss = loss # MMD loss
        self.optimizer = self.get_optimizer()#Adam
        self.schedular = torch.optim.lr_scheduler.StepLR(self.optimizer, 20, gamma = 0.1, last_epoch = -1)
        self.best_model_params = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.best_loss = 1000000
        self.best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())
        
        #early stop
        self.max_train_acc=0
        self.early_stop_timer=0

    def train(self):
        '''
        function of training
        '''
        train_acc_list = []
        train_loss_list = []
        for epoch in range(self.args.start_epoch, self.args.start_epoch+self.args.num_epochs):
            train_loss = 0.0
            train_acc = 0.0
            for batch_idx, (data, target) in enumerate(self.source_train_loader):
                self.model.train()
                if self.args.cuda:
                    data = data.cuda()
                    target = target.cuda()
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                
                #predict
                output, feature = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.data.item()
                index = output.cpu().data.numpy().argmax(axis = 1)
                label =target.cpu().data.numpy()[:, 0]
                train_acc += sum(index == label)
                
            train_acc /= len(self.source_train_loader.dataset)
            train_loss /= len(self.source_train_loader.dataset)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            
            #early stop
            if train_acc > self.max_train_acc:
                self.max_train_acc = train_acc
                self.early_stop_timer = 0
            else:
                self.early_stop_timer += 1
            if self.early_stop_timer >= self.args.early_stop_patience:
                print("\nearly stop\n")
                break;
                
            #print results
            print("epoch : {}\ntrain : acc {:.4} | loss {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,self.early_stop_timer))
            
            #test model
            if self.args.TestWhenTraining == 1:
                test_acc, test_loss, index_list, target_list=self.test()
                TP,FP,TN,FN=0,0,0,0
                for i in range(len(index_list)):
                    for j in range(len(index_list[i])):
                        if index_list[i][j]==1 and target_list[i][j]==1:
                            TP+=1
                        elif index_list[i][j]==0 and target_list[i][j]==1:
                            FN+=1
                        elif index_list[i][j]==0 and target_list[i][j]==0:
                            TN+=1
                        else:
                            FP+=1
                print("test : TP {} | FN {} | TN {} | FP {} | sen {:.4%} | spe {:.4%} | acc {:.4%}\n".format(TP,FN,TN,FP,TP/(TP+FN),TN/(TN+FP),(TP+TN)/(TP+FN+TN+FP)))

        return self.model,train_acc_list,train_loss_list
        
    def test(self):
        '''
        function of testing
        '''
        target_list=[]
        index_list=[]
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.target_test_loader):
                if self.args.cuda:
                    data = data.cuda()
                    target = target.cuda()
                    
                #model predict
                output, feature = self.model(data)
                loss = self.loss(output, target)
                test_loss += loss.data.item()
                index = output.cpu().data.numpy().argmax(axis = 1)
                label =target.cpu().data.numpy()[:, 0]
                test_acc += sum(index == label)
                
                target_list.append(target)
                index_list.append(index)
            
        test_acc /= len(self.target_test_loader.dataset)
        test_loss /= len(self.target_test_loader.dataset)
        self.model.train()
        return test_acc, test_loss, index_list, target_list#index.reshape(5, 20)

    def test_on_trainings_set(self):
        print('testing...')
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.source_train_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar, z = self.model(data)
            test_loss += self.loss(recon_batch, data, mu, logvar).data[0]
            '''
            if i % 50 == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(-1, 3, 32, 32)[:n]])
                self.summary_writer.add_image('training_set/image', comparison, i)
            '''
        test_loss /= len(self.target_test_loader.dataset)
        print('====> Test on training set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def get_optimizer(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)
        #return optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.args.learning_rate)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            #param_group['lr'] = param_group['lr']*0.2
    
    def adjust_learning_rate_step(self):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        #learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in self.optimizer.param_groups:
            #param_group['lr'] = learning_rate
            param_group['lr'] = param_group['lr']*0.99

    def save_checkpoint(self, epoch, state, is_best=False, filename='checkpoint{}.pth'):
        '''
        a function to save checkpoint of the training
        :param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        :param is_best: boolean to save the checkpoint aside if it has the best score so far
        :param filename: the name of the saved file
        '''
        torch.save(state, self.args.checkpoint_dir + filename.format(epoch))
        if is_best:
            shutil.copyfile(self.args.checkpoint_dir + filename,
                            self.args.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.args.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch']))
        except:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.checkpoint_dir))
