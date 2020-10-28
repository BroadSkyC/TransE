import copy
import math
import operator
import random
import time
import numpy as np
import torch

entity2id={}
relation2id={}
tripleList=[]
entityEmbeddingsDic = {}
relationEmbeddingsDic = {}

def getDataFromFile(directory):
    file1,file2,file3=directory+"train.txt",directory+"entity2id.txt",directory+"relation2id.txt"
    with open(file1,'r') as f1,open(file2,'r') as f2,open(file3,'r') as f3:
        lines1,lines2,lines3=f1.readlines(),f2.readlines(),f3.readlines()
        for line in lines2:
            line=line.split("\t")
            if len(line)!=2:continue
            entity2id[line[0]]=line[1]
        for line in lines3:
            line=line.split("\t")
            if len(line)!=2:continue
            relation2id[line[0]]=line[1]
        for line in lines1:
            triple=line.split("\t")
            if len(triple)!=3:continue
            tripleList.append([triple[0],triple[2].replace('\n',''),triple[1]])#[h,r,t]

#L2-norm
def L2(h,r,t):
    return math.sqrt(np.sum(np.square(h+r-t)))

class TransE:
    def __init__(self,tripleList,embeddingDim,learningRate,margin,nbatch):
        self.tripleList=tripleList#训练集三元组列表
        self.embeddingDim=embeddingDim#向量维度数
        self.learningRate=learningRate#学习速率
        self.margin=margin#误差修正
        self.nbatch=nbatch#每一个epochs中训练集随机划分抽样进行训练的次数
        self.entityEmbeddingsDic = {}#实体向量集
        self.relationEmbeddingsDic = {}#关系向量集

    #初始化实体集和关系集
    def initEandL(self):
        for relation in relation2id.keys():
            rEmb=np.random.uniform(-6/math.sqrt(self.embeddingDim),6/math.sqrt(self.embeddingDim),self.embeddingDim)
            initR=rEmb/np.linalg.norm(rEmb,2)
            self.relationEmbeddingsDic[relation]=initR
        for entity in entity2id.keys():
            eEmb = np.random.uniform(-6 / math.sqrt(self.embeddingDim), 6 / math.sqrt(self.embeddingDim),
                                     self.embeddingDim)
            initE = eEmb / np.linalg.norm(eEmb, 2)
            self.entityEmbeddingsDic[entity] = initE
        global entityEmbeddingsDic
        global relationEmbeddingsDic
        entityEmbeddingsDic=self.entityEmbeddingsDic
        relationEmbeddingsDic=self.relationEmbeddingsDic

    #训练
    def train(self,epochs):
        batchSize=len(self.tripleList)//self.nbatch#样本大小
        #循环训练epochs次
        for i in range(epochs):
            #归一化entity向量集
            for eneity in self.entityEmbeddingsDic.keys():
                self.entityEmbeddingsDic[eneity]=self.normalization(self.entityEmbeddingsDic[eneity])

            self.loss=0#损失量
            #进行nbatch次向量更新
            for j in range(self.nbatch):
                Sbatch=random.sample(self.tripleList,batchSize)#随机抽取batchSize个三元组
                Tbacth=[]#保存元组（原三元组，更换头实体或尾实体后得到的新三元组）
                for triple in Sbatch:
                    corruptedTriple=self.getCorruptTriple(triple)#获取新三元组（更换了头实体或尾实体）
                    pair=(triple,corruptedTriple)#将原三元组和新三元组组合起来
                    if pair not in Tbacth:
                        Tbacth.append(pair)
                self.updateEmbeddings(Tbacth)#更新实体向量集和关系向量集
        global relationEmbeddingsDic
        global entityEmbeddingsDic
        relationEmbeddingsDic=self.relationEmbeddingsDic#保存训练后的实体向量集
        entityEmbeddingsDic=self.entityEmbeddingsDic#保存训练后的关系向量集

    def updateEmbeddings(self,Tbatch):
        #复制实体和关系向量集用于下面的遍历计算
        entityEmbeddingsDicCopy=copy.deepcopy(self.entityEmbeddingsDic)
        relationEmbeddingsDicCopy=copy.deepcopy(self.relationEmbeddingsDic)

        for triple,corruptedTriple in Tbatch:
            #取copy里的vector循环更新
            #更换实体（h or t）前的三元组
            hCorrectUpdate=entityEmbeddingsDicCopy[triple[0]]
            tCorrectUpdate=entityEmbeddingsDicCopy[triple[2]]
            rUpdate=relationEmbeddingsDicCopy[triple[1]]
            #更换实体(h or t)后的三元组
            hCorruptUpdate=entityEmbeddingsDicCopy[corruptedTriple[0]]
            tCorruptUpdate=entityEmbeddingsDicCopy[corruptedTriple[2]]

            #取原始的vector计算梯度
            #更换实体(h or t)前的三元组
            hCorrect=self.entityEmbeddingsDic[triple[0]]
            tCorrect=self.entityEmbeddingsDic[triple[2]]
            r=self.relationEmbeddingsDic[triple[1]]
            #更换实体(h or t)后的三元组
            hCorrupt=self.entityEmbeddingsDic[corruptedTriple[0]]
            tCorrupt=self.entityEmbeddingsDic[corruptedTriple[2]]

            #分别计算更换实体前的、更换实体后的三元组的L2-norm
            distanceCorrect=L2(hCorrect,r,tCorrect)
            distanceCorrupt=L2(hCorrupt,r,tCorrupt)

            #计算替换实体后的损失数
            g=max(0,self.margin+distanceCorrect-distanceCorrupt)

            if g>0:
                self.loss+=g#增加总损失数
                gradPos=2*(hCorrect+r-tCorrect)
                gradNeg=2*(hCorrupt+r-tCorrupt)

                #根据算法更新向量
                if triple[0]==corruptedTriple[0]: #替换的是尾实体
                    hCorrectUpdate+=self.learningRate*(gradNeg-gradPos)
                    tCorrectUpdate+=self.learningRate*gradPos
                    tCorruptUpdate-=self.learningRate*gradNeg
                else:#替换的是头实体
                    hCorruptUpdate+=self.learningRate*gradNeg
                    hCorrectUpdate-=self.learningRate*gradPos
                    tCorrectUpdate=self.learningRate*(gradPos-gradNeg)
                rUpdate=self.learningRate*(gradNeg-gradPos)
                #归一化实体向量
                entityEmbeddingsDicCopy[triple[0]]=self.normalization(hCorrectUpdate)
                entityEmbeddingsDicCopy[triple[2]]=self.normalization(tCorrectUpdate)
                #若替换的是尾实体，更新实体集copy中尾实体对应的向量
                if triple[0]==corruptedTriple[0]:
                    entityEmbeddingsDicCopy[corruptedTriple[2]]=self.normalization(tCorruptUpdate)
                #若替换的是头实体，更新实体集copy中头实体对应的向量
                else:
                    entityEmbeddingsDicCopy[corruptedTriple[0]]=self.normalization(hCorruptUpdate)
                #更新关系集copy中关系对应的向量
                relationEmbeddingsDicCopy[corruptedTriple[1]]=rUpdate

        #保存更新后的向量集
        self.entityEmbeddingsDic=entityEmbeddingsDicCopy
        self.relationEmbeddingsDic=relationEmbeddingsDicCopy

    def normalization(self, vector):
        return vector / np.linalg.norm(vector,2)

    #get currupted triple
    def getCorruptTriple(self,triple):
        #头实体、尾实体择一替换
        seed=random.random()
        if seed>0.5:
            #替换head
            head=triple[0]
            while head==triple[0]: head=random.sample(entity2id.keys(),1)[0]
            return [head,triple[1],triple[2]]
        else:
            #替换tail
            tail=triple[2]
            while tail==triple[2]: tail=random.sample(entity2id.keys(),1)[0]
            return [triple[0],triple[1],tail]

class testTransE:
    def __init__(self,filter):
        self.testTriple=[]#测试集
        self.trainTriple=[]#训练集
        self.validTriple=[]#验证集
        self.filter=filter#判断是否过滤的标志（True-False）
        global relationEmbeddingsDic
        global entityEmbeddingsDic
        self.relationEmbeddingsDic=relationEmbeddingsDic#训练后的关系向量集
        self.entityEmbeddingsDic=entityEmbeddingsDic#训练后的实体向量集
        trainFile,testFile,validFile="FB15k\\train.txt","FB15k\\test.txt","FB15k\\valid.txt"
        with open(trainFile,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=line.split('\t')
                if len(line)!=3:continue
                self.trainTriple.append(line)
        with open(testFile,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=line.split('\t')
                if len(line)!=3:continue
                self.testTriple.append(line)
        with open(validFile,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=line.split('\t')
                if len(line)!=3:continue
                self.validTriple.append(line)

    def testRun(self):
        hits = 0
        rank_sum = 0
        num = 0

        #遍历测试集
        for triple in self.testTriple:
            num += 1#测试集中的第n个三元组
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            head_embedding = []
            tail_embedding = []
            relation_embedding = []
            tamp = []

            head_filter = []
            tail_filter = []
            #针对测试集、验证集、训练集中的特殊三元组（和当前三元组关系相同且两个实体其一相同）进行保存
            if self.filter:
                for tr in self.trainTriple:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[2] == triple[2] and tr[1] != triple[1]:
                        tail_filter.append(tr)
                for tr in self.testTriple:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[2] == triple[2] and tr[1] != triple[1]:
                        tail_filter.append(tr)
                for tr in self.validTriple:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[2] == triple[2] and tr[1] != triple[1]:
                        tail_filter.append(tr)

            #用实体集中的所有实体替换头实体，得到新三元组集，计算新三元组的h+r-t得到的向量进行L2-norm计算后得到distance
            global entityEmbeddingsDic
            global relationEmbeddingsDic
            for i, entity in enumerate(entityEmbeddingsDic.keys()):
                head_triple = [entity, triple[2], triple[1]]#替换头实体得到的新三元组
                #若已存在，过滤
                if self.filter:
                    if head_triple in head_filter:
                        continue
                head_embedding.append(entityEmbeddingsDic[head_triple[0]])
                tail_embedding.append(entityEmbeddingsDic[head_triple[2]])
                relation_embedding.append(relationEmbeddingsDic[head_triple[1].replace('\n','')])
                tamp.append(tuple(head_triple))
            #计算所有新三元组的distance
            distance = self.distance(head_embedding, relation_embedding, tail_embedding)
            #保存所有distance
            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]

            head_embedding = []
            tail_embedding = []
            relation_embedding = []
            tamp = []

            for i, tail in enumerate(entityEmbeddingsDic.keys()):
                tail_triple = [triple[0], triple[2], tail]
                if self.filter:
                    if tail_triple in tail_filter:
                        continue
                head_embedding.append(entityEmbeddingsDic[tail_triple[0]])
                tail_embedding.append(entityEmbeddingsDic[tail_triple[2]])
                relation_embedding.append(relationEmbeddingsDic[tail_triple[1].replace('\n', '')])
                tamp.append(tuple(tail_triple))

            distance = self.distance(head_embedding, relation_embedding, tail_embedding)
            for i in range(len(tamp)):
                rank_tail_dict[tamp[i]] = distance[i]
            '''            
            sorted(iterable, cmp=None, key=None, reverse=False)
            '''

            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)
            # calculate the mean_rank and hit_10
            # head data
            for i in range(0,len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            # tail rank
            for i in range(0,len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][2]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
        self.hit_10 = hits / (2 * len(self.testTriple))#平均命中率
        self.mean_rank = rank_sum / (2 * len(self.testTriple))#命中的平均排名
        return self.hit_10, self.mean_rank

    def distance(self, h, r, t):
        head = torch.from_numpy(np.array(h))
        rel = torch.from_numpy(np.array(r))
        tail = torch.from_numpy(np.array(t))

        distance = head + rel - tail
        score = torch.norm(distance, p=2, dim=1)
        return score.numpy()

if __name__ == '__main__':
    getDataFromFile("FB15K\\")
    #train
    start=time.time()
    transE=TransE(tripleList,embeddingDim=50,learningRate=0.1,margin=1,nbatch=400)
    transE.initEandL()
    print("start train...")
    transE.train(100)#epochs=100
    print("finish training,cost time "+str(time.time()-start))
    #test
    print("start test...")
    start=time.time()
    test1=testTransE(True)#filter?True.
    hit10,meanRank=test1.testRun()
    print("cost time:"+str(time.time()-start))
    print("raw entity hits@10: "+str(hit10))
    print("raw entity meanrank: "+str(meanRank))