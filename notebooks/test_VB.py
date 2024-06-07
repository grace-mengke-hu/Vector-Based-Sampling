from medspacy_io.vectorizer import Vectorizer
def convert_docs_medspacyIOvec(docs:List[Doc]):
    '''
    return a dataframe: sentence	concept	y	doc_name  sentence_id
    '''
    sdf_labels=Vectorizer.docs_to_sents_df(docs, track_doc_name=True).rename(columns={"X":"sentence"})
    #
    uniq_sentSet = set(sdf_labels['sentence'].to_list())
    uniq_sentList = list(uniq_sentSet)
    sentIndexDic = {}
    for i in range(len(uniq_sentList)):
        sentIndexDic[uniq_sentList[i]] = i
    sendIDlist = []
    for s in sdf_labels['sentence'].to_list():
        sendIDlist.append(sentIndexDic[s])
    sdf_labels['sentence_id']=sendIDlist
    return sdf_labels

    

class VBSamplingSimulator(SamplingSimulator):
    def __init__(self, 
                 total_sentsDF:pd.DataFrame, 
                 total_round:int=10, 
                 modelWrapper:object=None, 
                 eval_sentsDF:pd.DataFrame, 
                 init_seed=14, 
                 faiss_index_path:str=None,
                 min_sent_length:int=10, 
                 max_retrieve:Union[int,None]=None,
                 embedding_df:pd.DataFrame=None, 
                 min_dist_diff:bool=False):
        """embedding_df to keep the generated embeddings, this will slower for large corpus, but fast when it's small.
            min_dist_diff: if true, prioritize the sentences that have distances to two centroids have smaller difference
                            if false, then prioritize the sentences that have a smaller difference of the max differences to all centroids (max distance- min distance).            
        """
        super().__init__(total_sentsDF=total_sentsDF, 
                         total_round=total_round, 
                         modelWrapper=modelWrapper, 
                         eval_sentsDF=eval_sentsDF, 
                         init_seed=init_seed, 
                         faiss_index_path=faiss_index_path,
                         min_sent_length=min_sent_length,
                         max_retrieve=max_retrieve, 
                         embedding_df=embedding_df, 
                         min_dist_diff=min_dist_diff)
        logger.debug('Loading index...')
        self.index=faiss.read_index(faiss_index_path) #vector search
        logger.debug('done')
        
        # store all the labels with sentences to be used for computing centroids later.
        # This is from medspacy_io/vectorizer.docs_to_sents_df will return 4 columns
        # column X the text of context sentences; column concepts the text of labeled concepts, y is the label; doc_name is docment id
        # This step is running before sampler  convert_docs_medspacyIOvec
        # self.sdf_labels=Vectorizer.docs_to_sents_df(total_docs, track_doc_name=True).rename(columns={"X":"sentence"})
        
        ## expecting to have 4 columns: sid, sentence, doc_name, embedding with directly constructed from doc.sents without taking entitiy labels
        #self.sid2doc_name={r['sid']:r['doc_name'] for i,r in self.embedding_df.iterrows()}
        
        # total number of sentences
        self.total_sents=self.embedding_df.shape[0]
        
        # now we only need a sentence to embedding dictionary, so no need duplicates
        self.embedding_df[['sentence', 'embedding']].drop_duplicates(subset='sentence',keep='first', inplace=True)

        #initial centroid
        self.centroid={}
        self.num_centroid=0
        if max_retrieve is None:
            self.max_retrieve=self.total_sents-self.num_per_round+1
        

    # def convert2df(self, docs: List[Doc], min_length:int=10):
    #     data={'sentence':[], 'doc_name':[]}
    #     for d in docs:
    #         sents=[str(s) for s in d.sents if len(str(s).strip())>min_length]
    #         data['sentence']+=sents
    #         data['doc_name']+=[d._.doc_name]*len(sents)
    #     return pd.DataFrame(data).rename_axis('sid').reset_index()
            

    def fit(self, sampled_sents):
        '''
        Input: sampled dataframe: sentence	concept	y	doc_name  sentence_id
        return: cetroid dictionary: label: centroid vector
                centroid number: num_centroid #number of concepts
        convert_docs_medspacyIOvec(docs:List[Doc]):
    return a dataframe: sentence	concept	y	doc_name  sentence_id
        '''
        self.centroid={t:self.compute_mean(v) for t, v in sampled_sents.groupby('y')}#group by label: t is grouped y label; v is df under this group
        self.num_centroid=len(self.centroid)
        logger.debug(f'{self.num_centroid} centroids detected from the given sampled_docs')
        
    #def fit(self, sampled_docs):
    #    '''
    #    this fit is not training a model, but try to compute the centroid for each label
    #    '''
    #    doc_names={d._.doc_name for d in sampled_docs}#document names from the sampled doc
    #    sampled_sdf=self.sdf_labels[self.sdf_labels.doc_name.isin(doc_names)] #4 columns: sentence context; concept, lable, doc
    #    self.centroid={t:self.compute_mean(v) for t, v in sampled_sdf.groupby('y')}#group by label: t is grouped y label; v is df under this group
    #    self.num_centroid=len(self.centroid)
    #    logger.debug(f'{self.num_centroid} centroids detected from the given sampled_docs')

    def compute_mean(self, sdf_single_label:pd.DataFrame):
        single_label_embeddings=sdf_single_label.merge(self.embedding_df, how='inner', on='sentence')        
        return np.mean(np.array(single_label_embeddings.embedding.tolist()), axis=0)

    def sort_dist_diff(self, distances:np.ndarray)->np.ndarray:
        if self.min_dist_diff:
            logger.debug('Compute min difference between distances to any two centroids.')
            rows, cols=distances.shape
            differences=np.abs(distances[:, None, :] -distances[:,:,None])
            triu_idx=np.triu_indices(cols, k=1)
            non_diag_diffs=differences[:, triu_idx[0], triu_idx[1]]            
            min_differences=np.amin(non_diag_diffs, axis=1)
            mask= ~np.isnan(min_differences)
            sorted_data=np.vstack((min_differences[mask], np.arange(len(distances))[mask])).T
            sorted_sids=sorted_data[:,1].astype(int)
        else:
            logger.debug('Compute max distance differences to centroids.')
            max_values=np.amax(distances, axis=1)
            min_values=np.amin(distances, axis=1)        
            
            max_differences=max_values-min_values        
    
            mask= ~np.isnan(max_differences)
            sorted_data=np.vstack((max_differences[mask], np.arange(len(distances))[mask])).T
            sorted_sids=sorted_data[:,1].astype(int)
        return sorted_sids
            
    def sample_next_round(self, sampled:pd.DataFrame, remaining:pd.DataFrame, randomly=True):
        if randomly: # first round randomly
            new_sampled, new_remaining=rand_sample_dfSentence(remaining, self.num_per_round, self.init_seed)
            updated_sampled = pd.concat([sampled, new_sampled], ignore_index=True, axis = 0)
            #sampled=sampled+new_sampled
            return updated_sampled, new_remaining

        #count the sentences when remaining or sampled are not empty
        num_df_sent_remained, list_df_UniqSentID_remained = sent_count(remaining)
        num_df_sent_sampled, list_df_UniqSentID_sampled = sent_count(sampled)

        #last round
        if self.num_per_round> num_df_sent_remained:
            updated_sampled = pd.concat([sampled, remaining], ignore_index=True, axis = 0)
            logger.info(f'Not enough documents left to sample {self.num_per_round} document. Add them all {len(remaining)} in this round.')
            return updated_sampled, pd.DataFrame()
                        
        logger.debug('Calculating centroids...')
        self.fit(sampled)
        
        logger.debug('Searching from the vector index...')
        
        #remain_doc_names=set(d._.doc_name for d in self.remaining)
        remain_sentID = remain['sid']
        
        # use numpy ops to speed up, set default value to np.nan (distance should be >=0), so that we can tell which cell has not been updated
        distances=np.full((self.total_sents, self.num_centroid), np.nan) #fill matrix size total number of sentences X num centroid
        max_retrieve=self.max_retrieve
        if self.total_sents< max_retrieve:
            max_retrieve=self.total_sents
        #estimated_needed_sents=self.num_per_round*300
        # if estimated_needed_sents<max_retrieve:
        #     max_retrieve=estimated_needed_sents
        logger.info(f'distance shape: {distances.shape}, max to retrieve {max_retrieve} sentences')
        # list distances to all centroid for each sid, then find the most uncertain ones---the difference between distances are small to at least two centroid
        for ci, (t,v) in enumerate(self.centroid.items()): 
        # this can be optimzed to limit to a smaller subset, when dealing with large corpus, no need to sort them all
            logger.debug(f'search for centroid: {t}') # t is label
            D, I=self.index.search(v.reshape(1, len(v)), max_retrieve) #faiss index search for centroid vector v
            for d, sid in zip(D[0], I[0]):
                if sid in remain_sentID:
                    distances[sid, ci]=d
                #if self.sid2doc_name[sid] in remain_doc_names: ## need to change match the two table's sentence ID
                #    distances[sid, ci]=d
                    
        # isolate the sorting logic for easier debugging
        sorted_sids=self.sort_dist_diff(distances)
        
        logger.debug('Locate the docs of these sentences')

        # from the sorted distances, find a num_per_round list of sentence ID that has the largest distance
        new_sampled_sentID = set()
        for sid in sorted_sids:
            new_sampled_sentID.add(sid)
            if len(new_sampled_sentID)>=self.num_per_round:
                break:
        new_remaining_sentID = set(remain['sid'].to_list())-new_sampled_sentID
       
        ## Now the sampled data df
        #      
        #new_sampled=set()
        #for sid in sorted_sids:
        #    new_sampled.add(self.sid2doc_name[sid])
        #    if len(new_sampled)>=self.num_per_round:
        #        break;

        #now the sampled data are DF
        new_sampled = remaining[remaining['sentence_id'].isin(list(new_sampled_sentID))]
        new_remaining = remaining[remaining['sentence_id'].isin(list(new_remaining_sentID))]
        # update the sampled dataframe
        sampled = pd.concat([sampled, new_sampled], ignore_index=True, axis = 0)
        logger.debug('Update model with new sampled data')
        return sampled, new_remaining
        
        
def compute_mean_ci(scores):
    ave=np.mean(scores)
    ci=np.percentile(scores, [2.5, 97.5])
    return ave, ci
def summarize(scores):
    summary={'precision': [], 'pl':[], 'pu': [], 'recall': [], 'rl':[], 'ru': [], 'f1':[], 'fl':[], 'fu': []}
    for s in scores:    
        for k,v in s.items():
            ave, (l, u)=compute_mean_ci(v)
            summary[k].append(ave)
            summary[k[0]+'l'].append(l)
            summary[k[0]+'u'].append(u)
    return pd.DataFrame(summary)            
            
               
