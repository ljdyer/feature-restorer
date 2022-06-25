        # if self.spaces:
        #     print(datapoint)
        #     print(type(datapoint))
        #     words = datapoint.split()
        #     try:
        #         substrs = []
        #     except:
        #         print('Huh?')
        #         quit()
        #     for i in range(len(words)):
        #         try:
        #             substrs.append(' '.join(words[i:]))
        #         except:
        #             print(words[i])
        #             quit()
        #     # substrs = [ for i in range(len(words))]
        #     substrs = [s for s in substrs
        #                if len(re.sub(rf"[{''.join(self.feature_chars)}]", '', s)) > self.seq_length]
        #     samples = []
        #     for substr in substrs:
        #         this_sample = self.substr_to_sample(substr)
        #         if this_sample:
        #             samples.append(this_sample)
        #     return samples
        # else:
        #     raise ValueError('Not implemented yet when spaces=False')
