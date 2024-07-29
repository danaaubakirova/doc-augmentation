import random
import numpy as np
from nltk.tag.stanford import StanfordPOSTagger
import RAKE
from nltk.corpus import wordnet
import re
class TextAugmenter:
    def __init__(self, stopwords_path, pos_model_path, pos_jar_path):
        self.stopwords = self.get_stopwords(stopwords_path)
        self.pos_tagger = StanfordPOSTagger(model_filename=pos_model_path, path_to_jar=pos_jar_path, java_options="-mx4000m")
        self.rake = RAKE.Rake(stopwords_path)

    @staticmethod
    def get_stopwords(path):
        with open(path, 'r') as f:
            stopwords = f.readlines()
        return [x.strip() for x in stopwords]

    def random_deletion(self, words, p):
        if len(words) <= 1:
            return words
        new_words = [word for word in words if random.uniform(0, 1) > p]
        return new_words if new_words else [random.choice(words)]

    def swap_word(self, new_words):
        if len(new_words) < 2:
            return new_words  # Early exit if there are not enough words to swap
    
        random_idx_1 = random.randint(0, len(new_words)-1)
        counter = 0
        while True:
            random_idx_2 = random.randint(0, len(new_words)-1)
            if random_idx_1 != random_idx_2:
                new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
                break
            counter += 1
            if counter > 3:
                break
        return new_words

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonyms.add(''.join([char for char in synonym if char.isalpha() or char == ' ']))
        synonyms.discard(word)
        return list(synonyms)

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        non_stopwords = [word for word in new_words if word not in self.stopwords]
        if not non_stopwords:
            return new_words
        random_word = random.choice(non_stopwords)
        synonyms = self.get_synonyms(random_word)
        if synonyms:
            random_synonym = random.choice(synonyms)
            random_idx = random.randint(0, len(new_words) - 1)
            new_words.insert(random_idx, random_synonym)

    def extract_keywords_and_POS(self, prompt):
        POS_dict = {}
        try:
            tagged_prompt = self.pos_tagger.tag(prompt.split())
        except Exception as e:
            print(f"ERROR PROMPT: {prompt}, {str(e)}")
            return False
        else:
            for word, pos in tagged_prompt:
                POS_dict[word] = pos
            keywords_dict = {}
            keywords = self.rake.run(prompt)
            for pair in keywords:
                words = pair[0].split()
                for word in words:
                    try:
                        keywords_dict[word] = POS_dict[word]
                    except:
                        pass
            return keywords_dict
    
    def get_new_keyword(self, word, pos):
        synonyms = []
        try:
            syn_lst = wordnet.synsets(word, pos)
            if len(syn_lst) == 0:
                syn_lst = wordnet.synsets(word)
        except:
            try:
                syn_lst = wordnet.synsets(word)
            except:
                return synonyms
        for syn in syn_lst:
            for l in syn.lemmas():
                if l.name().lower() != word:
                    synonyms.append(l.name().lower())
        return list(dict.fromkeys(synonyms))
        
    def single_prompt_helper(self, keywords_lst, keywords_dict, fnc, chosen_nums):
        counter = 1
        chosen_keywords_lst = []
        chosen_replacements_lst = []
        for i in range(0,len(keywords_lst)):
            if counter <= max(chosen_nums):
                keyword = keywords_lst[i]
                keyword_pos = keywords_dict[keyword][0].lower()
                if keyword_pos == 'j':
                    keyword_pos = 'a'
                candidates = fnc(keyword, keyword_pos)
                if len(candidates) != 0:
                    counter += 1
                    chosen_keywords_lst.append(keyword)
                    chosen_replacement = random.choice(candidates)
                    chosen_replacements_lst.append(chosen_replacement)
            else:
                return chosen_keywords_lst, chosen_replacements_lst
        return chosen_keywords_lst, chosen_replacements_lst


    def single_prompt_wordnet(self, prompt, nums_lst):
        original_prompt = prompt
        synonyms_prompt_str = ""  # Initialize an empty string to store the synonyms
        keywords_dict = self.extract_keywords_and_POS(prompt)
        
        if keywords_dict is False:  # Check if keyword extraction failed
            return ''
        
        keywords_lst = list(keywords_dict.keys())
        num_keywords = len(keywords_lst)
        prompt_synonym = original_prompt
        
        chosen_keywords, chosen_synonyms = self.single_prompt_helper(keywords_lst, keywords_dict, self.get_new_keyword, nums_lst)
        counter = 1
        
        for chosen_word, chosen_synonym in zip(chosen_keywords, chosen_synonyms):
            prompt_synonym = re.sub(r"\b%s\b" % chosen_word, chosen_synonym, prompt_synonym)
            if counter in nums_lst:
                synonyms_prompt_str += re.sub('_', ' ', prompt_synonym) + " "  # Concatenate the generated synonym to the string
            counter += 1
            
        return synonyms_prompt_str.strip() 

    def random_aug(self, sentence, alpha, choice):
        words = sentence.split(' ')
        words = [word for word in words if word != '']
        num_words = len(words)
        n1 = max(1, int(alpha*num_words))
    
        if choice == 'insertion':
            a_words = self.random_insertion(words, n1)
            if len(a_words) == 0 or ' '.join(a_words) == sentence:
                result_sentence = ''
            else:
                result_sentence = ' '.join(a_words)
                result_sentence = re.sub(' +', ' ', result_sentence)
    
        elif choice == 'swap':
            a_words = self.random_swap(words, n1)
            if len(a_words) == 0 or ' '.join(a_words) == sentence:
                result_sentence = ''
            else:
                result_sentence = ' '.join(a_words)
    
        elif choice == 'deletion':
            a_words = self.random_deletion(words, alpha)
            if len(a_words) == 0 or ' '.join(a_words) == sentence:
                result_sentence = ''
            else:
                result_sentence = ' '.join(a_words)
        elif choice == 'kreplacement':
            result_sentence = self.single_prompt_wordnet(sentence, [3])
        else:
            raise ValueError("Invalid choice. Choose from 'insertion','kreplacement', 'swap', or 'deletion'.")
    
        return result_sentence
