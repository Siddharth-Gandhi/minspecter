"""
The script reads the coview/co-citation data and returns them as triplet format
i.e., ['Query_paper', ('Positive_paper_id', num-coviews), ('Negative_paper_id', num_coviews)
"""
import logging
import logging.config
import math
import operator
import random
from typing import Dict, Iterator, List, Tuple

import numpy as np

logging.config.fileConfig(fname="logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__file__)  # pylint: disable=invalid-name
# logger.setLevel(logging.INFO)

# positive example: high coviews
# easy: those that have 0 coviews
# hard: those that have non-zero coviews but have smaller coviews than the positive
# note - hard will only be possible if there are at least 2 non-zero coview papers

np.random.seed(321)
random.seed(321)

DIRECT_CITATION=5
CO_CITATION=1
NO_CITATION=0

class MinTripletSampler:
    """ Class to generate triplets"""

    def __init__(self,
                 paper_ids: List[str],
                 coviews: Dict[str, List[Tuple[str, int]]],
                #  margin_fraction: float,
                #  margin: int,
                 samples_per_query: int,
                 ratio_hard_easy_neg: float) -> None:
        """
        Args:
            paper_ids: list of all paper ids
            coviews: a dictionary where keys are paper ids and values are lists of [paper_id, count] pairs
                showing the number of coviews for each paper
            margin_fraction: minimum margin of co-views between positive and negative samples
            samples_per_query: how many samples for each query
            query_list: list of query ids. If None, it will return for all papers in the coviews file
            ratio_hard_negatives: ratio of negative samples selected from difficult and easy negatives
                respectively. Difficult negative samples are those that are also coviewed but less than
                a positive sample. Easy negative samples are those with zero coviews
        """
        self.paper_ids = paper_ids
        self.paper_ids_set = set(paper_ids)
        self.coviews = coviews
        self.samples_per_query = samples_per_query
        self.ratio_hard_easy_neg = ratio_hard_easy_neg

    def _get_triplet(self, query):
        if query not in self.coviews:
            return
        # self.coviews[query] is a dictionary of format {paper_id: {count: 1, frac: 1}}
        candidates = [(k, v['count']) for k, v in self.coviews[query].items()]
        candidates = sorted(candidates, key=operator.itemgetter(1), reverse=True)
        # if len(candidates) < self.samples_per_query:
        #     # logger.warning(f'Not enough candidates for query {query}')
        #     print(f'Not enough candidates for query {query}')
        #     return None

        # get positive, easy negative and hard negative samples
        pos_citations = [candidate for candidate in candidates if candidate[1] == DIRECT_CITATION]
        easy_neg_citations = list(self.paper_ids_set.difference(
            {candidate[0] for candidate in candidates}
        )) + [query]
        hard_neg_citations = [candidate for candidate in candidates if candidate[1] == CO_CITATION]


        # determine number of positive, easy anegative and hard negative samples
        num_pos, num_hard_neg = len(pos_citations), len(hard_neg_citations)
        num_pos = min(num_pos, self.samples_per_query)
        num_hard_neg = min(num_hard_neg, math.floor(self.ratio_hard_easy_neg * self.samples_per_query))
        num_easy_neg = self.samples_per_query - num_hard_neg

        # randomly sample from all lists
        random.shuffle(pos_citations)
        pos_citations = pos_citations[:num_pos]
        # easy_neg_citations = list(np.random.choice(easy_neg_citations, size=num_easy_neg, replace=False))
        temp_set = set()
        counter = 0

        # WARNING: this works because I assume there are large enough number of easy negatives, but this is better than sampling from a huge list. Might go in infinite loop.
        while True:
            idx = np.random.randint(0, len(easy_neg_citations))
            item = easy_neg_citations[idx]
            if item in temp_set:
                continue
            temp_set.add(item)
            counter += 1
            if counter == num_easy_neg:
                break
        easy_neg_citations = list(temp_set)

        easy_neg_citations = [(citation, NO_CITATION) for citation in easy_neg_citations]
        random.shuffle(hard_neg_citations)
        hard_neg_citations = hard_neg_citations[:num_hard_neg]

        neg_citations = easy_neg_citations + hard_neg_citations

        triplets = []
        for i in range(num_pos):
            neg = (neg_citations[i], NO_CITATION) if isinstance(neg_citations[i], str) else neg_citations[i]
            pos = pos_citations[i]
            triplet =  [query, pos, neg]
            triplets.append(triplet)

        return triplets




    def generate_triplets(self, query_ids: List[str]) -> Iterator[List[Tuple]]:
        """ Generate triplets from a list of query ids

        This generates a list of triplets each query according to:
            [(query_id, (positive_id, coviews), (negative_id, coviews)), ...]
        The upperbound of the list length is according to self.samples_per_query

        Args:
            query_ids: a list of query paper ids

        Returns:
            Lists of tuples
                The format of tuples is according to the triples
        """
        logger.info('Generating triplets for queries')
        count_skipped = 0  # count how many of the queries are not in coveiws file
        count_success = 0

        for query in query_ids:  # tqdm.tqdm(query_ids):
            results = self._get_triplet(query)
            if results:
                count_success += 1
                yield from results
            else:
                count_skipped += 1
        print(f'Done generating triplets, #successful queries: {count_success}, #skipped queries: {count_skipped}')
        print(f'Total #triplets: {count_success * self.samples_per_query}')


def main():
    import cProfile
    import json
    import pstats

    training_folder = 'data/training'

    with open(f'{training_folder}/train copy.txt', 'r', encoding='utf-8') as f:
        train = f.read().splitlines()

    with open(f'{training_folder}/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    with open(f'{training_folder}/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    paper_ids = list(metadata.keys())


    ts = MinTripletSampler(paper_ids=paper_ids, coviews=data, samples_per_query=5, ratio_hard_easy_neg=0.5)

    with cProfile.Profile() as pr:
        sum(1 for _ in ts.generate_triplets(train))

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats('profile.prof')


if __name__ == '__main__':
    main()
