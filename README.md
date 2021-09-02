# Cue-Queue

Prototype work for solving Council Data Project's
[Linear Topic Segmentation Problem](https://github.com/CouncilDataProject/cdp-roadmap/issues/9).

## Steps

### Generation of Encoding

1. Create collection section break sentences ("cue sentences") for each transcript
2. Processing all section break sentences:
    1. Run TF-IDF over section break sentences to find the common terms
       that are generally used in such sentences.
       These can be thought of as "cue phrases"
    2. Create encodings of all "cue sentences" and take average of them all to
       create a "general cue sentence encoding".
3. Store the general "cue sentence encoding" and "commen cue phrases".

### Usage of Encodings

1. For each sentence:
    1. Generate and store the distance of the sentence from the
       "general cue sentence encoding".
    2. (Optional for validation) Generate and store the distance of the sentence
       from each topic.
2. Find the `N` most similar (least distant) sentences from the
   "general cue sentence encoding", where `N` is the number of topics.
3. Using the found `N` most similar sentences to the "general cue sentence encoding,"
   search for the single sentence within a window with the highest term frequency of
   terms in the TFIDF "common cue phrases set."
4. Apply labels by zipping the selected cue sentences and the topics.
5. Optionally validate by taking average of all sentence encodings within each section
   and ensure all distances meet some threshold.
