# Draft Paper Notes

### We don't know

We don't know how to align two texts, one a timeline of events in a meeting (minutes), and one a transcript of a meeting, together, in an automated fashion. Many "text segmentation models" are largely "topic segmentation" and not strict alignment models.

### It is worth knowing because

There are many scenarios in which this automatic alignment would be incredible useful for information retrieval and navigation. In my case, with my dataset of municipal council meetings, in creating an automated method for aligning a meetings minutes to the transcript, we can easily create subsections of the meeting for later use in further study (i.e. "give me a dataset of all meetings but only give me the public comment sections from each meeting), or in navigation of the meeting in a web application (i.e. "jump to 'discussion on bill 1234' in meeting video").

### It requires investigation/research because

Outside of my use-case, during and post COVID-19-pandemic, virtual meetings become common and, the ability for meeting recording, transcription, and more, is generally available and used by more than municipal councils. This research is largely applicable to many meeting types, regardless of meeting content. Further, utilizing this work and future work we can see how methods for segmenting text by alignment rather than topical segmentation generalizes to meeting content or where methods fail (i.e. how domain specific is each method, are there general methods for segmentation from alignment regardless of meeting content?)

## Specific Research Questions:

-   Existing work for transcript segmentation is largely based on topical shift and there is limited work in document-document alignment, what heuristics can we impose on an alignment process to improve upon on state-of-the-art minutes alignment.
-   There has been promising early results for neural methods for alignment in genomics with DeepBLAST -- will a neural method for alignment provide a better method for alignment that current semantic vector and topical shift methods.

## Results Table

| Method           | Choi    | Wiki-727k | CDP-Seattle |
| ---------------- | ------- | --------- | ----------- |
| Koshorek et. al. | 5.6-7.2 | 22.13     | Pk          |
| Tardy et. al.    | Pk      | Pk        | Pk          |
| SliceCast        | Pk      | Pk        | Pk          |
| cue-queue (ours) | Pk      | Pk        | Pk          |

Koshorek et. al. is the somewhat foundational "text segmentation and a supervised learning task" paper.

Anywhere there is a `Pk` has not previously been reported.

Tardy et. al. didn't compare their method with any existing dataset at all but rather provided a new corpus "public_meetings" which is a French public meetings dataset they created themselves (22 meetings in total). Fortunately, their method is easy enough to recreate and was basically what I had in mind just without my imposed heuristic so it's easy to "recreate".

SliceCast has a reported value for Wiki-813k but not for Wiki-727k.

### Future Paper

| Method                                 | Choi    | Wiki-727k | CDP-Seattle |
| -------------------------------------- | ------- | --------- | ----------- |
| Koshorek et. al.                       | 5.6-7.2 | 22.13     | Pk          |
| Tardy et. al.                          | Pk      | Pk        | Pk          |
| SliceCast                              | Pk      | Pk        | Pk          |
| cue-queue (ours)                       | Pk      | Pk        | Pk          |
| cue-queue-neural-base (ours)           | Pk      | Pk        | Pk          |
| cue-queue-neural-with-heuristic (ours) | Pk      | Pk        | Pk          |

Build off of DeepBLAST for our neural method.

## Spanner in the Works

While working on setting up the basic CDP seattle data the last couple of days I realized why "alignment" is a hard thing....

Some minutes items are solved in bulk:

```json
{
    "name": "Appt 02047"
},
{
    "name": "Appt 02048"
},
{
    "name": "Appt 02049"
},
{
    "name": "Appt 02050"
},
{
    "name": "Appt 02053"
},
{
    "name": "Appt 02051"
},
{
    "name": "Appt 02052"
}
```

```json
{
    "text": "The clerk please affix my signature to the legislation on my behalf and please read item four through eight into the record."
}
```

```json
{
    "text": "Hearing none will the Closuring please call the role on confirmation of appointments 2047 through 2050 and 2053."
}
```

I can see now why many similar problems approach this as a "topic segmentation" and not "minutes alignment"

We could very easily segment this collection of sentences as "Discussion and Voting on Appointments" but because they are all discussed and voted on together we can't _really_ use an alignment method unless we consider all but the first appointment minutes item to be "deletions" from the aligned sequence.

Additionally, there are many minutes items that are _real and valid_ but are hard to align because they take up a single sentence in the transcript / meeting and as such aren't the most valuable to use as navigation or dataset points.

```json
{"name": "Department Overview Presentations"},
{"name": "The City Budget Office (CBO) and City Department Directors present changes reflected in the Mayor's Proposed 2022 Budget."},
{"name": "PRESENTATIONS"},
{"name": "APPROVAL OF THE JOURNAL"},
{"name": "Min 348"},
{"name": "ADOPTION OF INTRODUCTION AND REFERRAL CALENDAR"},
{"name": "IRC 320"},
{"name": "COMMITTEE REPORTS"},
{"name": "PAYMENT OF BILLS"}
```

While most of these are ignorable from my perspective, the important two are "MIN 348" and "IRC 320" which correspond to the minutes of the prior meeting being approved and the "Introduction and Referral Calendar" which is the document that councilmembers use as a timeline of when things are getting introduced from subcommittee to full council.

Any many of these have supporting documents that are useful for quick viewing.

The more I work on this problem, it seems to change. It started out as topic segmentation, then switched to minutes item alignment, now I feel like I am going back to topic segmentation but I want to use the minutes items as seeds...
