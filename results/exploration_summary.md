# Dataset exploration — Resume Entities for NER

## Source
- Kaggle (recommended): https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner
  - Owner DataTurks, 220 manually labelled resumes from indeed.com,
    10 entity categories, 1,220,776 bytes total, updated 2018-07-12.
- Files used (identical official files from the original publisher,
  DataTurks, because the Kaggle API download needs credentials and the
  Kaggle API host is unreachable through this VM's egress proxy):
  - `data/traindata.json` — 1,106,139 bytes, 200 resumes
  - `data/testdata.json` — 111,557 bytes, 20 resumes
  - https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy

## Size
- Total resumes: 220 (train 200, test 20)
- Resume text: mean 472 words, median 400,
  min 29, max 1250
- Resumes with a name: 218
- Resumes with a designation: 202
- Resumes with skills: 183
- Resumes with a degree: 192
- Resumes with companies: 190
- Mean Skills spans per resume: 1.6 (max 21) — each span can list several skills
- Mean years of experience (where stated): 4.8
- Graduation years present: 21 distinct
  values, range 1980–
  2018

## Entity label counts (mentions across all resumes)
              label  entity_mentions  resumes_with_label
               Name              218                 218
       College Name              305                 194
             Degree              285                 192
    Graduation Year              157                 104
Years of Experience               34                  34
Companies worked at              226                 190
        Designation              268                 202
             Skills              356                 183
           Location              203                 199
      Email Address              190                 190

## Notes for the RAG pipeline
- Only 220 resumes: the whole pool fits in one ChromaDB collection, so no
  sampling is needed for the index (unlike Tasks 6/7, where subsets were used).
- The `Skills` and `Designation` annotations are the gold labels used for
  retrieval evaluation (precision@k / recall@k in `src/evaluate.py`).
- No gender, ethnicity or age field exists in the data. The bias check
  therefore uses only explicitly labelled attributes present in the data
  (graduation-year cohort, college, location, years of experience) and
  never infers protected attributes from names.
- Data quirks: 2 annotations carry the label `UNKNOWN`; 2 'Years of
  Experience' values are just a year ('2004') and are treated as noise.
  Some resumes have no Skills annotation at all (37 of 220).
