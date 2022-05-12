from os.path import dirname

PROJECT_ROOT = dirname(dirname(dirname(__file__))) + '/'

NLON_PATH = PROJECT_ROOT + 'datasets/NLoN_dataset/'
NLON_DATASETS = [('kubernetes', 'lines.10k.cfo.sample.2000 - Kubernetes (Slackarchive.io).csv'),
                 ('lucene', 'lines.10k.cfo.sample.2000 - Lucene-dev mailing list.csv'),
                  ('mozilla', 'lines.10k.cfo.sample.2000 - Mozilla (Firefox, Core, OS).csv')]

REVIEWER_1_PATH = PROJECT_ROOT + 'datasets/validation_set_researcher_1.csv.zip'
REVIEWER_2_PATH = PROJECT_ROOT + 'datasets/validation_set_researcher_2.csv.zip'

TRAINING_SET_BUG_REPORTS_PATH = PROJECT_ROOT + 'datasets/training_set_bug_reports.csv.zip'
TEST_SET_BUG_REPORTS_PATH = PROJECT_ROOT + 'datasets/test_set_bug_reports.csv.zip'
MD_DOCUMENTATION_PATH = PROJECT_ROOT + 'datasets/documentation_set.csv.zip'