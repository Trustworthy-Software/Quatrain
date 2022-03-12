# Structured Bug Report Data
1. bug report summary: title for bug issue  
2. bug report description: detailed description for bug issue

## BugReport folder
* Bug reports for Defects4j, Bugsjar, Bears. Structured as `project-id $$ bug report summary $$ bug report description`

## bugreport_patch.txt
* Natural language representations of bug reports and patch natural language text. Structured as `project-id $$ bug report summary $$ bug report description $$ patchId $$ patch text $$ label`

After extracting all bug reports:
1. run data/BugReport/assemble_bug_report.py to get Bug_Report_All.json
2. run data/save_bugreport_patch.py to get standarized dataset: bugreport_patch.txt
3. run a few script under data/CommitMessage