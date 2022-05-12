import re


def split_by_md_code_block(documents):
    artifacts = []
    text = []
    for document in documents:
        splits = document.split("```")
        for i, split in enumerate(splits):
            if i % 2:
                artifacts.append("```")
                artifacts.extend(split.splitlines())
                artifacts.append("```")
            else:
                for line in split.splitlines():
                    if is_markdown_artifact(line):
                        artifacts.append(line)
                    else:
                        text.append(line)

    artifacts = [i for i in artifacts if i.strip() != ""]
    text = [i for i in text if i.strip() != ""]

    return artifacts, text


def regex_cleanup(lines):
    artifacts = []
    text = []
    for line in lines:
        if is_log_output_artifact(line) or \
                is_filelisting_or_prompt_artifact(line) or \
                is_java_code_artifact(line) or \
                is_json_artifact(line) or \
                is_xml_artifact(line) or \
                is_commentary(line) or \
                is_not_separable(line):
            pass
        else:
            text.append(line)

    return artifacts, text


def is_markdown_artifact(line):
    if re.match(r"^\s{4}.*$", line): # starts with four spaces, NO STRIP!
        return True

    md_enumerate = r"^([-\*]|([0-9]+\.))?\s*"

    rex = [md_enumerate + r"`[^`]*`$", # single line quote
           md_enumerate + r"http(s)?://[A-Za-z0-9\-\._~]+(:\d+)?(?:/[A-Za-z0-9\-\._~\?&%$#!=]+)*/?$", # urls
           r"^\|(.*\|){2,}$", # tables "| factory123      | null | user123 |"
           r"^\s*!{0,1}[\*\-#]*\s*(?:`.+`)?\[.*\]\(.+\)\s*$", # md links "[logcat.txt](https://github.com/google/ExoPlayer/files/3783649/logcat.txt)"
           r"^[0-9\.\s\{\}\(\);\.,:\-\+#@!\$%\^\\&=\[\]\|<>\?_\*]*$"] # line contains only special chars and numbers
    return match_any(rex, line)


def is_log_output_artifact(line):
    rex = [ r"^Caused\s+by:\s+(org|com)\..*$", # stacktrace
            r"^at (?:\w+\.)+\w+.*", # stacktrace "at com.codename1.ui.RunnableWrapper.run(RunnableWrapper.java:119)"
            r"^\[[A-Z]{3,}\].*$", # log in form "[EDT] 0:0:1,467 - Data changed:AB"
            r"^\[?\d{2,4}[:\-\./]\d{2}.*"] # [10:55:16] Verify if minikube is running [completed]
    return match_any(rex, line)


def is_filelisting_or_prompt_artifact(line):
    rex = [ r"^\$\s+.*$", # bash prompt
            # r"^\s*@\s+\./..*$", #  @ ./node_modules/@theia/monaco/lib/browser/textmate/monaco-textmate-service.js
            # r"^\d+\s[A-Za-z]{3}\s\d{1,2}\s.*", #    23122 Jan 23 11:30 LaunchImage-700-Portrait~ipad.png
            r"^[A-Za-z]:\\(.*?\\)+.*"] # windows paths "c:\eXist-db\tools\yajsw\classes\org\xmlunit\Input.class"
    return match_any(rex, line)


def is_java_code_artifact(line):
    rex = [r"^([\}\{].*)|(.*[\}\{])$", # starts or ends with curly bracket
           r"^.*;$"] # ends with ;
    return match_any(rex, line)


def is_json_artifact(line):
    rex = [r"^.*\",$"] # "applicationType": "gateway",
    return match_any(rex, line)


def is_xml_artifact(line):
    rex = [r"^(?!<b>)<.*$"] # everything that starts with "<" but is not a html bold portion
                            # <hibernate.entitymanager.version>4.2.3.Final</hibernate.entitymanager.version>
    return match_any(rex, line) and not is_commentary(line)


def is_commentary(line):
    rex = [r"^<!--.*$", # xml comment <!-- A clear and concise description of what you expected to happen or to show. -->
           r"^(/\*.*)|(.*\*/)$", # java block comment start and end
           r"^//.*$"] # java line comment
    return match_any(rex, line)


def is_not_separable(line):
    rex = [r"^>.*$", # > I'm using realm for databse processingin my android project. And I got some user's report about the crash:
                     # > java.lang.reflect.Method.invoke(Method.java:606)
           r"^#\s.*$"] # "# <intercept-url pattern="/index.jsf" filters="none" />"
                       # "# Updated Description"
    return match_any(rex, line)


def match_any(rex_list, line):
    for r in rex_list:
        if re.match(r, line.strip()):
            return True
    return False
