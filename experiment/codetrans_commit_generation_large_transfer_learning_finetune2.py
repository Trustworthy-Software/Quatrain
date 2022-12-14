# -*- coding: utf-8 -*-
"""codeTrans_commit_generation_large_transfer_learning_finetune

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/agemagician/CodeTrans/blob/main/prediction/transfer%20learning%20fine-tuning/commit%20generation/large_model.ipynb

**<h3>Generate the commit for github code changes using codeTrans transfer learning finetuning model</h3>**
<h4>You can make free prediction online through this 
<a href="https://huggingface.co/SEBIS/code_trans_t5_large_commit_generation_transfer_learning_finetune">Link</a></h4> (When using the prediction online, you need to parse and tokenize the code first.)

**1. Load necessry libraries including huggingface transformers**
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

#!pip install -q transformers sentencepiece

from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
# from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline

"""**2. Load the token classification pipeline and load it into the GPU if avilabile**"""

pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_large_commit_generation_transfer_learning_finetune"),
    tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_large_commit_generation_transfer_learning_finetune"),
    device=0
)

"""**3 Give the code for summarization, parse and tokenize it**"""

#code = "--- a/source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java\n +++ b/source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java\n@@ -1794,7 +1794,7 @@ public abstract class AbstractCategoryItemRenderer extends AbstractRenderer\n}\n\nint index = this.plot.getIndexOf(this);\n\n         CategoryDataset dataset = this.plot.getDataset(index);\n-        if (dataset != null) {\n+        if (dataset == null) {\n\n             return result;\n         }\nint seriesCount = dataset.getRowCount();\n"

import nltk
nltk.download('punkt')
from nltk.tokenize import WordPunctTokenizer
import os

path_root = './ISSTA2022'

def text_generate(path_root):
    for root, dirs, files in os.walk(path_root):
        for file in files:
            if file.endswith('.patch'):
                target = os.path.join(root, file).replace('.patch', '.txt')
                if os.path.exists(target):
                    continue
                patch_text = ''
                head = ''
                content = ''
                found = False
                at_number = 0
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            if line.startswith('--- ') or line.startswith('+++ '):
                                head += line
                            if not found and not line.startswith('@@ '):
                                continue
                            elif found:
                                if not line.startswith('@@ '):
                                    content += line
                                else:
                                    at_number += 1
                                    patch_code_partly = content
                                    # generate natural language text for patch code
                                    patch_text_partly = code_trans(head + patch_code_partly)
                                    patch_text += patch_text_partly + '. '
                                    # end
                                    content = line
                            else:
                                # first @@ are found
                                if at_number == 0:
                                    at_number += 1
                                    found = True
                                    content += line
                except Exception as e:
                    print(e)
                    continue
                # deal with the last @@
                patch_code_partly = content
                # generate natural language text for patch code
                patch_text_partly = code_trans(head + patch_code_partly)
                patch_text += patch_text_partly + '. '
                print(patch_text)

                with open(target, 'w+') as f2:
                    f2.write(patch_text)

def code_trans(patch_code):
    tokenized_list = WordPunctTokenizer().tokenize(patch_code)
    tokenized_code = ' '.join(tokenized_list)
    #print("tokenized code: " + tokenized_code)

    """**4. Make Prediction**"""
    result = pipeline([tokenized_code])[0]['summary_text']
    print(result)
    return result

if __name__ == '__main__':
    text_generate(path_root)
