from transformers import BartForConditionalGeneration, BartTokenizer
from os import path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


NeoXT_path = "src/GPTAlten/models/NeoXT"
NeoXT_token_path = "src/GPTAlten/models/NeoXT_token"
if path.exists(NeoXT_path) is False and path.exists(NeoXT_token_path) is False:
    print("if marche")
    # if model is on hugging face Hub
    model = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
    tokenizer = AutoModelForCausalLM.from_pretrained(
        "togethercomputer/GPT-NeoXT-Chat-Base-20B", torch_dtype=torch.bfloat16
    )
    model.save_pretrained(NeoXT_path, from_pt=True)
    tokenizer.save_pretrained(NeoXT_token_path, from_pt=True)
else:
    print("else marche")
    # from local folder
    model = AutoTokenizer.from_pretrained(NeoXT_path)
    tokenizer = AutoModelForCausalLM.from_pretrained(NeoXT_token_path)


# infer
inputs = tokenizer("Hello!:", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.8)
output_str = tokenizer.decode(outputs[0])
print(output_str)


# from transformers import pipeline

# text = "Existing law authorizes state agencies to enter into contracts for the acquisition \
#     of goods or services upon approval by the Department of General Services. Existing \
#         law sets forth various requirements and prohibitions for those contracts, including, \
#             but not limited to, a prohibition on entering into contracts for the acquisition of goods\
#                   or services of $100,000 or more with a contractor that discriminates between spouses \
#                     and domestic partners or same-sex and different-sex couples in the provision of benefits. \
#                         Existing law provides that a contract entered into in violation of those requirements and \
#                             prohibitions is void and authorizes the state or any person acting on behalf of the state \
#                                 to bring a civil action seeking a determination that a contract is in violation and \
#                                     therefore void. Under existing law, a willful violation of those requirements and \
#                                         prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency \
#                                             from entering into contracts for the acquisition of goods or services of $100,000 \
#                                                 or more with a contractor that discriminates between employees on the basis of gender\
#                                                       identity in the provision of benefits, as specified. By expanding the scope of a crime, \
#                                                         this bill would impose a state-mandated local program.\nThe California Constitution \
#                                                             requires the state to reimburse local agencies and school districts for certain costs\
#                                                                   mandated by the state. Statutory provisions establish procedures for making that \
#                                                                     reimbursement.\nThis bill would provide that no reimbursement is required by this\
#                                                                           act for a specified reason.',"

# summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
# print("\n\n\n\n\nCA MARCHE\n", summarizer(text))


# The model 'BertLMHeadModel' is not supported for summarization. Supported models are ['BartForConditionalGeneration',
#                                                                                       'BigBirdPegasusForConditionalGeneration',
#                                                                                         'BlenderbotForConditionalGeneration',
#                                                                                         'BlenderbotSmallForConditionalGeneration',
#                                                                                           'EncoderDecoderModel',
#                                                                                           'FSMTForConditionalGeneration',
#                                                                                           'GPT SanJapaneseForConditionalGeneration',
#                                                                                           'LEDForConditionalGeneration', 'LongT5ForConditionalGeneration',
#                                                                                           'M2M100ForConditionalGeneration', 'MarianMTModel',
#                                                                                           'MBartForConditionalGeneration', 'MT5ForConditionalGeneration',
#                                                                                             'MvpForConditionalGeneration', 'NllbMoeForConditionalGeneration',
#                                                                                             'PegasusForConditionalGeneration', 'PegasusXForConditionalGeneration',
#                                                                                               'PLBartForConditionalGeneration', 'ProphetNetForConditionalGeneration',
#                                                                                                 'SwitchTransformersForConditionalGeneration', 'T5ForConditionalGeneration',
#                                                                                                   'XLMProphetNetForConditionalGeneration'].

# model.save_pretrained("/home/my_name/Desktop/t5small", from_pt=True)

# # And then, you can load the model:

# model = BertModel.from_pretrained("/home/my_name/Desktop/t5small")
