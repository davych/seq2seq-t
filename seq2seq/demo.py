import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("zh_core_web_sm")

# Process whole documents
text = ("枇杷树不高，枝干离地也就一米多到两米，一般踮着脚就可采摘，高一点的树枝可用竹钩勾下来。我们在树丛中来来回回认真地采摘，大家都很兴奋，不时飞扬起欢声笑语。和煦的阳光透过枝叶，在我们身上洒下斑驳的投影，树林中弥漫着淡淡的果香。因为朱果农允许我们可以边摘边品尝，所以我们不时地摘下一个枇杷，轻轻的剥了皮，放到嘴里轻轻一咬，噢，软软的，细细的，水水的，有的微酸，有的甘甜，不论甜中带酸还是酸中带甜，都是恰到好处的鲜美，那滋味，真是一种享受啊。黄澄澄圆溜溜的枇杷果，也使我不禁想起了古人“东园载酒西园醉，摘尽枇杷一树金”的诗句，把枇杷当作下酒菜，也是绝了，只是这种境界是宋人的，现代人谁还讲究这些")
doc = nlp(text)

# Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)