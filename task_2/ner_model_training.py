from transformers import TFBertForTokenClassification,BertTokenizerFast
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import Dataset, DatasetDict
import numpy as np
import tensorflow as tf


######################################################################################################################################################################################################################################
train_texts = [
    "I saw a beetle in the garden today.",
    "The butterfly gracefully fluttered by.",
    "My cat loves to sleep on the couch.",
    "We have a cow in our farm.",
    "The dog barked loudly at the mailman.",
    "An elephant is a majestic creature.",
    "I saw a gorilla at the zoo.",
    "The hippo was swimming in the river.",
    "A lizard was sunbathing on the rock.",
    "The monkey swung from tree to tree.",
    "A mouse scurried across the floor.",
    "The panda was eating bamboo.",
    "There was a spider in the corner of the room.",
    "The tiger roared in the jungle.",
    "A zebra galloped across the savannah.",
    "The beetle crawled under the leaves.",
    "A butterfly danced above the meadow.",
    "The cat napped in the sunbeam.",
    "The cow stood by the fence.",
    "The dog chased after a rabbit.",
    "An elephant shook the ground with its footsteps.",
    "The gorilla groomed its fur.",
    "The hippo wallowed in the mud.",
    "A lizard scampered up the tree.",
    "The monkey clambered up the branches.",
    "A mouse darted into a crevice.",
    "The panda lounged under the tree.",
    "A spider crawled along the wall.",
    "The tiger crouched in the tall grass.",
    "The zebra sprinted across the savannah.",
    "The beetle hid under the bark.",
    "The butterfly rested on a petal.",
    "The cat licked its paws.",
    "The cow drank from the pond.",
    "The dog howled at the moon.",
    "An elephant walked through the clearing.",
    "The gorilla sat under the shade.",
    "The hippo emerged from the water.",
    "A lizard lay motionless on the stone.",
    "The monkey played with the leaves.",
    "A mouse squeaked and ran away.",
    "The panda lay on its back.",
    "A spider spun an intricate web.",
    "The tiger's eyes gleamed in the night.",
    "The zebra's stripes shimmered in the sun.",
    "The beetle scurried across the field.",
    "The butterfly floated above the grass.",
    "The cat curled up on the bed.",
    "The cow wandered through the meadow.",
    "The dog sniffed around the yard.",
    "An elephant picked up a branch with its trunk.",
    "The gorilla looked out from its enclosure.",
    "The hippo splashed in the shallow waters.",
    "A lizard ran up the garden wall.",
    "The monkey grabbed a handful of fruit.",
    "A mouse chewed on a small piece of food.",
    "The panda lounged in the bamboo grove.",
    "A spider climbed up the tree trunk.",
    "The tiger's fur rippled as it moved.",
    "The zebra grazed with other animals.",
    "The beetle explored the undergrowth.",
    "The butterfly hovered near the blossoms.",
    "The cat chased a feather toy.",
    "The cow lowed softly in the barn.",
    "The dog panted after a long run.",
    "An elephant trumpeted at the waterhole.",
    "The gorilla sat and observed.",
    "The hippo submerged to keep cool.",
    "A lizard darted into a crack.",
    "The monkey picked fruit from the tree.",
    "A mouse hid from the light.",
    "The panda sat munching on bamboo.",
    "A spider waited in its web for prey.",
    "The tiger crouched, ready to pounce.",
    "The zebra galloped across the plain.",
    "The beetle burrowed into the ground.",
    "The butterfly fluttered past the roses.",
    "The cat purred contentedly.",
    "The cow roamed the pasture.",
    "The dog rolled in the grass.",
    "The hippo opened its large mouth.",
    "A lizard scurried up the tree.",
    "The monkey ate a banana.",
    "A mouse hid under the bed.",
    "The panda climbed the tree.",
    "There was a spider web in the corner.",
    "The tiger prowled through the jungle.",
    "The zebra grazed on the grass.",
    "The beetle scurried under the fallen log.",
    "A butterfly rested on the daisy.",
    "The cat chased its own tail in circles.",
    "The cow lay down in the shade.",
    "The dog fetched the ball enthusiastically.",
    "An elephant splashed water with its trunk.",
    "The gorilla ate bananas in the enclosure.",
    "The hippo yawned widely, showing its large teeth.",
    "A lizard climbed up the wall effortlessly.",
    "The monkey stole a piece of fruit.",
    "A mouse peeked out of its hole.",
    "The panda slept soundly on a tree branch.",
    "A spider hung from its silk thread.",
    "The tiger's stripes blended into the foliage.",
    "The zebra's stripes created a beautiful pattern.",
    "The beetle climbed the stem of a plant.",
    "The butterfly rested on a sunflower.",
    "The cat hunted a toy mouse.",
    "The cow drank from the trough.",
    "The dog barked at passing cars.",
    "An elephant bathed in the river.",
    "The gorilla foraged for fruit.",
    "The hippo dozed by the water's edge.",
    "A lizard hid among the rocks.",
    "The monkey swung from vine to vine.",
    "A mouse nibbled on some cheese.",
    "The panda tumbled playfully.",
    "A spider crawled on the ceiling.",
    "The tiger watched its surroundings closely.",
    "The zebra stood with its herd."
   "The Tiger prowled silently through the tall grass.",
    "The Zebra's stripes are unique and beautiful." "I saw a Beetle in the garden today.",
    "The Butterfly gracefully fluttered by.", "My Cat loves to sleep on the couch.",
    "We have a Cow in our farm.", "The Dog barked loudly at the mailman.",
    "An Elephant is a majestic creature.", "I saw a Gorilla at the zoo.",
    "The Hippo was swimming in the river.", "A Lizard was sunbathing on the rock.",
    "The Monkey swung from tree to tree.", "A Mouse scurried across the floor.",
    "The Panda was eating bamboo.", "There was a Spider in the corner of the room.",
    "The Tiger roared in the jungle.", "A Zebra galloped across the savannah.",
    "The Beetle crawled across the leaf.", "The Butterfly danced in the breeze.",
    "Our Cat purred softly on my lap.", "The Cow mooed gently in the field.",

    "The Dog wagged its tail excitedly.", "The Elephant trumpeted loudly in the savannah.",
    "The Gorilla swung from branch to branch.", "The Hippo lounged in the muddy water.",
    "A Lizard darted across the patio.", "The Monkey chattered noisily in the treetops.",
    "A Mouse nibbled on a piece of cheese.", "The Panda rolled playfully on the grass.",
    "A Spider spun a web in the garden.", "The Tiger stalked its prey silently.",
    "The Beetle scurried under the fallen log.", "A Butterfly rested on the daisy.",
    "The Cat chased its own tail in circles.", "The Cow lay down in the shade.",
    "The Dog fetched the ball enthusiastically.", "An Elephant splashed water with its trunk.",
    "The Gorilla ate bananas in the enclosure.", "The Hippo yawned widely, showing its large teeth.",
    "A Lizard climbed up the wall effortlessly.", "The Monkey stole a piece of fruit.",
    "A Mouse peeked out of its hole.", "The Panda slept soundly on a tree branch.",
    "A Spider hung from its silk thread.", "The Tiger's stripes blended into the foliage.",
    "The Zebra's stripes were mesmerizing.", "The Beetle scuttled across the floor.",
    "The Butterfly gently landed on the flower.", "The Cat meowed and rubbed against my leg.",

    "The Cow chewed cud peacefully.", "The Dog ran after the frisbee.", "The Elephant raised its trunk to trumpet.",
    "The Gorilla played with its young.", "The Hippo submerged itself in the water.",
    "A Lizard basked in the sunlight.", "The Monkey swung from branch to branch.",
    "A Mouse sniffed around for food.", "The Panda lazily ate bamboo.", "A Spider crawled along its web.",
    "The Tiger prowled through the jungle.", "The Zebra trotted across the plain.", "The Beetle hid under a rock.",
    "The Butterfly fluttered in the meadow.", "The Cat purred while napping.", "The Cow grazed in the pasture.",
    "The Dog barked at the stranger.", "The Elephant trumpeted loudly.", "The Gorilla climbed a tree.",
    "The Hippo lounged by the riverbank.", "A Lizard darted across the sand.", "The Monkey chattered excitedly.",
    "A Mouse found some crumbs.", "The Panda rolled down the hill.", "A Spider spun its web in the corner.",
    "The Tiger leapt gracefully.", "The Zebra stood near the watering hole.", "The Beetle explored the garden.",
    "The Butterfly landed on a leaf.", "The Cat stretched and yawned.", "The Cow lay in the shade of a tree.",
    "The Dog wagged its tail happily.",
    "An Elephant waded through the water.", "The Gorilla beat its chest.", "The Hippo opened its massive mouth.",
    "A Lizard sunned itself on a rock.",
    "The Monkey swung from a vine.", "A Mouse hid in a small hole.", "The Panda ate a large bamboo shoot.",
    "A Spider dangled from its thread.",
    "The Tiger prowled silently.",
    "The Zebra grazed with its herd."

]

test_texts=[
    "The Tiger stalked its prey.",
    "A Zebra was drinking from the river.",
    "No, there is no Lion here.",
    "I don't believe there is a Unicorn in the room.",
    "The Rabbit dug a hole in the ground.",
    "We couldn't find a Bear anywhere.",
    "The Fish glided through the water.",
    "The Parrot squawked loudly.",
    "The Dolphin swam near the boat.",
    "There is absolutely no Alligator here.",
    "The Kangaroo had a joey in its pouch.",
    "I didn't notice any Deer around.",
    "The Sheep were gathered near the fence.",
    "There isn't a Squirrel in sight.",
    "The Owl sat on the branch.",
    "The Peacock strutted around the yard."]
animals = [
    "Eagle", "Robin", "Fox", "Hawk", "Elk", "Seal", "Snake", "Chickens", "Wolf", "Bison", "Pigeon",
    "Otter", "Whale", "Canary", "Panther", "Antelope", "Hedgehog", "Frog", "Bee", "Lynx", "Buffalo",
    "Coyote", "Mole", "Mountain Goat", "Sparrows", "Raccoon", "Rooster", "Pig", "Woodpecker", "Falcon",
    "Lemur", "Crab", "Skylark", "Iguana", "Duck", "Snail", "Gecko", "Chipmunk", "Tarantula", "Osprey",
    "Kingfisher", "Kestrel", "Ladybug", "Heron", "Python", "Salamander", "Warbler", "Dragonfly",
    "Caterpillar", "Jaguar", "Tortoise", "Flamingo", "Gibbon", "Capybara", "Gazelle", "Dormouse",
    "Finch", "Crocodile", "Snow Leopard", "Howler Monkey", "Water Buffalo", "Leopard", "Horse",
    "Gull", "Chameleon","beetle", "butterfly", "cat", "cow", "dog", "elephant", "gorilla", "hippo",
    "lizard", "monkey", "mouse", "panda", "spider", "tiger", "zebra", "Beetle", "Butterfly", "Cat", "Cow", "Dog", "Elephant", "Gorilla", "Hippo",
    "Lizard", "Monkey", "Mouse", "Panda", "Spider", "Tiger", "Zebra"
]

########################################################################################################################################################################################################################################################################################################################
def encode_message(message, animals):
    words = message.split()
    encoded_words = [1 if any(animal in word for animal in animals) else 0 for word in words]
    return encoded_words

train_labels = [encode_message(text, animals) for text in train_texts]
test_labels = [encode_message(text, animals) for text in test_texts]

def encode_texts(label):
    labels_encoding={0:"O", 1: "B-ANIMAL"}
    return [labels_encoding[label[i]] for i in range(len(label))]
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_texts=np.array(train_texts)
test_texts=np.array(test_texts)

train_ner_tags=[encode_texts(label) for label in train_labels]
test_ner_tags=[encode_texts(label) for label in test_labels]

train_tokens=[tokenizer.tokenize(text) for i,text in enumerate(train_texts)]
test_tokens=[tokenizer.tokenize(text) for i,text in enumerate(test_texts)]

train_dataset=Dataset.from_dict({"text": train_texts, "labels": train_labels, "tokens": train_tokens,'ner_tags':train_ner_tags})
test_dataset=Dataset.from_dict({"text": test_texts, "labels": test_labels, "tokens": test_tokens,'ner_tags':test_ner_tags})

dataset = DatasetDict({"train": train_dataset,
                             "test": test_dataset})


def tokenize_and_align_labels(examples,label_all_tokens=True):

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    max_length=24
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        current_word = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx != current_word:
                current_word=word_idx
                label_temp=0 if (word_idx is None) or (word_idx >= len(label)) else label[word_idx]
                label_ids.append(label_temp)

            elif word_idx is None:
                label_ids.append(0)
            else:
                label_ids.append(0)


        labels.append(label_ids)
        if len(label_ids)==20:
            ttt=examples["text"][i]
            continue
        max_length = max(max_length, len(label_ids))

    tokenized_inputs['input_ids'] = pad_sequences(tokenized_inputs['input_ids'], maxlen=max_length, padding="post", value=0)
    labels = pad_sequences(labels, maxlen=max_length, padding="post", value=0)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
def pad_to_max_length(sequences, max_length, padding_value=0):
    return [seq + [padding_value] * (max_length - len(seq)) for seq in sequences]


max_length_train = max(len(seq) for seq in tokenized_dataset["train"]["input_ids"])
max_length_test = max(len(seq) for seq in tokenized_dataset["test"]["input_ids"])
max_length = max(max_length_train, max_length_test)


train_inputs = tokenized_dataset["train"]["input_ids"]
train_attention_mask = pad_to_max_length(tokenized_dataset["train"]["attention_mask"], max_length, padding_value=0)
train_labels = tokenized_dataset["train"]["labels"]

test_inputs = tokenized_dataset["test"]["input_ids"]
test_attention_mask = pad_to_max_length(tokenized_dataset["test"]["attention_mask"], max_length, padding_value=0)
test_labels = tokenized_dataset["test"]["labels"]


train_data = tf.data.Dataset.from_tensor_slices((
    {"input_ids": train_inputs, "attention_mask": train_attention_mask},
    train_labels
)).batch(2)

test_data = tf.data.Dataset.from_tensor_slices((
    {"input_ids": test_inputs, "attention_mask": test_attention_mask},
    test_labels
)).batch(2)




model = TFBertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.fit(train_data, epochs=5)
predictions = model.predict(test_data)
u=model.evaluate(test_data)
logits = predictions.logits

predicted_labels = np.argmax(logits, axis=-1)
rf=predicted_labels-test_labels

model.save_pretrained("./trained_ner_model")
tokenizer.save_pretrained("./trained_ner_model")
