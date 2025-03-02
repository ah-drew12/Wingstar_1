from transformers import TFBertForTokenClassification,BertTokenizerFast
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import Dataset, DatasetDict
import numpy as np
import tensorflow as tf


######################################################################################################################################################################################################################################
train_texts = [
    "I saw a Beetle in the garden today.",
    "The Butterfly gracefully fluttered by.",
    "My Cat loves to sleep on the couch.",
    "We have a Cow in our farm.",
    "The Dog barked loudly at the mailman.",
    "An Elephant is a majestic creature.",
    "I saw a Gorilla at the zoo.",
    "The Hippo was swimming in the river.",
    "A Lizard was sunbathing on the rock.",
    "The Monkey swung from tree to tree.",
    "A Mouse scurried across the floor.",
    "The Panda was eating bamboo.",
    "There was a Spider in the corner of the room.",
    "The Tiger roared in the jungle.",
    "A Zebra galloped across the savannah.",
    "I wish I had a pet Unicorn.",
    "The Rabbit hopped away quickly.",
    "The Fish swam in the aquarium.",
    "The Parrot mimicked our words.",
    "The Dolphin jumped out of the water.",
    "The Kangaroo jumped high.",
    "The Sheep were grazing in the field.",
    "The Owl hooted in the night.",
    "Our Cat chased a Mouse.",
    "The Cow was grazing near the barn.",
    "A Dog ran across the street.",
    "The Elephant trumpeted loudly.",
    "The Gorilla beat its chest.",
    "The Hippo opened its large mouth.",
    "A Lizard scurried up the tree.",
    "The Monkey ate a banana.",
    "A Mouse hid under the bed.",
    "The Panda climbed the tree.",
    "There was a Spider web in the corner.",
    "The Tiger prowled through the jungle.",
    "The Zebra grazed on the grass.",
    "A Lion roared across the savannah.",
    "The Unicorn pranced under the rainbow.",
    "A Rabbit nibbled on a carrot.",
    "The Bear hibernated in its cave.",
    "An Alligator lurked in the swamp.",
    "The Shark circled in the ocean.",
    "A Deer grazed in the meadow.",
    "The Squirrel gathered nuts.",
    "The Owl hooted softly at night.",
    "The Panda rolled playfully on the grass.",
    "A Spider spun a web in the garden.",
    "The Tiger stalked its prey silently.",
    "The Lion basked in the sun.",
    "A Bear lumbered through the forest.",
    "A Fish darted around the coral reef.",
    "A Dolphin swam gracefully alongside the boat.",
    "A Shark sliced through the water.",
    "A Deer bounded gracefully.",
    "The Sheep grazed on the hillside.",
    "A Squirrel chattered in the tree.",
    "The Owl stared with large, round eyes.",
    "In the grass hopped the Rabbit.",
    "Majestically glided the Eagle above the cliffs.",
    "On the branch sat the Robin.",
    "Across the field ran the Fox.",
    "Under the tree rested the Deer.",
    "In the sky soared the Hawk.",
    "At the pond drank the Elk.",
    "Near the shore lay the Seal.",
    "On the rock basked the Snake.",
    "Around the yard chased the Chickens.",
    "In the forest howled the Wolf.",
    "Across the plains roamed the Bison.",
    "On the roof perched the Pigeon.",
    "In the river swam the Otter.",
    "In the ocean leaped the Whale.",
    "In the garden sang the Canary.",
    "In the shadows crept the Panther.",
    "Under the leaves hid the Hedgehog.",
    "In the meadow flew the Butterfly.",
    "On the lily pad sat the Frog.",
    "Near the hive buzzed the Bee.",
    "In the alley prowled the Cat.",
    "At the edge of the forest stalked the Lynx.",
    "In the waterhole wallowed the Buffalo.",
    "Across the valley echoed the call of the Coyote.",
    "In the burrow slept the Mole.",
    "On the cliff stood the Mountain Goat.",
    "In the cave hibernated the Bear.",
    "On the fence roosted the Rooster.",
    "In the barn lived the Pig.",
    "On the lawn hunted the Falcon.",
    "In the jungle swung the Lemur.",
    "On the beach scurried the Crab.",
    "In the bushes rustled the Hedgehog.",
    "Over the meadow flew the Skylark.",
    "On the rock warmed itself the Iguana.",
    "In the pond splashed the Duck.",
    "Under the leaves hid the Snail.",
    "On the tree bark rested the Gecko.",
    "On the wall climbed the Tarantula.",
    "In the sky circled the Osprey.",
    "Over the field hunted the Kestrel.",
    "In the flower sat the Ladybug.",
    "At the water's edge stood the Heron.",
    "In the tall grass slithered the Python.",
    "Under the log hid the Salamander.",
    "In the reed sang the Warbler.",
    "Above the meadow hovered the Dragonfly.",
    "On the leaf crawled the Caterpillar.",
    "In the shadows prowled the Jaguar.",
    "Across the yard scuttled the Beetle.",
    "In the canopy swung the Gibbon.",
    "By the stream relaxed the Capybara.",
    "In the open field ran the Gazelle.",
    "Under the bush slept the Dormouse.",
    "In the cage chirped the Finch.",
    "On the riverbank basked the Crocodile.",
    "Near the mountain roared the Snow Leopard.",
    "In the trees howled the Howler Monkey.",
    "By the stream lounged the Water Buffalo.",
    "In the shadows crept the Leopard.",
    "On the rooftop prowled the Cat.",
    "In the barn slept the Horse.",
    "At the edge of the lake sang the Frog.",
    "In the field played the Rabbit.",
    "In the sky wheeled the Gull.",
    "On the leaf hid the Chameleon.",
    "The Beetle scurried under the fallen log.",
    "A Butterfly rested on the daisy.",
    "The Cat chased its own tail in circles.",
    "The Cow lay down in the shade.",
    "The Dog fetched the ball enthusiastically.",
    "An Elephant splashed water with its trunk.",
    "The Gorilla ate bananas in the enclosure.",
    "A Lizard climbed up the wall effortlessly.",
    "The Monkey stole a piece of fruit.",
    "A Mouse peeked out of its hole.",
    "The Panda slept soundly on a tree branch.",
    "A Spider hung from its silk thread.",
    "The Tiger's stripes blended into the foliage.",
    "A Lioness led her cubs through the grass.",
    "A Rabbit snuggled into the hay.",
    "The Bear found a honeycomb.",
    "A Fish swam playfully with its school.",
    "The Parrot whistled a tune.",
    "The Dolphin performed tricks for the audience.",
    "An Alligator snapped its jaws shut.",
    "The Shark glided silently beneath the surface.",
    "The Kangaroo carried its joey in its pouch.",
    "A Deer drank from the clear stream.",
    "A Squirrel buried its acorn.",
    "The Owl stared with large, round eyes.",
    "In the grass hopped the Rabbit.",
    "The Beetle crawled slowly on the leaf.",
    "The Butterfly landed softly on the flower.",
    "The Cat watched the birds from the window.",
    "The Cow gave us fresh milk every morning.",
    "The Dog loves to play fetch in the park.",
    "The Elephant trumpeted loudly in the distance.",
    "The Gorilla plays with its young in the zoo.",
    "The Hippo submerged itself in the cool water.",
    "The Lizard basked in the sun on the rock.",
    "The Monkey swung playfully from branch to branch.",
    "The Mouse darted quickly to avoid the cat.",
    "The Panda loves to munch on bamboo shoots.",
    "The Spider spun a beautiful web in the corner.",
    "The Tiger prowled silently through the tall grass.",
    "The Zebra's stripes are unique and beautiful."
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
    "Gull", "Chameleon"
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
    max_length=17
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