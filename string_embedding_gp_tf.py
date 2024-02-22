from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name)

def compute_embeddings(text, class_embedding=True, mean_embedding=True):
    global model, tokenizer
    # Encode the text
    input_ids = tokenizer.encode(text, add_special_tokens=True)  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    input_ids = tf.expand_dims(input_ids, 0)  # Create a tf tensor and add an extra dimension for the batch

    # Get the embeddings
    out = model(input_ids)

    # The last hidden-state is the first element of the output tuple
    embeddings = out[0][0]

    # You can get the embedding for [CLS] (which can be used for classification tasks) like this:
    cls_embedding = embeddings[0]

    # Or get the mean of all token embeddings (useful for sentence-level embeddings)
    mean_embedding = tf.reduce_mean(embeddings, axis=0)

    if class_embedding and mean_embedding is not None:  
        return tf.concat([cls_embedding, mean_embedding], 0)
    elif class_embedding:
        return cls_embedding
    elif mean_embedding is not None:
        return mean_embedding
    else:
        raise ValueError('At least one of class_embedding and mean_embedding must be True')

if __name__ == '__main__':
    text = "Apple Inc. is an American multinational technology company"
    embeddings = compute_embeddings(text)
    print(embeddings)
    print(embeddings.shape)