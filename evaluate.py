import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import torch



def evaluate_model(model, dataloader, top_k=30):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings, labels = [], []

    # Extract embeddings
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            output = model(images)

            embedding = output[0] if isinstance(output, tuple) else output
            embedding = embedding.view(embedding.size(0), -1)

            embeddings.append(embedding.cpu().numpy())
            labels.extend(label.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    all_labels, all_predictions = [], []
    lrap_relevance, lrap_similarity = [], []
    precisions, recalls = [], []

    # Compute metrics for each query
    for i in range(len(embeddings)):
        query_embedding = embeddings[i].reshape(1, -1)
        query_label = labels[i]

        similarities = cosine_similarity(query_embedding, embeddings)[0]

        top_k_indices = similarities.argsort()[-top_k:][::-1]
        retrieved_labels = labels[top_k_indices]

        all_labels.append(query_label)
        all_predictions.append(retrieved_labels[0])

        relevance = (retrieved_labels == query_label).astype(int)
        lrap_relevance.append(relevance)
        lrap_similarity.append(similarities[top_k_indices])

        relevant_retrieved = np.sum(relevance)
        total_relevant = np.sum(labels == query_label)

        precision = relevant_retrieved / top_k
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 1.0

        precisions.append(precision)
        recalls.append(recall)

    # Calculate Metrics
    accuracy = np.sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean([2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)])

    # Compute LRAP
    lrap_relevance = np.array(lrap_relevance)
    lrap_similarity = np.array(lrap_similarity)
    try:
        lrap = label_ranking_average_precision_score(lrap_relevance, lrap_similarity)
    except ValueError:
        lrap = 0.0

    return precision, recall, f1, accuracy, lrap
