import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_skills(file_path):
    skills_df = pd.read_csv(file_path)
    return skills_df['skill'].values.tolist()

def load_resources(file_path):
    resources_df = pd.read_csv(file_path)
    return resources_df[['title', 'description']].values.tolist()

def match_skills(resources, skills):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = []

    for title, description in resources:
        content = f"{title} {description}"
        content_embedding = model.encode([content])
        
        skills_embeddings = model.encode(skills)
        cosine_similarities = cosine_similarity(content_embedding, skills_embeddings).flatten()
        related_skills_indices = cosine_similarities.argsort()[:-4:-1]

        for i in related_skills_indices:
            results.append({
                "resource title": title,
                "resource description": description,
                "skill": skills[i],
                "score": cosine_similarities[i]
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv('matches.csv', index=False)

skills = load_skills('skills.csv')
resources = load_resources('resources.csv')

match_skills(resources, skills)
