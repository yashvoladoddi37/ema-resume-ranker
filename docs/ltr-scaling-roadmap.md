# Scaling Roadmap: From Heuristics to Neural Networks (LTR)

## 1. The Current State: Heuristic Weighting
Currently, the system uses **Heuristic Weighting**:
```python
Final Score = (0.6 * LLM) + (0.4 * Deterministic)
```
While effective for a POC, this relies on "expert intuition" rather than data-driven optimization.

## 2. The Scale-up Architecture: Learning to Rank (LTR)
At enterprise scale (thousands of resumes), we transition to **Neural Networks** or **Gradient Boosted Decision Trees (GBDT)** to learn the optimal weights.

### Phase A: Feature Engineering
Instead of a single score, the scores from our current components become **Features**:
- `f1`: LLM Relevance Score
- `f2`: Years of Experience (Deterministic)
- `f3`: Skill Coverage % (Deterministic)
- `f4`: Embedding Cosine Similarity (Semantic)
- `f5`: Role Seniority Match (Derived)

### Phase B: Data Collection & Labeling
We collect **Judgment Data**:
- Pairs of `(Job Description, Resume)` with a ground truth label (0 to 4 scale).
- These labels come from actual Recruiter clicks, "Move to Interview" actions, or manual expert labeling.

### Phase C: Model Training
We train a model (e.g., **LambdaRank**, **XGBoost**, or a **Simple Feed-Forward Neural Network**) to optimize for a listwise loss function.

#### Objective Functions:
The model doesn't just predict a score; it optimizes to ensure the "Best" candidate is at the top.
- **nDCG (Normalized Discounted Cumulative Gain)**: Rewards the model for putting high-quality items at the top of the list.
- **MRR (Mean Reciprocal Rank)**: Penalizes the model based on how far down the first relevant item is.

## 3. Why Neural Networks?
1. **Non-Linear Relationships**: A neural network can learn that 2 years of experience is "enough" and 10 years isn't 5x better—something a linear weight (0.4 * years) struggles with.
2. **Cross-Feature Interaction**: It can learn that a high "Semantic Similarity" is only valuable if the "Deterministic Skills" match.
3. **Continuous Optimization**: As more recruiters use the system, their decisions (click/reject) provide a constant stream of training data to refine the weights automatically.

## 4. Evaluation at Scale
At this stage, we stop looking at individual candidates and start looking at **System Performance Metrics**:
- **Precision@5**: On average, how many of the top 5 candidates are actually "Good"?
- **Recall@K**: Did we find all "Good" candidates in the top `K` results?
- **nDCG**: Is our 1st result better than our 2nd, and our 2nd better than our 3rd?
