import numpy as np

if st.button("Predict"):
    X = tfidf.transform([text])
    
    scores = clf.decision_function(X)[0]
    classes = clf.classes_
    
    top3 = np.argsort(scores)[-3:][::-1]
    
    st.subheader("Top Predictions")
    for i in top3:
        st.write(f"{classes[i]}  (score: {scores[i]:.2f})")
