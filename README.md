# 💎 RefineAI

**RefineAI** is a data-centric framework designed to improve the reliability of machine learning models — especially in safety-critical domains such as healthcare.  
It integrates two complementary stages:

1. **Filtering** hard or problematic training instances using **Instance Hardness (IH)** or **Influence Functions (IF)**.  
2. **Rejecting** predictions during inference using a **confidence-based** or **uncertainty-based** criterion.

By combining these two steps, RefineAI enhances data quality, model confidence, and prediction reliability while maintaining a practical balance between accuracy and coverage.

The best combination of thresholds (*Tf*, *Tr*) — which determine the extent of filtering and rejection — is selected by minimizing a **cost function** that balances predictive performance, rejection rate, and model confidence.

---

## 📚 Case Studies

1. **Hospitalization_dengue** and **HSL** are real-world case studies conducted using routinely collected data from the **Brazilian Public Health System (SUS)**.  
2. **Heart**, **Lymph**, and **Bioresponse** are additional health-related datasets available on the [**OpenML**] repository and classified as *hard* datasets by [**TabZilla**].

---

For detailed information, please refer to our **preprint (ArXiv link to be added)**.

If this repository was useful to you, please consider citing our work as:

> Valeriano, M. G., Marzagão, D. K., Montelongo, A., Kiffer, C. R. V., Katz, N., & Lorena, A. C. (2025).  
> *Filtering instances and rejecting predictions to obtain reliable models in healthcare.*  
> Preprint available at ArXiv.

---

### 🔗 References
- **OpenML**: Vanschoren, J., et al. (2014). *OpenML: Networked Science in Machine Learning.* ACM SIGKDD Explorations, 15(2), 49–60. 
- **TabZilla**: Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* Advances in neural information processing systems, 35, pp.507-520.

---


