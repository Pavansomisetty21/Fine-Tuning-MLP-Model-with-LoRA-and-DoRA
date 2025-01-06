# Fine-Tuning-MLP-Model-with-LoRA-and-DoRA

In this we fine-tune MLP with LoRA and DoRA on CIFAR-10 Dataset


LoRA (Low-Rank Adaptation) [paper](https://arxiv.org/pdf/2106.09685) and DoRA (Weight-Decomposed Low-Rank Adaptation) [paper](https://arxiv.org/pdf/2402.09353) are techniques designed to fine-tune large pre-trained language models (PLMs) efficiently. Both aim to reduce the computational and storage costs of fine-tuning while maintaining or improving performance. Here's an overview:

---

### **1. LoRA (Low-Rank Adaptation)**  

**Key Idea:**  
LoRA introduces a low-rank decomposition method for adapting pre-trained language models (PLMs). Instead of fine-tuning all the parameters of a PLM, LoRA focuses on a low-rank subset of parameters within the model. 


#### **How It Works:**
- **Parameter Decomposition:** LoRA assumes that the weight updates required for fine-tuning lie in a low-rank subspace. It represents these updates as the product of two low-rank matrices \( A \) and \( B \), where \( A \) and \( B \) are much smaller than the original parameter matrix.
- **Frozen Base Model:** The original weights of the PLM remain frozen. LoRA injects the low-rank weight updates into the model dynamically during forward passes.
- **Efficiency:** This approach drastically reduces the number of trainable parameters, leading to lower memory requirements and faster training.

#### **Advantages of LoRA:**
- Reduces memory and computation costs significantly.
- Does not require full fine-tuning of all model weights, which is useful for large-scale PLMs like GPT or T5.
- Simplifies storage and deployment of adapted models since only the low-rank matrices \( A \) and \( B \) need to be saved.

---

### **2. DoRA (Weight-Decomposed Low-Rank Adaptation)**

**Key Idea:**  
DoRA builds on LoRA by introducing a more efficient weight-decomposition approach for low-rank adaptation. It aims to further improve the parameter efficiency and performance of LoRA.

#### **How It Works:**
- **Weight Decomposition:** DoRA decomposes the weight matrices of the model into smaller components, focusing on parameter sparsity and rank-reduction simultaneously.
- **Dynamic Weight Allocation:** It optimizes which weights to adapt and decomposes them to achieve a balance between performance and computational efficiency.
- **Granular Adaptation:** DoRA often leverages task-specific decomposition techniques, making it more adaptable to diverse downstream tasks.

#### **Advantages of DoRA:**
- Enhanced performance over LoRA in tasks requiring more granular fine-tuning.
- Higher compression ratios due to better decomposition techniques.
- Improved task-specific adaptation through weight-decomposition strategies.

---

### **Comparison:**

| Feature               | LoRA                              | DoRA                              |
|-----------------------|-----------------------------------|-----------------------------------|
| **Core Mechanism**    | Low-rank weight updates.          | Weight decomposition and low-rank adaptation. |
| **Efficiency**        | High efficiency in reducing trainable parameters. | Higher efficiency with better compression. |
| **Task Adaptability** | Suitable for general tasks.       | Better for task-specific scenarios. |
| **Performance**       | Good performance with fewer parameters. | Often outperforms LoRA in downstream tasks. |

---

Both methods are powerful tools for adapting large-scale language models in a resource-efficient manner. LoRA is simpler and widely used, while DoRA offers enhancements for more fine-grained and efficient adaptations.


Here’s a deeper dive into **LoRA** and **DoRA** with additional details and context:

---

### **1. LoRA (Low-Rank Adaptation)**

#### **Motivation:**
- Pre-trained language models (PLMs) like GPT-3, T5, or BERT have billions of parameters, making full fine-tuning costly in terms of computation and storage.
- In many cases, fine-tuning involves small changes to model weights, which can often be represented in a **low-rank subspace**. LoRA leverages this observation.

#### **Benefits in Practice:**
- **Reduced Storage:** Instead of saving a full fine-tuned model, only the low-rank matrices \( A \) and \( B \) are stored, which are orders of magnitude smaller than \( W \).
- **Scalability:** Enables fine-tuning even on resource-constrained devices.
- **Transferability:** LoRA fine-tuned updates can be reused across tasks with similar characteristics.

#### **Limitations:**
- **Task Sensitivity:** Performance depends on the rank \( r \), which needs careful tuning.
- **Selective Application:** Requires determining which layers benefit most from low-rank adaptation.

---

### **2. DoRA (Weight-Decomposed Low-Rank Adaptation)**

#### **Motivation:**
- While LoRA provides efficient fine-tuning, its approach is uniform and might not fully exploit task-specific nuances in the weight matrices.
- DoRA improves efficiency by **decomposing weights into components** before applying low-rank adaptation.

#### **Advantages in Practice:**
- **Higher Parameter Efficiency:** More granular decomposition reduces redundancy in updates, enabling even smaller fine-tuned models.
- **Task-Specific Adaptation:** By adapting decomposed components selectively, DoRA captures task-specific nuances better than LoRA.
- **Improved Generalization:** By operating on decomposed weights, DoRA often achieves better generalization across related tasks.

#### **Use Cases:**
- **Low-Resource Scenarios:** Fine-tuning large models on low-resource tasks, where every parameter matters.
- **Domain Adaptation:** Adapting PLMs to specialized domains like healthcare, finance, or law.

---

### **Comparison with Other Fine-Tuning Methods**

#### **Fine-Tuning Approaches Overview:**

| Method                   | Trainable Parameters | Storage Overhead | Performance | Scalability |
|--------------------------|----------------------|------------------|-------------|-------------|
| **Full Fine-Tuning**     | 100%                | High             | High        | Low         |
| **Adapter Layers**       | Small percentage    | Medium           | Medium      | High        |
| **LoRA**                 | Very small          | Low              | High        | Very High   |
| **DoRA**                 | Small but task-tuned | Very Low         | Very High   | High        |

#### **Strengths of LoRA and DoRA:**
- Both LoRA and DoRA align with the modern need to fine-tune **gigantic PLMs** on specific tasks or domains efficiently.
- They enable **personalized AI systems** by allowing individual users or organizations to adapt massive models to their data without the need to train from scratch.

---

### **Future Directions**
- **Dynamic Rank Selection:** Automatically determining the best rank \( r \) based on task complexity.
- **Hybrid Methods:** Combining LoRA/DoRA with other efficient fine-tuning methods, like adapters or prefix-tuning.
- **Task-Driven Sparsity:** Leveraging sparsity in weight matrices to further compress models without sacrificing performance.

Both **LoRA** and **DoRA** are pivotal in making large language models adaptable and accessible, paving the way for broader adoption in real-world applications.



Here’s an even deeper exploration of **LoRA (Low-Rank Adaptation)** and **DoRA (Weight-Decomposed Low-Rank Adaptation)** with additional technical details, potential enhancements, and related concepts:

---

### **3. Extended Details on LoRA (Low-Rank Adaptation)**

#### **Theoretical Background:**
LoRA is built upon the assumption that the required fine-tuning updates can be expressed in a **low-rank approximation** of the weight matrix. This is grounded in the fact that a large proportion of model weights in a pre-trained language model (PLM) often contribute little to task-specific updates. These small, task-specific changes can typically be captured in low-rank matrices.

**Key Insights from Linear Algebra:**
- When you apply low-rank decomposition, the model essentially learns a **low-rank subspace** that best represents the weight updates needed for the task. This significantly reduces the number of parameters that must be adapted.
- For a matrix \( W \) of dimension \( m \times n \), decomposing \( W \) into matrices \( A \) (of dimension \( m \times r \)) and \( B \) (of dimension \( r \times n \)) where \( r \ll \min(m, n) \) ensures that we are only learning a few parameters (the rank \( r \)).

#### **LoRA in Practice:**
- **Modular Integration:** LoRA can be applied to specific layers, like attention heads or feed-forward layers, without affecting other parts of the model.
- **Multi-Layer Adaptation:** In multi-layer architectures like transformers, LoRA can be selectively applied to attention layers, feed-forward networks, and even the output heads, allowing fine-grained control over which components are fine-tuned.
  
  This gives flexibility in applications where we may want to reduce computation further (e.g., applying LoRA only to attention layers in certain tasks).

#### **Considerations and Challenges:**
- **Hyperparameter Tuning:** One of the key challenges of LoRA is setting the **rank \( r \)** correctly. Too small a rank may lead to insufficient adaptation, while too large a rank can negate the benefits of low-rank approximation.
- **Domain Specificity:** While LoRA works for many tasks, its performance may be less optimal for tasks requiring complex, domain-specific adaptations, where other methods (like DoRA) could be better suited.

---

### **4. Deep Dive into DoRA (Weight-Decomposed Low-Rank Adaptation)**

#### **Theoretical Foundation of DoRA:**
DoRA goes a step further by introducing **weight decomposition** strategies that focus on understanding which parts of the weight matrix are most relevant for adaptation in a **domain-specific** manner. This approach is designed to take into account the latent structure of the weights and adapt them accordingly, leading to more efficient adaptation and better performance.

**Key Concepts:**
1. **Latent Structure Understanding:**
   - DoRA looks deeper into the weight matrices by decomposing them into **singular components** (using SVD or other matrix factorization techniques) to identify and adapt only the most important components.
   - Unlike LoRA, which applies low-rank updates uniformly, DoRA selectively applies rank updates to the most influential components of the matrix. This improves **parameter efficiency** and **adaptation specificity**.
  
2. **Task-Specific Adaptation:**
   - **Granular Adaptation** in DoRA allows fine-tuning of individual parameters, enhancing adaptation on specialized tasks like medical or legal NLP, where domain-specific knowledge is crucial.
   - DoRA is often coupled with **sparsity** techniques that allow only a subset of parameters to be updated during fine-tuning. This is especially helpful when dealing with niche tasks that don't require extensive weight updates across the entire model.

#### **Optimizing with Weight Decomposition:**
- **Singular Value Decomposition (SVD):** DoRA often uses SVD to factorize the weight matrix into three components: \( U, \Sigma, V \). It can then fine-tune only the most significant singular values in \( \Sigma \), representing the most impactful directions in the parameter space.
  
  This selective fine-tuning is more computationally efficient compared to traditional fine-tuning and can lead to better generalization on downstream tasks.

#### **DoRA and Large-Scale Models:**
- In large models (e.g., GPT, BERT), weight decomposition ensures that only the essential parts of the model are updated, making **DoRA scalable** to billions of parameters.
- By reducing the number of trainable parameters during fine-tuning, DoRA enables training on smaller datasets and lower-end hardware.

#### **Limitations of DoRA:**
- **Higher Complexity:** The method of weight decomposition itself can be computationally expensive, especially in very large models. The process of identifying which components of the weight matrix should be adapted may require additional complexity.
- **Task Dependency:** As with LoRA, the performance boost from DoRA depends on how well the decomposition fits the task. For some tasks, the benefits might be marginal if the model does not exhibit a strong latent structure.

---

### **5. Advanced Enhancements and Hybrid Methods**

Both LoRA and DoRA have opened the door to **innovative combinations** with other techniques for even more efficient adaptation of PLMs:

#### **LoRA + Adapters:**
- **Adapters** are small neural modules added to transformer layers that help the model adapt to a new task without modifying the pre-trained weights. Combining LoRA with adapters allows the model to learn task-specific representations while retaining the low-rank structure of weight updates.
- **Advantages:** Reduces the need for significant adaptation in the full model while benefiting from task-specific modules (adapters).

#### **LoRA + Prompt Tuning:**
- **Prompt tuning** involves modifying the input prompt to guide the model's response. Combining LoRA with prompt tuning can reduce the need for fine-tuning larger portions of the model, focusing adaptation solely on the low-rank space while also optimizing prompt formulations.

#### **DoRA + Knowledge Injection:**
- **Knowledge injection** involves introducing domain-specific knowledge (e.g., medical, legal, financial) into the training process. Combining DoRA with knowledge injection could enable more **task-specific weight decomposition**, making the low-rank adaptation even more targeted and efficient.

---

### **6. Applications of LoRA and DoRA**

#### **Low-Resource Tasks:**
- **Fine-Tuning on Small Datasets:** Both methods excel in scenarios where training data is limited. LoRA and DoRA allow the model to be adapted without overfitting or requiring vast amounts of data.
- **Transfer Learning in Niche Domains:** For tasks like medical NLP (e.g., clinical text understanding) or financial sentiment analysis, LoRA and DoRA can provide high efficiency in adapting large pre-trained models to specific tasks.

#### **Real-Time Inference:**
- **Efficiency in Deployment:** Due to the reduced number of parameters being adapted, models fine-tuned with LoRA or DoRA can be deployed more efficiently, requiring fewer resources for inference, which is crucial in edge computing or on-device applications.

#### **AI for Specialized Industries:**
- **Legal:** In the legal domain, DoRA’s task-specific fine-tuning can help models adapt to the jargon, norms, and nuances of legal documents.
- **Medical AI:** In healthcare, where domain-specific knowledge is essential, DoRA can be used to adapt large models like BioBERT for medical coding, diagnosis prediction, or radiology report generation.

---

### **Future Directions:**

- **Meta-Learning and Hyperparameter Optimization:** Meta-learning approaches could be integrated with LoRA and DoRA to automatically select the optimal rank \( r \) or weight decomposition scheme for a given task, reducing the manual tuning required.
- **End-to-End Optimizers:** Optimizing the entire fine-tuning process with end-to-end differentiable optimizers for both LoRA and DoRA could improve convergence rates and model performance.
- **Hybrid Decomposition:** Combining both **low-rank matrix approximation** (from LoRA) and **weight decomposition** (from DoRA) could lead to new hybrid approaches that dynamically adjust based on task complexity and model size.

---

### **Conclusion:**

LoRA and DoRA represent cutting-edge techniques in **efficient fine-tuning** of large pre-trained models. By focusing on low-rank adaptations and weight decomposition, they offer powerful alternatives to traditional fine-tuning methods, enabling both **efficient model deployment** and **high performance on specialized tasks**. As the field continues to evolve, combining these techniques with other advanced methods, like **sparsity**, **transfer learning**, and **meta-learning**, promises even more breakthroughs in NLP and beyond.


For fine-tuning purpose we use CIFAR-10 (Canadian Institute for Advanced Research 10) dataset it is



**CIFAR-10 (Canadian Institute for Advanced Research 10)** is a well-known dataset in the field of machine learning, specifically used for training and evaluating image classification algorithms. It is often used as a benchmark dataset for evaluating the performance of models in computer vision tasks.

### **Key Features of CIFAR-10:**

1. **Dataset Overview:**
   - CIFAR-10 consists of **60,000 color images** divided into **10 classes**, with each class containing **6,000 images**.
   - The images are **32x32 pixels** in size and are in **RGB color format** (3 channels).
   - It is a **10-class classification problem**, where each image belongs to one of the following categories:
     - **Airplane**
     - **Automobile**
     - **Bird**
     - **Cat**
     - **Deer**
     - **Dog**
     - **Frog**
     - **Horse**
     - **Ship**
     - **Truck**

2. **Data Split:**
   - **Training Set:** 50,000 images (5,000 images per class).
   - **Test Set:** 10,000 images (1,000 images per class).
   
3. **Image Characteristics:**
   - The images in CIFAR-10 are small (32x32 pixels), which makes it a good dataset for evaluating models that need to generalize to lower resolution images.
   - Despite the small size of the images, the dataset contains a variety of objects, backgrounds, and lighting conditions, which poses a challenge for classification tasks.

4. **Usage:**
   - **Benchmark for Machine Learning Algorithms:** CIFAR-10 is frequently used as a standard benchmark dataset to test various machine learning algorithms, especially in the context of deep learning (e.g., convolutional neural networks or CNNs).
   - **Object Recognition and Classification:** It helps evaluate the performance of models in classifying different objects, making it a useful dataset for developing and testing classification models.

5. **Accessing the Dataset:**
   - The CIFAR-10 dataset is freely available and can be easily accessed through libraries like TensorFlow, PyTorch, or Keras.

### **Why CIFAR-10 is Popular:**
- **Ease of Use:** CIFAR-10 is a relatively small and manageable dataset for experimentation, making it ideal for testing new algorithms without requiring extensive computational resources.
- **Well-Established Benchmark:** It is widely used in the research community, so it provides a useful point of comparison for new algorithms.
- **Classification Task Variety:** The dataset covers a variety of object categories, allowing the testing of models on a diverse set of images.

### **Limitations:**
- **Low Resolution:** The 32x32 pixel resolution is quite low compared to more complex datasets like ImageNet, which has images at much higher resolutions (e.g., 224x224).
- **Class Imbalance:** Although the CIFAR-10 dataset is well-balanced, some classes may still be harder to classify due to differences in object size and background clutter.

we fine-tune this dataset using MLP with LoRA and DoRA 

# Fine-Tuning Results

after fine-tuning we got the results as 

Accuracy of original model: **44.49%**

Test accuracy LoRA finetune: **46.38%**

Test accuracy DoRA finetune: **47.51%**

