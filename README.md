<div align="center">

# 🎨 PosterCraft

**Rethinking High-Quality Aesthetic Poster Generation in a Unified Framework**

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXX-red)](https://arxiv.org/abs/XXXX)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/ephemeral182/PosterCraft)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/PosterCraft)
[![Demo](https://img.shields.io/badge/🎥-Demo-green)](https://ephemeral182.github.io/PosterCraft/)

<img src="images/logo/logo.png" alt="PosterCraft Logo" width="200"/>

*From your prompts to high-quality aesthetic posters*

[**🎯 Demo**](https://ephemeral182.github.io/PosterCraft/) | [**📄 Paper**](https://arxiv.org/abs/XXXX) | [**🤗 Models**](https://huggingface.co/PosterCraft) | [**📚 Datasets**](https://huggingface.co/datasets/PosterCraft)

</div>

---

## 👥 Authors

[**Sixiang Chen**](https://ephemeral182.github.io/)¹\*, [**Jianyu Lai**](https://openreview.net/profile?id=~Jianyu_Lai1)¹\*, [**Jialin Gao**](https://scholar.google.com/citations?user=sj4FqEgAAAAJ&hl=zh-CN)²\*, [**Tian Ye**](https://owen718.github.io/)¹, [**Haoyu Chen**](https://haoyuchen.com/)¹, [**Hengyu Shi**](https://openreview.net/profile?id=%7EHengyu_Shi1)², [**Shitong Shao**](https://shaoshitong.github.io/)¹, [**Yunlong Lin**](https://scholar.google.com.hk/citations?user=5F3tICwAAAAJ&hl=zh-CN)³, [**Song Fei**](https://openreview.net/profile?id=~Song_Fei1)¹, [**Zhaohu Xing**](https://ge-xing.github.io/)¹, [**Yeying Jin**](https://jinyeying.github.io/)⁴, **Junfeng Luo**², [**Xiaoming Wei**](https://scholar.google.com/citations?user=JXV5yrZxj5MC&hl=zh-CN)², [**Lei Zhu**](https://sites.google.com/site/indexlzhu/home)¹'⁵†

¹The Hong Kong University of Science and Technology (Guangzhou)  
²Meituan  
³Xiamen University  
⁴National University of Singapore  
⁵The Hong Kong University of Science and Technology  

\*Equal Contribution, †Corresponding Author

---

## 🌟 What is PosterCraft?

PosterCraft excels in **precise text rendering**, **seamless integration of abstract art**, **striking layouts**, and **stylistic harmony**. Our unified framework addresses four critical stages of aesthetic poster generation:

- ✨ **Text Rendering Optimization**: Accurate text generation with diverse styles
- 🎨 **High-quality Poster Fine-tuning**: Artistic integrity preservation  
- 🎯 **Aesthetic-Text RL**: Higher-order aesthetic trade-offs
- 🔄 **Vision-Language Feedback**: Multi-modal feedback refinement

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ephemeral182/PosterCraft.git
cd PosterCraft

# Create conda environment
conda create -n postercraft python=3.11
conda activate postercraft

# Install dependencies
pip install -r requirements.txt
```

## 📊 Performance

PosterCraft achieves state-of-the-art performance across multiple dimensions:

| Method | Text Recall ↑ | Text F-score ↑ | Text Accuracy ↑ |
|--------|---------------|----------------|-----------------|
| Flux1.dev | 0.723 | 0.707 | 0.667 |
| Ideogram-v2 | 0.711 | 0.685 | 0.680 |
| Gemini2.0-Flash-Gen | 0.798 | 0.786 | 0.746 |
| **PosterCraft (ours)** | **0.787** | **0.774** | **0.735** |

## 🎭 Gallery

<div align="center">
<table>
<tr>
<td align="center"><img src="images/gallery/gallery_demo1.png" width="200"><br><b>Adventure Travel</b></td>
<td align="center"><img src="images/gallery/gallery_demo2.png" width="200"><br><b>Post-Apocalyptic</b></td>
<td align="center"><img src="images/gallery/gallery_demo3.png" width="200"><br><b>Sci-Fi Drama</b></td>
</tr>
<tr>
<td align="center"><img src="images/gallery/gallery_demo4.png" width="200"><br><b>Space Thriller</b></td>
<td align="center"><img src="images/gallery/gallery_demo5.png" width="200"><br><b>Cultural Event</b></td>
<td align="center"><img src="images/gallery/gallery_demo6.png" width="200"><br><b>Luxury Product</b></td>
</tr>
</table>
</div>

## 🏗️ Model Architecture

Our unified framework consists of four critical optimization stages:

📝 Text Rendering → 🎨 Poster Fine-tuning → 🎯 Aesthetic-Text RL → 🔄 Vision-Language Feedback

Each stage builds upon the previous one to ensure both text accuracy and aesthetic quality.

## 📚 Datasets

We provide four specialized datasets for training:

### 🔤 Text-Render-2M
- **2 million** high-quality text rendering examples
- Multi-instance text rendering with diverse selections
- Dynamic content generation through template-based and random approaches

### 🎨 HQ-Poster-100K  
- **100,000** meticulously curated high-quality posters
- Advanced filtering techniques and multi-modal scoring
- Gemini-powered mask generation with detailed captions

### 👍 Poster-Preference-100K
- **100,000** preference learning poster pairs
- Comprehensive evaluation by Gemini and aesthetic evaluators
- Human-aligned poster generation training

### 🔄 Poster-Reflect-120K
- **120,000** poster pairs with text reflections
- Vision-language feedback refinement
- Aesthetic style analysis and content evaluation

## 📝 Citation

If you find PosterCraft useful for your research, please cite our paper:

```bibtex
@article{chen2024postercraft,
  title={PosterCraft: Rethinking High-Quality Aesthetic Poster Generation in a Unified Framework},
  author={Chen, Sixiang and Lai, Jianyu and Gao, Jialin and Ye, Tian and Chen, Haoyu and Shi, Hengyu and Shao, Shitong and Lin, Yunlong and Fei, Song and Xing, Zhaohu and Jin, Yeying and Luo, Junfeng and Wei, Xiaoming and Zhu, Lei},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors and researchers who made this work possible
- Special thanks to the open-source community for inspiration and support

---

<div align="center">
Made with ❤️ by the PosterCraft Team
</div>