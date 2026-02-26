||Key points| Note| Link|
|---|---|---|---|
DewarpNet | refinement network | | |
BEDSR-Net | shadow removal | 19.8M parameters| [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_BEDSR-Net_A_Deep_Shadow_Removal_Network_From_a_Single_Document_CVPR_2020_paper.pdf) |
Sauvola | | algorithm| |
Niblack | | algorithm| |
NUnet | | no read access yet | [Link](https://www.researchgate.net/publication/377100444_A_Lightweight_NUet_Model_for_Document_Image_Restoration)|
MobileIE |  Low-Light Enhancement (LLE), Underwater Image Enhancement (UIE), and end-to-end Image Signal Processing (ISP).| 4K parameters ???| [Link](https://arxiv.org/pdf/2507.01838)|
LiR | Image Restoration, denoise|  7.31 params| [Link](https://arxiv.org/pdf/2402.01368)|
| | | | [Link](https://arxiv.org/pdf/2406.00629)|
|||| [Link](https://arxiv.org/pdf/2401.11831)|
|||| [Link](https://link.springer.com/chapter/10.1007/978-981-19-1673-1_21)|
|High-Resolution Document Shadow Removal via A Large-Scale Real-World
Dataset and A Frequency-Aware Shadow Erasing Net||| [Link](https://arxiv.org/pdf/2308.14221)
DvD: Unleashing a Generative Paradigm for Document Dewarping via Coordinates-based Diffusion Model|||https://arxiv.org/pdf/2505.21975
Efficient Document Image Dewarping via Hybrid Deep Learning and Cubic Polynomial Geometry Restoration|Dùng YOLOv8 detect 4 góc + boundary → sau đó fit cubic polynomial surface để tạo grid warp. Kết hợp deep learning (detection) + classical CV (polynomial fitting) → rất nhanh, ít parameter, robust với text-poor documents.||https://arxiv.org/pdf/2501.03145
TADoc: Robust Time-Aware Document Image Dewarping|||https://arxiv.org/pdf/2508.06988
DocMamba: Robust Document Image Dewarping via Selective State Space Sequence Modeling|||
BookNet: Dual-Page Book Image Rectification via Cross-Page Attention|Chuyên xử lý dual-page book spread (left-right page asymmetric deformation do binding).||https://arxiv.org/pdf/2601.21938
ForCenNet: Foreground-Centric Network for Document Image Rectification|||https://arxiv.org/pdf/2507.19804
DocAttentionRect: Attention-Guided Document Image Rectification|||https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_39/paper.pdf
DocCPLNet: Document Image Rectification via Control Point and Illumination Correction|||https://pmc.ncbi.nlm.nih.gov/articles/PMC12567372/
DocPINN: A Neural PDE-Based Framework for Document Image Dewarping|Thêm physics constraint (smoothness, curvature prior) vào loss.||https://dl.acm.org/doi/10.1007/978-3-032-04627-7_22
