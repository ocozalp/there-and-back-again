# Reading list

Every reference used in the project, grouped by the phase that first needs it, with one sentence
on why it matters. Chapters link here on first mention.

## Foundations (Phases 0 to 2)

- [3Blue1Brown, Neural networks series](https://www.3blue1brown.com/topics/neural-networks): visual intuition for gradient descent and backpropagation before any code.
- [Karpathy, The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0): the video Phase 1 follows; the [micrograd repository](https://github.com/karpathy/micrograd) is the reference implementation.
- [PyTorch, A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html): the library counterpart to Phase 1.
- [PyTorch, Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html): how the real engine records and frees graphs.
- [Goodfellow, Bengio, Courville, Deep Learning](https://www.deeplearningbook.org/): the reference text; chapter 6 covers feedforward networks and backprop.
- [Prince, Understanding Deep Learning](https://udlbook.github.io/udlbook/): a newer, more visual reference text with free PDF.
- [CS231n notes](https://cs231n.github.io/): practical optimization, initialization, and CNN notes; the [backprop notes](https://cs231n.github.io/optimization-2/) are the best short treatment.

## Language models (Phases 3 and 4)

- [Karpathy, makemore series](https://www.youtube.com/watch?v=PaCmpygFfXo): bigram and MLP language models, the path Phase 3 takes.
- [Bengio et al. 2003, A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf): the paper behind the Phase 3 MLP and the idea of learned embeddings.
- [Karpathy, Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY): Phase 4's spine.
- [Karpathy, Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE): byte-pair encoding from nothing.
- [Karpathy, nanoGPT](https://github.com/karpathy/nanoGPT): the compact reference GPT.
- [Vaswani et al. 2017, Attention Is All You Need](https://arxiv.org/abs/1706.03762): the transformer paper.
- [Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): the diagrams to have in mind while reading the paper.
- [Diary of Samuel Pepys, Project Gutenberg](https://www.gutenberg.org/ebooks/4200): the public-domain diary the language models pretrain on.

## Vision and vision-language (Phases 5 and 6)

- [He et al. 2015, Deep Residual Learning](https://arxiv.org/abs/1512.03385): residual connections, which every later architecture uses.
- [Dosovitskiy et al. 2020, An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929): the Vision Transformer.
- [Kingma and Welling 2013, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): the VAE, which becomes the latent space in Phase 8.
- [Radford et al. 2021, Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020): CLIP.
- [Li et al. 2022, BLIP](https://arxiv.org/abs/2201.12086): a production captioner for comparison.
- [Liu et al. 2023, Visual Instruction Tuning](https://arxiv.org/abs/2304.08485): LLaVA, how image patches enter an LLM.

## Generative models (Phases 7 and 8)

- [Ho et al. 2020, Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239): DDPM.
- [Song et al. 2020, Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502): DDIM, deterministic sampling.
- [Ho and Salimans 2022, Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598): the guidance scale knob.
- [Weng, What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): the clearest derivation of the DDPM math.
- [Hugging Face, The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion): DDPM in code with commentary.
- [Lipman et al. 2022, Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747): the flow-matching objective.
- [Lipman et al. 2024, Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264): a tutorial-length treatment with code.
- [Esser et al. 2024, Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206): Stable Diffusion 3, flow matching at scale.
- [Rombach et al. 2021, High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752): latent diffusion, the Stable Diffusion paper.
- [Alammar, The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/): the anatomy diagram Phase 8 rebuilds.
- [diffusers documentation](https://huggingface.co/docs/diffusers): the production library for Phases 7 to 10.

## Customization and control (Phases 9 and 10)

- [Hu et al. 2021, LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): the LoRA paper.
- [peft documentation](https://huggingface.co/docs/peft): the production LoRA library.
- [Gal et al. 2022, An Image is Worth One Word](https://arxiv.org/abs/2208.01618): textual inversion.
- [Ruiz et al. 2022, DreamBooth](https://arxiv.org/abs/2208.12242): subject-driven fine-tuning.
- [Zhang et al. 2023, Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543): ControlNet.
- [Ye et al. 2023, IP-Adapter](https://arxiv.org/abs/2308.06721): image-prompt conditioning.
- [Hertz et al. 2022, Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626): editing by steering attention.
