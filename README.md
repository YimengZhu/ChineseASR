# ChinesASR

This repo contains the pytorch implementation of [Deep Speech 2](https://arxiv.org/pdf/1512.02595.pdf), [SAN-CTC](https://arxiv.org/pdf/1901.10055.pdf), [gated convnets](https://arxiv.org/pdf/1712.09444.pdf) with data augmentation from [SpectAugmentation](https://arxiv.org/pdf/1904.08779.pdf) and [this paper](https://arxiv.org/pdf/1910.13296.pdf). I also applied some tricks from [this paper](https://arxiv.org/pdf/1904.13377.pdf), and batch the data with buckets as proposed in [this paper](https://arxiv.org/pdf/1705.02414.pdf).

All the implementation may contain modifications to the original paper to fit my own case. If you have a problem, simply leave me a message on issue or send me an email. I'm too lazy to write everything down in this readme but I'm kind enough to answer your individual message.

Although the name of this repo is ChineseASR, you can run the code for any language you want. I call this name because I just created the script for downloading and preparing some open Chinese dataset. For other languages like English you can simply find thousands of scripts on GitHub.
