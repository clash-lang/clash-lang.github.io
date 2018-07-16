---
title: "Matrix multiplication with Clash"
date: "2018-07-09"
description: "Building, restructuring, and pipelining matrix multiplication with Clash"
disable_comments: false
author: "martijnbastiaan"
authorbox: true # Optional, enable authorbox for specific post
summary: hello
toc: true
mathjax: true
categories:
  - "Tutorial"
tags:
  - "Matrices"
  - "Design"
---

*Matrix multiplications happen to be useful in a very broad range of computational applications, such as computer graphics, artificial intelligence, and climate change research. At QbayLogic we help implement these (and more) applications on FPGAs using Clash. In this blogpost we will explore the intricacies of implementing matrix multiplications on FPGAs. We will explore the apparent differences between hardware and software development, how to use Clash to convert a “naive” algorithm to one suitable for an FPGA, and the use of Clash dependent types.*

*Our goal is to create a flexible yet efficient matrix multiplier. We want a pipelined architecture, polymorphic in its element type. That is, it should handle different number types (float, double, integers) even when the operations on these different types have different timing characteristics. We will see Clash is up for the task, providing a generic description for hardware polymorphic in timing and matrix dimensions. At the end of the series we’ll reflect on the experience and offer thoughts on the difficulties we encountered and how Clash could ease them in the future.*

<hr>

# Setting up
The source code corresponding to this blogpost (including cabal files, etc.) can be found at [github.com/clash-lang/TODO](https://github.com/clash-lang/TODO). Checkout the branch belonging to a certain stage in the tutorial. In order to execute the project, you need at least Cabal 2.2 and GHC 8.4. If your default compiler is GHC 8.4 you can simply run:

{{< highlight bash >}}
cabal new-run
{{< / highlight >}}

Otherwise, you would point Cabal to the compiler you wish to use:

{{< highlight bash >}}
cabal new-run --with-compiler=/opt/ghc/8.4.1/bin/ghc
{{< / highlight >}}

# Matrix multiplication
In order to define matrix multiplication, we first need to define a fundamental operation it uses: the dot-product. Given two vectors of equal length *a* and *b*, the dot product is:

$$ a \cdot b = \sum_i a_i b_i $$

or in Haskell:

{{< highlight haskell >}}
dot a b = sum (zipWith (*) a b)
{{< / highlight >}}

The matrix multiplication is then defined as:

$$ (AB( $$
