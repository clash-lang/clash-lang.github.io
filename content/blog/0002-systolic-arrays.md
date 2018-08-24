---
title: "Building systolic arrays with Clash"
date: "2018-07-25"
description: "Building a "
disable_comments: false
author: "martijnbastiaan"
authorbox: true # Optional, enable authorbox for specific post
summary: hello
toc: true
mathjax: false
categories:
  - "Tutorial"
tags:
  - "Systolic arrays"
  - "Design"
---

Systolic arrays are networks of locally coupled processing elements, continuously receiving and sending their inputs and outputs from and to their neighbors. They cannot access main memory or global buses, thus allowing them to keep critical paths short. Because of this, they are extremely good at solving problems in the field of image processing, artificial intelligence, and computer vision. This blogpost will take a look at how to build systolic arrays with Clash and subsequently build a matrix multiplier with it. If you're new to Clash or matrix multiplication, [read this blogpost first](/blog/0001-matrix-multiplication/).

# Concepts

Let's first have a look at a simple systolic array where each processing element has an input from their left and upper neighbor, and an output to their right and bottom neighbor. It looks like:

<center><img src="/blog/0002-systolic-arrays/LeftTopPE.svg"></img></center>

Processing elements can do arbitrary things with their inputs and pass arbitrary things as their outputs, but we'll keep it simple for now. The following example consists of processing elements which each simply pass along the data they receive from their left neighbour to their right neighbor. Similarly, they pass their top input to their bottom neighbor. Simulating a grid of three by three for a total of nine processing elements looks like:

<div class="sysarray" id="mm0">
    <center>
        <table></table>
        <button class="next">Next</button>
        <button class="reset">Reset</button>
    </center>
</div>

<br/>

Yellow elements move to the right every cycle, while blue ones move to the bottom. As can be seen, not every processing element can has two (valid) inputs at all points in the simulation. This is not always an issue, but many systolic array applications would like `G` and `8` to end up in the same cell at the same time. By delaying the inputs strategically, this ends up being true:


<div class="sysarray" id="mm1">
    <center>
        <table></table>
        <button class="next">Next</button>
        <button class="reset">Reset</button>
    </center>
</div>

<br/>

In fact, all values of the yellow rows end up in the same cell (at different times) as the values in the blue columns. I that makes you think of matrix multiplication, well, that's because it is! By changing the processing elements to perform multiply-accumulate we end up with a fully piplelined and parallelly executing matrix multiplier:

<div class="sysarray" id="mm2">
    <center>
        <table></table>
        <button class="next">Next</button>
        <button class="reset">Reset</button>
    </center>
</div>

<br/>

An example with actually multiplies two concrete matrices:

<div class="sysarray" id="mm4">
    <center>
        <table></table>
        <button class="next">Next</button>
        <button class="reset">Reset</button>
    </center>
</div>

<br/>

Lots of other applications exist, such as matrix inversion, correlation, and QR decomposition. We'll implement some of them in this post.

# Generic systolic array
A generic systolic array consists of processing elements consuming and producing from and to all their direct neighbors, chained together to create that large interconnected structure. A single processing element therefore looks like:

<center><img src="/blog/0002-systolic-arrays/LinearPE.svg"></img></center>

Apart from style choices, its type is fairly straightforward in Clash. We simply define it as a function taking four inputs, and producing four outputs. For debugging purposes, each processing element will also receive its index in the systolic array. One might later use this in combination with `trace`. To ease working with this function later, it is defined in its [uncurried form](https://wiki.haskell.org/Currying).

{{< highlight haskell >}}
type ProcessingElement dom m n lr rl tb bt
   = ( (Index m, Index n)
     , Signal dom lr
     , Signal dom rl
     , Signal dom tb
     , Signal dom bt
     )
  -> ( Signal dom lr
     , Signal dom rl
     , Signal dom tb
     , Signal dom bt
     )
{{< / highlight >}}

In order to create a systolic array these processing elements need to be chained together. Let's first focus on creating a single column of processing elements, which -for a column of three elements- looks like:

<center><img style="min-width:25%" src="/blog/0002-systolic-arrays/SysColumn.svg"></img></center>

Any function constructing the array above would need to (internally) construct the colored edges, given the uncolored ones. In code, we'll use the following names for the inputs:

* `tb`: the input at the start of the 'top to bottom' chain. Marked in the diagram as `TB_0`.
* `bt`: the input at the start of the 'bottom to top' chain. Marked in the diagram as `BT_0`
* `lrs`: inputs from left to right. `LR_0`, `LR_1`, ...
* `rls`, `tbs`, `bts`: analogous to `lrs`

And the following names for the outputs:

* `tb'`: the output at the end of the 'top to bottom' chain. Marked in the diagram as `TB_3`.
* `bt'`: the output at the end of the 'bottom to top' chain. Marked in the diagram as `BT_3`
* `lrs'`: outputs from left to right. `LR_0*`, `LR_1*`, ...
* `rls'`, `tbs'`, `bts'`: analogous to `lrs'`

Additionally, we create the indices assuming we've got some `n` in context: `mn = zip indicesI (repeat n)`. Again, this is for debugging purposes only. The trick to creating the columns is to take a leap of faith and just *assume* all variables are well-defined. Then it is simply a matter of mapping over these inputs and applying them to the processing element function. (`pelem'` is a slightly modified version of the user-supplied processing elements, which will be explained later on.)

{{< highlight haskell >}}
  (lrs', rls', tbs', bts') = unzip4 $ map pelem' $ zip5 mn lrs rls tbs bts
{{< / highlight >}}

The diagram indicates that our function already has `rls` and `lrs`, so we don't have to think about those. However, `tbs` and `bts` are missing. Let's focus on `tbs` first, which consists of all top-bottom *inputs* to the processing elements, i.e. `TB_0`, `TB_1`, and `TB_2`. We do have `TB_0` as an input to our function, but the others are still missing. However, we also know `tbs'` consists of all top-bottom *outputs* to the processing elements, i.e. `TB_1`, `TB_2`, and `TB_3`. Thus:

{{< highlight haskell >}}
  tbs = tb :> init tbs'
{{< / highlight >}}

We can define `bts` similarly. We then end up with a single function constructing a single column of the systolic array:

{{< highlight haskell >}}
syscol (n, lrs, rls, tb, bt) = (lrs', rls', tb', bt')
  where
    mn = zip indicesI (repeat n)

    (lrs', rls', tbs', bts') =
      unzip4 $ map pelem' $ zip5 mn lrs rls tbs bts

    tbs = tb :> init tbs'
    bts = tail bts' :< bt
    tb' = last tbs'
    bt' = head bts'
{{< / highlight >}}

To create a the whole array, we apply the same strategy again. Instead of using `pelem'`, we'll use `syscol` and instead of dealing with vectors of signals, we have to deal with vectors of vectors of signals to accomodate all the right-left / left-right connections between each column. 

{{< highlight haskell >}}
systolicArray2D pelem lrs rls tbs bts = (lrs''', rls''', tbs''', bts''')
  where
    -- From `Signal dom (Vec m a)` to `Vec m (Signal dom a)`:
    (lrs', rls', tbs', bts') =
      (unbundle lrs, unbundle rls, unbundle tbs, unbundle bts)

    -- Tie PE columns together:
    (lrss', rlss', tbs'', bts'') =
      unzip4 $ map syscol $ zip5 indicesI lrss rlss tbs' bts'

    lrss  = lrs' :> init lrss'
    rlss  = tail rlss' :< rls'
    lrs'' = last lrss'
    rls'' = head rlss'

    -- From `Vec m (Signal dom a)` to `Signal dom (Vec m a)`:
    (lrs''', rls''', tbs''', bts''') =
      (bundle lrs'', bundle rls'', bundle tbs'', bundle bts'')
{{< / highlight >}}

And that's it for actually tying the processing elements together in a grid. This doesn't quite correspond to the examples shown at the very beginning of this blogpost, the grid is now a continuous circuit. That is, data flows from the sides of the systolic array all through it in a single clock cycle. All outputs need to be delayed a single clock cycle, as such:

<center><img style="min-width:25%" src="/blog/0002-systolic-arrays/LinearPEReg.svg"></img></center>

By simply using [register](http://hackage.haskell.org/package/clash-prelude-0.99.3/docs/Clash-Signal.html#v:register) we can delay its output by one:

{{< highlight haskell >}}
delayPelem pelem lrdflt rldflt tbdflt btdflt input =
  register (lrdflt, rldflt, tbdflt, btdflt) (pelem input)
{{< / highlight >}}

We'd need to make a small adjustment to `systolicArray2D` to take element defaults, but that's it. We've built the systolic array corresponding to the very first interactive example given in this blogpost.

# Delayed systolic array
Lots of applications, including matrix multiplication, have some need to delay their inputs in such a way that the right elements "meet" each other at the same time. Similarly, the outputs need to be delayed strategically such that results belonging to the same entity (for example, a row in a result matrix) arrive synchronously. When data flows left to right and top to bottom, the most natural delay strategy is such that the n<sup>th</sup> element of the n<sup>th</sup> input from the left, arrives at the same time as the n<sup>th</sup> element of the n<sup>th</sup> input from the top. Visually, this equals the second example (repeated here):

<div class="sysarray" id="mm1-rep">
    <center>
        <table></table>
        <button class="next">Next</button>
        <button class="reset">Reset</button>
    </center>
</div>
<br/>

The example suggests the first left-input is delayed by zero, the second by one, etc. For outputs it would make sense to be delayed the other way around. The exact configuration depends on the application and whether paths are used to push results out of the array, or flow data into it. For now, we'll assume the simple case where all inputs are delayed as described, and all outputs are delayed the other way around. 

At the moment, Clash does not have a function to delay a signal a number of times, so we need to build one ourselves. A quick solution is to fold over a vector of units, while adding a register at each step. Clash will filter empty types, so this actually won't interfere with our HDL output at all.

{{< highlight haskell >}}
-- | Put /n/ registers after given signal.
delayN
  :: forall n dom gated synchronous a
   . HiddenClockReset dom gated synchronous
  => a
  -- ^ Default value register
  -> SNat n
  -- ^ Number of registers to insert
  -> Signal dom a
  -- ^ Signal to delay
  -> Signal dom a
  -- ^ Delayed signal
delayN dflt n@SNat signal =
  foldl (\s _ -> register dflt s) signal (replicate n ())
{{< / highlight >}}

Delaying the signals is fairly easy with [smap](https://hackage.haskell.org/package/clash-prelude-0.99/docs/Clash-Sized-Vector.html#v:smap). Most of the code is related to packing/unpacking signals so they can be mapped over ("type torturing" 🙂):

{{< highlight haskell >}}
systolicArray2Dd lr rl tb bt pelem lrs rls tbs bts = (lrs''', rls''', tbs''', bts''')
  where
    -- Append delays to array inputs
    lrs' = bundle (smap (delayN lr) (unbundle lrs))
    rls' = bundle (smap (delayN rl) (unbundle rls))
    tbs' = bundle (smap (delayN tb) (unbundle tbs))
    bts' = bundle (smap (delayN bt) (unbundle bts))

    -- Create systolic array without delays of outputs
    (lrs'', rls'', tbs'', bts'') =
      systolicArray2D lr rl tb bt pelem lrs' rls' tbs' bts'

    -- Append delays to array outputs
    lrs''' = bundle $ reverse $ smap (delayN lr) (reverse $ unbundle lrs'')
    rls''' = bundle $ reverse $ smap (delayN rl) (reverse $ unbundle rls'')
    tbs''' = bundle $ reverse $ smap (delayN tb) (reverse $ unbundle tbs'')
    bts''' = bundle $ reverse $ smap (delayN bt) (reverse $ unbundle bts'')
{{< / highlight >}}

And that's all there is to it. 



# Matrix multiplication
So far we've built a generic systolic array and a delayed one on top of it. We haven't built anything useful yet though, which is what this section is for. We've selected a few amongst the most commonly used. Even with a rigid structures such systolic arrays, many design choices still exist. The implemented algorithms are therefore by no means meant as perfect solutions. This subsection will deal with matrix multiplication.

<center><img style="min-width:40%" src="/blog/0002-systolic-arrays/MM.svg"></img></center>

To test and communicate various communication strategies, we'll use spacetime diagrams. On the vertical axis there's space: the processing elements. On the horizontal axis there's time. We'll only consider the case where processing elements can communicate in one dimension: either left-right or top-bottom. If they communicate left-right, the processing elements represent a row in the systolic array, if they communicate top-down, the processing elements represent a column in the systolic array. It actually doesn't really matter, so to ease talking about this problem let's assume the communicate top-bottom. A <span style="background-color:#66CC00; color:white;">green</span> background represents every moment in time a specific element produces useful data:

<table cellspacing="0" border="0">
	<colgroup width="34"></colgroup>
	<colgroup width="23" span="10"></colgroup>
	<colgroup width="26" span="5"></colgroup>
	<tbody><tr>
		<td height="21" align="left"><b>c \ t</b></td>
		<td sdval="0" sdnum="1043;" align="center"><b>0</b></td>
		<td sdval="1" sdnum="1043;" align="center"><b>1</b></td>
		<td sdval="2" sdnum="1043;" align="center"><b>2</b></td>
		<td sdval="3" sdnum="1043;" align="center"><b>3</b></td>
		<td sdval="4" sdnum="1043;" align="center"><b>4</b></td>
		<td sdval="5" sdnum="1043;" align="center"><b>5</b></td>
		<td sdval="6" sdnum="1043;" align="center"><b>6</b></td>
		<td sdval="7" sdnum="1043;" align="center"><b>7</b></td>
		<td sdval="8" sdnum="1043;" align="center"><b>8</b></td>
		<td sdval="9" sdnum="1043;" align="center"><b>9</b></td>
		<td sdval="10" sdnum="1043;" align="center"><b>10</b></td>
		<td sdval="11" sdnum="1043;" align="center"><b>11</b></td>
		<td sdval="12" sdnum="1043;" align="center"><b>12</b></td>
		<td sdval="13" sdnum="1043;" align="center"><b>13</b></td>
		<td sdval="14" sdnum="1043;" align="center"><b>14</b></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe1</b></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe2</b></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe3</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe4</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center"><br></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe5</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
	</tr>
</tbody></table>

Empty cells will be used to indicate where some piece of data resides. We'll see an example in the next section.

## General matrix multiplication
Matrix multiplication can implemented by having each processing element multiply both its input signals, accumulating, and pushing its data out periodically as shown in the first part of this blogpost. The period at which processing elements need to push out data depends on `n`, the number of columns in the left matrix and the number of rows in the right. Visually:

<center><img style="min-width:70%" src="/blog/0002-systolic-arrays/Dimensions.svg"></img></center>

Assuming that each cell communicates its result downwards and each cell can only push a single element, we need a number of flush rounds if `m` exceeds `n`. After all, the bandwidth of the outer processing element to its environment is a single element per cycle. Thus, more than one result per cycle per column exceeds that bandwidth. If `n` exceeds `m` no flush rounds are needed, but the systolic array produces "garbage" values some of the time as the bandwidth exceeds the result production. 

For now, let's assume `n = m`. Communication downwards effectively binds the systolic array to communicate as follows:

<table cellspacing="0" border="0">
	<colgroup width="33" span="16"></colgroup>
	<tbody><tr>
		<td height="20" align="left"><b>c \ t</b></td>
		<td sdval="0" sdnum="1043;" align="center"><b>0</b></td>
		<td sdval="1" sdnum="1043;" align="center"><b>1</b></td>
		<td sdval="2" sdnum="1043;" align="center"><b>2</b></td>
		<td sdval="3" sdnum="1043;" align="center"><b>3</b></td>
		<td sdval="4" sdnum="1043;" align="center"><b>4</b></td>
		<td sdval="5" sdnum="1043;" align="center"><b>5</b></td>
		<td sdval="6" sdnum="1043;" align="center"><b>6</b></td>
		<td sdval="7" sdnum="1043;" align="center"><b>7</b></td>
		<td sdval="8" sdnum="1043;" align="center"><b>8</b></td>
		<td sdval="9" sdnum="1043;" align="center"><b>9</b></td>
		<td sdval="10" sdnum="1043;" align="center"><b>10</b></td>
		<td sdval="11" sdnum="1043;" align="center"><b>11</b></td>
		<td sdval="12" sdnum="1043;" align="center"><b>12</b></td>
		<td sdval="13" sdnum="1043;" align="center"><b>13</b></td>
		<td sdval="14" sdnum="1043;" align="center"><b>14</b></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe1</b></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center"><font color="#CCCCCC">r1</font></td>
		<td align="center">r1</td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe2</b></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center"><font color="#CCCCCC">r2</font></td>
		<td align="center"><font color="#CCCCCC">r2</font></td>
		<td align="center">r2</td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center"><font color="#CCCCCC">r2</font></td>
		<td align="center"><font color="#CCCCCC">r2</font></td>
		<td align="center">r2</td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center"><font color="#CCCCCC">r2</font></td>
		<td align="center"><font color="#CCCCCC">r2</font></td>
		<td align="center">r2</td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe3</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><font color="#CCCCCC">r3</font></td>
		<td align="center">r3</td>
		<td align="center">r2</td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><font color="#CCCCCC">r3</font></td>
		<td align="center">r3</td>
		<td align="center">r2</td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><font color="#CCCCCC">r3</font></td>
		<td align="center">r3</td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe4</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center">r4</td>
		<td align="center">r3</td>
		<td align="center">r2</td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center">r4</td>
		<td align="center">r3</td>
		<td align="center">r2</td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center">r4</td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe5</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
		<td align="center">r4</td>
		<td align="center">r3</td>
		<td align="center">r2</td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
		<td align="center">r4</td>
		<td align="center">r3</td>
		<td align="center">r2</td>
		<td align="center">r1</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
	</tr>
</tbody></table>

Elements need to store their results for a while, before synchronously passing them to their neighbors. In the diagram, the moments where each element pushes its own result is `t=4`, `t=9`, and `t=14`. At all other cycles, processing elements simply pass the results they receive from their upper neighbors down. Because processing elements need to synchronously push their data, this either requires:

1. a signal from the left telling if `t=5n - 1`; or
2. each element keeping a local counter counting the global time; or
3. each element keeping a local counter counting last time since fire; or

The last two waste hardware. The first one requires a split between timing strategies for data and control. That does requires a combination of `systolicArray2D` and `systolicArray2Dd`, but that's easy enough - though tedious - to implement:

{{< highlight haskell >}}
systolicArray2Dud
  lr@(lru, lrd)
  rl@(rlu, rld)
  tb@(tbu, tbd)
  bt@(btu, btd)
  pelem
  lrus lrds
  rlus rlds
  tbus tbds
  btus btds =
  (lrus', lrds''', rlus', rlds''', tbus', tbds''', btus', btds''')
  where
    -- Append delays to array inputs
    lrds' = bundle (smap (delayN lrd) (unbundle lrds))
    rlds' = bundle (smap (delayN rld) (unbundle rlds))
    tbds' = bundle (smap (delayN tbd) (unbundle tbds))
    btds' = bundle (smap (delayN btd) (unbundle btds))

    -- Bundle delayed and undelayed signals
    lrs = zipWith (,) <$> lrus <*> lrds'
    rls = zipWith (,) <$> rlus <*> rlds'
    tbs = zipWith (,) <$> tbus <*> tbds'
    bts = zipWith (,) <$> btus <*> btds'

    -- Create systolic array without delays of outputs
    (lrs', rls', tbs', bts') =
      systolicArray2D lr rl tb bt pelem lrs rls tbs bts

    -- Unbundle delayed and undelayed signals
    (lrus', lrds'') = (fmap fst <$> lrs', fmap snd <$> lrs')
    (rlus', rlds'') = (fmap fst <$> rls', fmap snd <$> rls')
    (tbus', tbds'') = (fmap fst <$> tbs', fmap snd <$> tbs')
    (btus', btds'') = (fmap fst <$> bts', fmap snd <$> bts')

    -- Append delays to array outputs
    lrds''' = bundle $ reverse $ smap (delayN lrd) (reverse $ unbundle lrds'')
    rlds''' = bundle $ reverse $ smap (delayN rld) (reverse $ unbundle rlds'')
    tbds''' = bundle $ reverse $ smap (delayN tbd) (reverse $ unbundle tbds'')
    btds''' = bundle $ reverse $ smap (delayN btd) (reverse $ unbundle btds'')
{{< / highlight >}}

Note that the only difference between `systolicArray2Dud` and `systolicArray2Dd` is that we pass in tuples of which the first element is passed undelayed to the matrix, while the second is delayed according to earlier discussed strategies. Each processing element needs to support the actions discussed just now:

{{< highlight haskell >}}
data SyncInstrPEDown
  = Pass
  -- ^ Take data from upper neighbor, pass to lower neighbor
  | Inject
  -- ^ Discard data from upper neighbor, pass own storage to lower neighbor

data AsyncInstrPEDown
  = Accum
  -- ^ Accumulate products of incoming signals
  | Store
  -- ^ Move current result to storage
{{< / highlight >}}

The process element then keeps two buffers: one to store an accumulation (`s1`), and one to store a result (`s2`). Processing elements are exactly the same everywhere and simply listen for incoming instructions as defined earlier.

{{< highlight haskell >}}
pelemDown ((m, n), lrs, rls, tbs, bts) = (lrs, rls, tbs'', bts)
  where
    tbs'  = mealy pelem' (0, 0) $ bundle (lrs, snd <$> tbs)
    tbs'' = liftA2 (,) (fst <$> tbs) tbs'

    pelem' (s1, s2)  ((Pass,   (a, Accum)), (b, res))  = ((s1+a*b, s2    ),  (b, res))
    pelem' (s1, _s2) ((Pass,   (a, Store)), (b, res))  = ((0,      s1+a*b),  (b, res))
    pelem' (s1, s2)  ((Inject, (a, Accum)), (b, _res)) = ((s1+a*b, s2    ),  (b, s2))
    pelem' (s1, s2)  ((Inject, (a, Store)), (b, res))  = ((0,      s1+a*b),  (b, s2))
{{< / highlight >}}

A wrapping function ties the systolic array and processing element functions.

{{< highlight haskell >}}
generalMatrixMultiplicationDown
  :: forall a n m p dom gated synchronous
   . HiddenClockReset dom gated synchronous
  => Num a
  => Show a
  => NFData a
  => KnownNat m
  => KnownNat p
  => SNat (n + 1)
  -- ^ Number of columns / rows of left matrix / right matrix
  -> Signal dom (Vec (m + 1) a)
  -- ^ Columns of left matrix
  -> Signal dom (Vec (p + 1) a)
  -- ^ Rows of right matrix
  -> Signal dom (Vec (p + 1) a)
  -- ^ Rows of result matrix, in reverse order
generalMatrixMultiplicationDown n@SNat cols rows = fmap snd <$> tbs'
  where
    -- Determine inputs for systolic array:
    counter :: Signal dom (Index (n + 1))
    counter = register minBound (satPlus SatWrap 1 <$> counter)

    sysCmd n
      | n == maxBound = (repeat Inject, repeat Store)
      | otherwise     = (repeat Pass,   repeat Accum)

    (lrus, dcmds) = unbundle (sysCmd <$> counter)

    -- Pass columns and delayed commands from the left, and the rows and dummy
    -- passthrough values from the top:
    lrds = zip <$> cols <*> dcmds
    tbds = zip <$> rows <*> pure (repeat 0)

    -- nothingP and nothingM differ in vector length, thus having different
    -- types, explaining the seemingly duplicate definitions:
    nothingP = pure $ repeat ()
    nothingM = pure $ repeat ()

    -- Create actual array:
    (_, _, _, _, _, tbs', _, _) =
      systolicArray2Dud
        -- Defaults for registers
        (Inject, (0, Store)) ((), ()) ((), (0, 0)) ((), ())
        -- Processing element
        pelemDown
        -- Inputs:
        lrus     lrds
        nothingM nothingM
        nothingP tbds
        nothingP nothingP
{{< / highlight >}}

Bandwidth requirements per node, where `|a|` is the number of bits needed to store numeric type `a`:

* Top-to-bottom: 2 &middot; `|a|`
* Left-to-right: `|a|` + 1 + 1

Registers needed systolic array:

* Top: 0.5 &middot; (p<sup>2</sup> - p) &middot; `|a|`
* Left: 0.5 &middot; (m<sup>2</sup> - m) &middot; (`|a|` + 1)
* Bottom: 0.5 &middot; (p<sup>2</sup> - p) &middot; `|a|`
* Nodes: 2pm &middot; `|a|`
* Edges: 3pm &middot; (`|a|` + 1 + 1)

The total latency from inputting the last row/column to receiving the last result row is `m`, the number of rows in the left matrix.

## `m` equals `n`
The general matrix multiplication algorithm needs an extra register in each processing elements to temporarily store the results generated by each element, in order to transmit it to their bottom neighbors later. A simply alternative strategy would be to introduce flush rounds, where PEs end up being utilized 50% of the time - trading utilization for bandwidth.

Square matrices turn out to have an interesting property which allows them to be calculated and read without introducing additional registers. By utilizing the bottom-to-top communication channel of our systolic array, processing elements can pass their results right after producing a meaningful result. The spacetime diagram then looks like:

<table cellspacing="0" border="0">
	<colgroup width="34"></colgroup>
	<colgroup width="23" span="10"></colgroup>
	<colgroup width="26" span="5"></colgroup>
	<tbody><tr>
		<td height="21" align="left"><b>c \ t</b></td>
		<td sdval="0" sdnum="1043;" align="center"><b>0</b></td>
		<td sdval="1" sdnum="1043;" align="center"><b>1</b></td>
		<td sdval="2" sdnum="1043;" align="center"><b>2</b></td>
		<td sdval="3" sdnum="1043;" align="center"><b>3</b></td>
		<td sdval="4" sdnum="1043;" align="center"><b>4</b></td>
		<td sdval="5" sdnum="1043;" align="center"><b>5</b></td>
		<td sdval="6" sdnum="1043;" align="center"><b>6</b></td>
		<td sdval="7" sdnum="1043;" align="center"><b>7</b></td>
		<td sdval="8" sdnum="1043;" align="center"><b>8</b></td>
		<td sdval="9" sdnum="1043;" align="center"><b>9</b></td>
		<td sdval="10" sdnum="1043;" align="center"><b>10</b></td>
		<td sdval="11" sdnum="1043;" align="center"><b>11</b></td>
		<td sdval="12" sdnum="1043;" align="center"><b>12</b></td>
		<td sdval="13" sdnum="1043;" align="center"><b>13</b></td>
		<td sdval="14" sdnum="1043;" align="center"><b>14</b></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe1</b></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center"><br></td>
		<td align="center">r2</td>
		<td align="center"><br></td>
		<td align="center">r3</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center">r4</td>
		<td align="center">r2</td>
		<td align="center">r5</td>
		<td align="center">r3</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r1</font></td>
		<td align="center">r4</td>
		<td align="center">r2</td>
		<td align="center">r5</td>
		<td align="center">r3</td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe2</b></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center"><br></td>
		<td align="center">r3</td>
		<td align="center"><br></td>
		<td align="center">r4</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center">r5</td>
		<td align="center">r3</td>
		<td align="center"><br></td>
		<td align="center">r4</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r2</font></td>
		<td align="center">r5</td>
		<td align="center">r3</td>
		<td align="center"><br></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe3</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><br></td>
		<td align="center">r4</td>
		<td align="center"><br></td>
		<td align="center">r5</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><br></td>
		<td align="center">r4</td>
		<td align="center"><br></td>
		<td align="center">r5</td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r3</font></td>
		<td align="center"><br></td>
		<td align="center">r4</td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe4</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center"><br></td>
		<td align="center">r5</td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center"><br></td>
		<td align="center">r5</td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r4</font></td>
		<td align="center"><br></td>
	</tr>
	<tr>
		<td height="17" align="left"><b>pe5</b></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td align="center"><br></td>
		<td bgcolor="#66CC00" align="center"><font color="#EEEEEE">r5</font></td>
	</tr>
</tbody></table>

Compared to the previous communication strategy, no additional buffers are needed - but latency is up. Also note that this is only possible with square matrices where each side is are of odd size. Even matrices would need a single flush cycle. Due to the missing register, the implementation is much less complex. First, the processing elements only need to support two commands:

{{< highlight haskell >}}
data InstrPEUp
  = Clear
  -- ^ Clear state, push old state plus product of incoming to upper neighbor
  | Data
  -- ^ Move data from neighbor below to upper neighbor
    deriving (Generic, Show, Eq, NFData)
{{< / highlight >}}

The implementation of the processing elements now looks like:

{{< highlight haskell >}}
pelemUp ((m, n), lrs, rls, tbs, bts) = (lrs, rls, tbs, bts')
  where
    bts' = mealy pelem' 0 $ bundle (lrs, tbs, bts)
    pelem' c ((Clear, a), b, _bt) = (0,       a*b + c)
    pelem' c ((Data, a),  b, bt)  = (a*b + c, bt)
{{< / highlight >}}

Similar to our previous strategy, we need a simple component generating the commands passed to the processing elements. In contrast to our previous approach, we don't need to split delayed and undelayed inputs to the systolic array. The command generation component simply looks like:

{{< highlight haskell >}}
sysInput n col row
  | n == maxBound = (zip (repeat Clear) col, row)
  | otherwise     = (zip (repeat Data)  col, row)
{{< / highlight >}}

which already concludes the implementation of this matrix multiplication algorithm.

Bandwidth requirements per node, where `|a|` is the number of bits needed to store numeric type `a`:

* Top-to-bottom: `|a|`
* Bottom-to-top: `|a|`
* Left-to-right: `|a|` + 1

Registers needed systolic array:

* Top: (p<sup>2</sup> - p) &middot; `|a|`
* Left: 0.5 &middot; (m<sup>2</sup> - m) &middot; (`|a|` + 1)
* Nodes: pm &middot; `|a|`
* Edges: 3pm &middot; (`|a|` + 1)

The total latency from inputting the last row/column to receiving the last result row is `2m`, the number of rows in the left matrix.

## A quick note on pipelined processing elements
One of the advantages of using a systolic array like this one is that integrating pipelined elements is easy. As long as all outputs are delayed by the same number of registers, the array will have the same behavior bar its increased latency. Clash offers some tools to make it easier to type these pipelined signals in the form of [delayed signals](https://hackage.haskell.org/package/clash-prelude-0.99/docs/Clash-Signal-Delayed.html). In fact, [a previous blogpost on matrix multiplication](/blog/0001-matrix-multiplication/) used this to guarantee some timing aspects of its pipelined functions.

# Triangular systolic arrays
Systolic arrays are not by definition of rectangular shape. For example, [Gentleman and Kung](http://www.csd.uwo.ca/~moreno/CS433-CS9624/Resources/Matrix_Triangularization_by_systetolic_arrays.pdf) describe a systolic array with a triangular shape for many different algorithms. Due to a varying number of processing elements on each "row" of the systolic array, we cannot use the same tactic for building systolic arrays as before. Let's first look at a visualized systolic array as described by Gentleman and Kung:

<center><img src="halfSysArray.svg"></img></center>

Let's define the types of the different wires as follows:

* `↓ :: tb` (<u>t</u>op-<u>b</u>ottom)
* `↘ :: dg` (<u>d</u>ia<u>g</u>onal)
* `→ :: lr` (<u>l</u>eft-<u>r</u>ight)

Now, with these types, we can imagine that:

* `○ :: Signal dom tb -> Signal dom dg -> (Signal dom dg, Signal dom lr)`
* `◻ :: Signal dom tb -> Signal dom lr -> (Signal dom tb, Signal dom lr)`

whereas our complete systolic array `triangularSystolicArray` is of type:

{{< highlight haskell >}}
triangularSystolicArray
  :: (Signal dom tb -> Signal dom dg -> (Signal dom dg, Signal dom lr))
  -- ^ Function for ○
  -> (Signal dom tb -> Signal dom lr -> (Signal dom tb, Signal dom lr))
  -- ^ Function for ◻
  -> (tb, dg, lr)
  -- ^ Register defaults
  -> Signal dom (Vec n tb)
  -- ^ Input from top
  -> Signal dom dg
  -- ^ Input for first ○
  -> (Signal dom dg, Signal dom (Vec n lr))
  -- ^ (right outputs, diagonal output of last ○)
{{< / highlight >}}

Building this systolic array is a bit more complex due to its non-square shape. We can't simply hold every signal corresponding to a wire between processing elements in a vector, as the first row has *n* wires, the next *n-1*, etc. Similarly, the first column has *1* wire from left to right, while the second has *2*, etc. Luckily, we're only actually interested in the most right wires and we can safely discard the rest. If we can write a function creating a single column given the results of the previous column, [dfold](https://hackage.haskell.org/package/clash-prelude-0.99.2/docs/Clash-Sized-Vector.html#v:dfold) promises to build the whole thing.

Building a single column can be achieved with `mapAccumL`, accumulating the top-bottom output, while producing a left-right output for every inner processing element (◻). The top-bottom output is combined with the diagonal input from the previous column and an edge processing element (○). Thus:

{{< highlight haskell >}}
triangularColumn edgeF innerF top diagonal lefts = (diagonal', bundle $ rights :< right)
  where
    -- Apply inner functions
    (bottom, rights) = mapAccumL innerF top (unbundle lefts)

    -- Terminate with edge function
    (diagonal', right) = edgeF bottom diagonal
{{< / highlight >}}

The definition must be slightly altered to include registers present at the output of every processing element. If omitted, all data would flow through the systolic array in a single cycle which is undesirable. This will finally get us:

{{< highlight haskell >}}
triangularColumn edgeF innerF (tb, dg, lr) top diagonal (unbundle -> lefts) =
  (diagonal', bundle $ rights :< right)
    where
      -- Simple helper function to delay tuples
      bidelay (adflt, bdflt) (a, b) = (register adflt a, register bdflt b)

      -- Apply inner functions
      (bottom, rights) = mapAccumL innerF' top lefts
      innerF' top left = bidelay (tb, lr) (innerF top left)

      -- Terminate with edge function
      (diagonal', right) = bidelay (dg, lr) (edgeF bottom diagonal)
{{< / highlight >}}

In order for `dfold` to work, it asks its users to define a type-level function yielding the type at iteration *n*. If the type works out after each iteration, the whole construct typechecks. We first need to define a data type used in to instantiate type level function application:

{{< highlight haskell >}}
-- | Collection of types that don't change between fold-iterations, which
-- we need to construct the type at some iteration.
data
  TriangularMotive
    (dg :: *)
    (lr :: *)
    (dom :: Domain)
    (f :: TyFun Nat *) :: *
{{< / highlight >}}

Then, we provide an instance for [Apply](https://hackage.haskell.org/package/singletons-2.2/docs/Data-Singletons.html#t:Apply), the class used to implement type level functions. The actual type we end up with at the *n<sup>th</sup>* iteration is simple enough: a tuple of the diagonal input and a vector of left-right outputs from our inner (◻) and edge (○) functions.

{{< highlight haskell >}}
-- | Intermediate type at fold-iteration /n/:
type instance Apply (TriangularMotive dg lr dom) n =
  (Signal dom dg, Signal dom (Vec n r))
{{< / highlight >}}

We can now glue the whole array together. The actual code is mostly taken by type signatures needed to not disambiguate certain types:

{{< highlight haskell >}}
triangularSystolicArray
  :: forall n tb dg lr dom gated synchronous
   . HiddenClockReset dom gated synchronous
  => KnownNat n
  => (Signal dom tb -> Signal dom dg -> (Signal dom dg, Signal dom lr))
  -- ^ Function for ○
  -> (Signal dom tb -> Signal dom lr -> (Signal dom tb, Signal dom lr))
  -- ^ Function for ◻
  -> (tb, dg, lr)
  -- ^ Register defaults
  -> Signal dom (Vec n tb)
  -- ^ Input from top
  -> Signal dom dg
  -- ^ Input for first ○
  -> (Signal dom dg, Signal dom (Vec n lr))
  -- ^ (right outputs, diagonal output of last ○)
triangularSystolicArray edgeF innerF dflts@(tbdflt, _, _) tops diagonal =
  (diagonal', rights)
    where
      -- Add delays to top inputs, as described in paper
      tops' = smap (delayN tbdflt) (unbundle tops)

      -- Fold over top inputs, progressively expanding the triangular array
      (diagonal', rights) =
        dfold
          (Proxy @ (TriangularMotive dg lr dom))
          go
          (diagonal, pure Nil)
          tops'

      -- Simple wrapping function around 'triangularColumn'. Explicit types are
      -- needed to not confuse the type checker.
      go
        :: forall l
         . SNat l
        -> Signal dom tb
        -> (Signal dom dg, Signal dom (Vec l lr))
        -> (Signal dom dg, Signal dom (Vec (l + 1) lr))
      go l@SNat tb (dg, lrs) =
        triangularColumn edgeF innerF dflts tb dg lrs
{{< / highlight >}}

Note that we haven't implemented any functionality yet, just like our previous general systolic array functions. It's now easy to build one though, as we only have to pass in the two functions, ◻ and ○.

# Matrix triangularization
Gentleman and Kung describe two algorithms expressed in terms of the triangular systolic array. We're going to implement the first one; *triangularization with neighbor pivoting* (page 3). The paper lays out the inner ("internal") and boundary ("edge") functions with pseudocode which we can more or less copy. The pseudo code is defined as a simple mealy machine, so that's how we'll do it as well. `safeQuot` is a function which returns zero if the denominator is zero, as also described in the paper:

{{< highlight haskell >}}
-- "Internal cell" as mealy machine
innerF x (x', (m', True))  = (x', x  + (m' * x'))
innerF x (x', (m', False)) = (x,  x' + (m' * x ))

-- "Boundary cell" as mealy machine
edgeF x x'
  | abs x' >= abs x = (x', (safeQuot x x',          True))
  | otherwise       = (x,  (negate $ safeQuot x' x, False))
{{< / highlight >}}

The systolic array is then created by slightly modifying these functions to signal notation, and calling `triangularSystolicArray`. Note that the algorithm doesn't actually use the diagonal communication lines. Just like when we didn't use all communication channels in the square systolic array, we'll simply pass Haskell's "empty" type: unit (`()`). Empty types will be filtered by Clash.

{{< highlight haskell >}}
neighborPivotTriangularization
  :: forall n a dom gated synchronous
   . HiddenClockReset dom gated synchronous
  => Integral a
  => KnownNat n
  => Signal dom (Vec n a)
  -- ^ Rows of matrix
  -> Signal dom (Vec n a)
neighborPivotTriangularization rows = fmap fst <$> lrs
  where
    -- Instantiate systolic array
    (_, lrs) =
      triangularSystolicArray
        -- Processing elements:
        edgeF' innerF'
        -- Defaults for registers:
        (0, (), (0, False))
        -- Top-bottom input:
        rows
        -- Diagonal input:
        (pure ())

    -- Turn mealy machines into signal constructs
    edgeF' tb dg  = (dg, mealy edgeF 0 tb)
    innerF' tb lr = (mealy innerF 0 (liftA2 (,) tb lr), lr)
{{< / highlight >}}

The paper does not tell us how to retrieve the results from the array. Like any systolic array, we could introduce flush rounds or increase bandwidth to move the results to the array borders. We've already seen this process for matrix multiplication, so we'll skip it for this one.

# Conclusion
We've built two types of systolic arrays in Clash, both solving real-world problems. Although the design methodology in Clash is somewhat different than other (traditional) tooling, it hopefully gave a feeling on how to build generalized solutions in Clash, while retaining readability. Any thoughts or questions can be left in the comments. See you in a next blogpost!


<!-- Javascript and CSS --> 
<script src="script.js"></script>

<style>
.sysarray{
  min-height:568px;
}

.sysarray table, 
.sysarray td{
  border:none;
  text-align:center;
}

.sysarray td{ 
  height:65px;
  width:65px;
}

.sysarray .a{
  background-color: #FFF2CC;
}

.sysarray .b{
  background-color: #DAE8FC;
}

.systolic table{ 
  height:auto;
  width:auto;
  font-family: monospace;
}

.sysarray td.pe{
  border: 1px solid black;
}

#mm0 table{
  width:80%;
}

#mm0 {
  min-height:439px;
}

#mm1 td.pe,
#mm2 td.pe,
#mm3 td.pe,
#mm4 td.pe{
  font-size:0.8em;

}

</style>
