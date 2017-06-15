hdrhistogram
============

[![Build Status](https://travis-ci.org/codahale/hdrhistogram.png?branch=master)](https://travis-ci.org/codahale/hdrhistogram)

A pure Go implementation of the [HDR Histogram](https://github.com/HdrHistogram/HdrHistogram).

> A Histogram that supports recording and analyzing sampled data value counts
> across a configurable integer value range with configurable value precision
> within the range. Value precision is expressed as the number of significant
> digits in the value recording, and provides control over value quantization
> behavior across the value range and the subsequent value resolution at any
> given level.

For documentation, check [godoc](http://godoc.org/github.com/codahale/hdrhistogram).
