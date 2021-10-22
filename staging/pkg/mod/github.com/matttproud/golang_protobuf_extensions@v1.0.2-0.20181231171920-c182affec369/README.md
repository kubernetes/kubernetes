# Overview
This repository provides various Protocol Buffer extensions for the Go
language (golang), namely support for record length-delimited message
streaming.

| Java                           | Go                    |
| ------------------------------ | --------------------- |
| MessageLite#parseDelimitedFrom | pbutil.ReadDelimited  |
| MessageLite#writeDelimitedTo   | pbutil.WriteDelimited |

Because [Code Review 9102043](https://codereview.appspot.com/9102043/) is
destined to never be merged into mainline (i.e., never be promoted to formal
[goprotobuf features](https://github.com/golang/protobuf)), this repository
will live here in the wild.

# Documentation
We have [generated Go Doc documentation](http://godoc.org/github.com/matttproud/golang_protobuf_extensions/pbutil) here.

# Testing
[![Build Status](https://travis-ci.org/matttproud/golang_protobuf_extensions.png?branch=master)](https://travis-ci.org/matttproud/golang_protobuf_extensions)
