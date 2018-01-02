# Common
[![Build Status](https://travis-ci.org/prometheus/common.svg)](https://travis-ci.org/prometheus/common)

This repository contains Go libraries that are shared across Prometheus
components and libraries.

* **config**: Common configuration structures
* **expfmt**: Decoding and encoding for the exposition format
* **log**: A logging wrapper around [logrus](https://github.com/Sirupsen/logrus)
* **model**: Shared data structures
* **route**: A routing wrapper around [httprouter](https://github.com/julienschmidt/httprouter) using `context.Context`
* **version**: Version informations and metric
