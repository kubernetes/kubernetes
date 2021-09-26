# Common
[![Build Status](https://travis-ci.org/prometheus/common.svg)](https://travis-ci.org/prometheus/common)

This repository contains Go libraries that are shared across Prometheus
components and libraries. They are considered internal to Prometheus, without
any stability guarantees for external usage.

* **config**: Common configuration structures
* **expfmt**: Decoding and encoding for the exposition format
* **model**: Shared data structures
* **promlog**: A logging wrapper around [go-kit/log](https://github.com/go-kit/kit/tree/master/log)
* **route**: A routing wrapper around [httprouter](https://github.com/julienschmidt/httprouter) using `context.Context`
* **server**: Common servers
* **version**: Version information and metrics

## Deprecated
* **log**: A logging wrapper around [logrus](https://github.com/sirupsen/logrus)
