# Minimal Go logging using klog

This package implements the [logr interface](https://github.com/go-logr/logr)
in terms of Kubernetes' [klog](https://github.com/kubernetes/klog).  This
provides a relatively minimalist API to logging in Go, backed by a well-proven
implementation.

This is a BETA grade implementation.
