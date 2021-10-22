# Minimal Go logging using klog

This package implements the [logr interface](https://github.com/go-logr/logr)
in terms of Kubernetes' [klog](https://github.com/kubernetes/klog).  This
provides a relatively minimalist API to logging in Go, backed by a well-proven
implementation.

Because klogr was implemented before klog itself added supported for
structured logging, the default in klogr is to serialize key/value
pairs with JSON and log the result as text messages via klog. This
does not work well when klog itself forwards output to a structured
logger.

Therefore the recommended approach is to let klogr pass all log
messages through to klog and deal with structured logging there. Just
beware that the output of klog without a structured logger is meant to
be human-readable, in contrast to the JSON-based traditional format.

This is a BETA grade implementation.
