# Example

This directory includes example logger setup allowing users to easily check and test impact of logging configuration.

Below we can see examples of how some features work.

## Default

Run:
```console
go run .
```
Metadata:
```console
ignore_klog_header
```

Expected output:
```console
I1227 11:41:04.897402   67295 logger.go:45] "Oops, I shouldn't be logging yet!"
This is normal output via stdout.
This is other output via stderr.
I1227 11:41:04.897711   67295 logger.go:77] Log using Infof, key: value
I1227 11:41:04.897718   67295 logger.go:78] "Log using InfoS" key="value"
E1227 11:41:04.897722   67295 logger.go:80] Log using Errorf, err: fail
E1227 11:41:04.897727   67295 logger.go:81] "Log using ErrorS" err="fail"
I1227 11:41:04.897731   67295 logger.go:83] Log with sensitive key, data: {"secret"}
I1227 11:41:04.897741   67295 logger.go:88] "Now the default logger is set, but using the one from the context is still better."
I1227 11:41:04.897749   67295 logger.go:91] "Log sensitive data through context" data={Key:secret}
I1227 11:41:04.897755   67295 logger.go:95] "runtime" duration="1m0s"
I1227 11:41:04.897760   67295 logger.go:96] "another runtime" duration="1h0m0s" duration="1m0s"
```
##Simple example

Run:
```console
echo hello
```

Expected output:
```console
hello
```
