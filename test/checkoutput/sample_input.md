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
```
I0605 22:03:07.224293 3228948 logger.go:58] Log using Infof, key: value
I0605 22:03:07.224378 3228948 logger.go:59] "Log using InfoS" key="value"
E0605 22:03:07.224393 3228948 logger.go:61] Log using Errorf, err: fail
E0605 22:03:07.224402 3228948 logger.go:62] "Log using ErrorS" err="fail"
I0605 22:03:07.224407 3228948 logger.go:64] Log message has been redacted. Log argument #0 contains: [secret-key]
```
##Simple example

Run:
```console
echo hello
```

Expected output:
```
hello
```
