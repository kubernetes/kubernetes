# Reporting bugs

If any part of the etcd project has bugs or documentation mistakes, please let us know by [opening an issue][etcd-issue]. We treat bugs and mistakes very seriously and believe no issue is too small. Before creating a bug report, please check that an issue reporting the same problem does not already exist.

To make the bug report accurate and easy to understand, please try to create bug reports that are:

- Specific. Include as much details as possible: which version, what environment, what configuration, etc. If the bug is related to running the etcd server, please attach the etcd log (the starting log with etcd configuration is especially important).

- Reproducible. Include the steps to reproduce the problem. We understand some issues might be hard to reproduce, please includes the steps that might lead to the problem. If possible, please attach the affected etcd data dir and stack strace to the bug report.

- Isolated. Please try to isolate and reproduce the bug with minimum dependencies. It would significantly slow down the speed to fix a bug if too many dependencies are involved in a bug report. Debugging external systems that rely on etcd is out of scope, but we are happy to provide guidance in the right direction or help with using etcd itself.

- Unique. Do not duplicate existing bug report.

- Scoped. One bug per report. Do not follow up with another bug inside one report.

It may be worthwhile to read [Elika Etemad’s article on filing good bug reports][filing-good-bugs] before creating a bug report.

We might ask for further information to locate a bug. A duplicated bug report will be closed.

## Frequently asked questions

### How to get a stack trace

``` bash
$ kill -QUIT $PID
```

### How to get etcd version

``` bash
$ etcd --version
```

### How to get etcd configuration and log when it runs as systemd service ‘etcd2.service’

``` bash
$ sudo systemctl cat etcd2
$ sudo journalctl -u etcd2
```

Due to an upstream systemd bug, journald may miss the last few log lines when its processes exit. If journalctl says etcd stopped without fatal or panic message, try `sudo journalctl -f -t etcd2` to get full log.

[etcd-issue]: https://github.com/coreos/etcd/issues/new
[filing-good-bugs]: http://fantasai.inkedblade.net/style/talks/filing-good-bugs/
