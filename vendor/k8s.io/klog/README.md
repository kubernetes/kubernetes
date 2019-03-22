klog
====

klog is a permanant fork of https://github.com/golang/glog. original README from glog is below

----

How to use klog
===============
- Replace imports for `github.com/golang/glog` with `k8s.io/klog`
- Use `klog.InitFlags(nil)` explicitly for initializing global flags as we no longer use `init()` method to register the flags
- You can now use `log-file` instead of `log-dir` for logging to a single file (See `examples/log_file/usage_log_file.go`)
- If you want to redirect everything logged using klog somewhere else (say syslog!), you can use `klog.SetOutput()` method and supply a `io.Writer`. (See `examples/set_output/usage_set_output.go`)
- For more logging conventions (See [Logging Conventions](https://github.com/kubernetes/community/blob/master/contributors/devel/logging.md))

### Coexisting with glog
This package can be used side by side with glog. [This example](examples/coexist_glog/coexist_glog.go) shows how to initialize and syncronize flags from the global `flag.CommandLine` FlagSet. In addition, the example makes use of stderr as combined output by setting `alsologtostderr` (or `logtostderr`) to `true`.

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community page](http://kubernetes.io/community/).

You can reach the maintainers of this project at:

- [Slack](https://kubernetes.slack.com/messages/sig-architecture)
- [Mailing List](https://groups.google.com/forum/#!forum/kubernetes-sig-architecture)

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).

----

glog
====

Leveled execution logs for Go.

This is an efficient pure Go implementation of leveled logs in the
manner of the open source C++ package
	https://github.com/google/glog

By binding methods to booleans it is possible to use the log package
without paying the expense of evaluating the arguments to the log.
Through the -vmodule flag, the package also provides fine-grained
control over logging at the file level.

The comment from glog.go introduces the ideas:

	Package glog implements logging analogous to the Google-internal
	C++ INFO/ERROR/V setup.  It provides functions Info, Warning,
	Error, Fatal, plus formatting variants such as Infof. It
	also provides V-style logging controlled by the -v and
	-vmodule=file=2 flags.

	Basic examples:

		glog.Info("Prepare to repel boarders")

		glog.Fatalf("Initialization failed: %s", err)

	See the documentation for the V function for an explanation
	of these examples:

		if glog.V(2) {
			glog.Info("Starting transaction...")
		}

		glog.V(2).Infoln("Processed", nItems, "elements")


The repository contains an open source version of the log package
used inside Google. The master copy of the source lives inside
Google, not here. The code in this repo is for export only and is not itself
under development. Feature requests will be ignored.

Send bug reports to golang-nuts@googlegroups.com.
