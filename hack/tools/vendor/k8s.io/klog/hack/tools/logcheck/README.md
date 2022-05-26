This directory contains a linter for checking log calls. It was originally
created to detect when unstructured logging calls like `klog.Infof` get added
to files that should only use structured logging calls like `klog.InfoS`
and now also supports other checks.

# Installation

`go install k8s.io/klog/hack/tools/logcheck`

# Usage

`$logcheck.go <package_name>`
`e.g $logcheck ./pkg/kubelet/lifecycle/`

# Configuration

Checks can be enabled or disabled globally via command line flags and env
variables. In addition, the global setting for a check can be modified per file
via a configuration file. That file contains lines in this format:

```
<checks> <regular expression>
```

`<checks>` is a comma-separated list of the names of checks that get enabled or
disabled when a file name matches the regular expression. A check gets disabled
when its name has `-` as prefix and enabled when there is no prefix or `+` as
prefix. Only checks that are mentioned explicitly are modified. All regular
expressions are checked in order, so later lines can override the previous
ones.

In this example, checking for klog calls is enabled for all files under
`pkg/scheduler` in the Kubernetes repo except for `scheduler.go`
itself. Parameter checking is disabled everywhere.

```
klog,-parameters k8s.io/kubernetes/pkg/scheduler/.*
-klog k8s.io/kubernetes/pkg/scheduler/scheduler.go
```

The names of all supported checks are the ones used as sub-section titles in
the next section.

# Checks

## structured (enabled by default)

Unstructured klog logging calls are flagged as error.

## klog (disabled by default)

None of the klog logging methods may be used. This is even stricter than
`unstructured`. Instead, code should retrieve a logr.Logger from klog and log
through that.

## parameters (enabled by default)

This ensures that if certain logging functions are allowed and are used, those
functions are passed correct parameters.

### all calls

Format strings are not allowed where plain strings are expected.

### structured logging calls

Key/value parameters for logging calls are checked:
- For each key there must be a value.
- Keys must be constant strings.

This also warns about code that is valid, for example code that collects
key/value pairs in an `[]interface` variable before passing that on to a log
call. Such valid code can use `nolint:logcheck` to disable the warning (when
invoking logcheck through golangci-lint) or the `parameters` check can be
disabled for the file.

## with-helpers (disabled by default)

`logr.Logger.WithName`, `logr.Logger.WithValues` and `logr.NewContext` must not
be used.  The corresponding helper calls from `k8s.io/klogr` should be used
instead. This is relevant when support contextual logging is disabled at
runtime in klog.
