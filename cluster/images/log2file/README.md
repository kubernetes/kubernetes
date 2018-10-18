# log2file

## Usage

Log to file executes the given command, and appends its stdout & stderr to the
specified file.

```
Usage: ./log2file [options] -- [command] [args...]
  -out string
    	The file to redirect stdout & stderr to (append).
```

### Example

The previous pattern of using shell redirection to log kube-apiserver output:

```
/bin/sh -c "exec /usr/local/bin/kube-apiserver {{params}} --allow-privileged={{pillar['allow_privileged']}} 1>>/var/log/kube-apiserver.log 2>&1"
```

Becomes:

```
log2file --out=/var/log/kube-apiserver.log -- /usr/local/bin/kube-apiserver {{params}} --allow-privileged={{pillar['allow_privileged']}}
```

## But why?

Kubernetes makes extensive usage of the glog logging library, which ([amoung
other problems][glog-issue]) cannot send the log output to a specific file.

To work around this, kubernetes containers would run a shell wrapper around the
binary, and use redirection to send the output to a log file. Problems with this
approach include:

1. Including a functional shell (& helper binaries) increases the tools
   available to attackers, making certain types of exploits much easier.
2. Vulnerabilities in the included dependencies must be managaged, creating toil
   for Kubernetes devs.

`log2file` replaces the fully functional shell with a minimal binary that does
exactly 1 thing: run a command, and redirect it's output (both stdout & stderr)
to a file (append mode).

### OK, but why not just fix glog?

Great question! We [want to move to a different log library][glog-issue], but
such changes take a long time in Kubernetes. This utility is a short-term
solution to the immediate problems causing toil for us.

Once our chosen log library supports specifying an output file directly, this
utility should be deleted, and images using it should be rebased on scratch.

[glog-issue]: https://github.com/kubernetes/kubernetes/issues/61006
