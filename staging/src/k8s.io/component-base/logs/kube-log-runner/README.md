# `kube-log-runner` (formerly known as go-runner)

The `kube-log-runner` is a Go based binary that can run commands and redirect stdout/stderr etc.

Why do we need this?

- Some of our images like kube-apiserver used bash output redirection for
  collecting logs, so we were not able to switch to distroless images directly
  for these images. The klog's `--log-file` parameter was supposed to fix this
  problem, but we ran into trouble with that in scalability CI jobs that never
  could get root caused and fixed. Using this binary worked.

- Windows services don't have a mechanism for redirecting output of a process.

- Nowadays, the `--log-file` parameter is deprecated for Kubernetes components
  and should not be used anymore. `kube-log-runner` is a direct replacement.

For example instead of running kube-apiserver like this:
```bash
"/bin/sh",
  "-c",
  "exec kube-apiserver {{params}} --allow-privileged={{pillar['allow_privileged']}} 1>>/var/log/kube-apiserver.log 2>&1"
```

Or this:
```bash
kube-apiserver {{params}} --allow-privileged={{pillar['allow_privileged']}} --log-file=/var/log/kube-apiserver.log --alsologtostderr=false"
```

We would use `kube-log-runner` like so:
```bash
kube-log-runner -log-file=/var/log/kube-apiserver.log --also-stdout=false \
   kube-apiserver {{params}} --allow-privileged={{pillar['allow_privileged']}}
```

The kube-log-runner then ensures that we run the
`/usr/local/bin/kube-apiserver` with the specified parameters and redirect both
stdout and stderr ONLY to the log file specified. It will always append to the
log file.

Possible invocations:
```bash
# Merge stderr and stdout, write to stdout (same as 2>&1).
kube-log-runner echo "hello world"

# Redirect both into log file (same as 1>>/tmp/log 2>&1).
kube-log-runner -log-file=/tmp/log echo "hello world"

# Copy into log file and print to stdout (same as 2>&1 | tee -a /tmp/log).
kube-log-runner -log-file=/tmp/log -also-stdout echo "hello world"

# Redirect only stdout into log file (same as 1>>/tmp/log).
kube-log-runner -log-file=/tmp/log -redirect-stderr=false echo "hello world"
```

# Container base image

The Kubernetes
[`k8s.gcr.io/build-image/go-runner`](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/build-image/go-runner)
image wraps the `gcr.io/distroless/static` image and provides `kube-log-runner`
under its traditional name as `/go-runner`. It gets maintained in
https://github.com/kubernetes/release/tree/master/images/build/go-runner.

# Prebuilt binary

The Kubernetes release archives contain kube-log-runner.
