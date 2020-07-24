# Kubernetes go-runner image

The Kubernetes go-runner image wraps the gcr.io/distroless/static image and provides a go based
binary that can run commands and wrap stdout/stderr etc. 

Why do we need this? Some of our images like kube-apiserver currently use bash for collecting
logs, so we are not able to switch to distroless images directly for these images. The klog's
`--log-file` was supposed to fix this problem, but we ran into trouble in scalability CI jobs
around log rotation and picked this option instead. we essentially publish a multi-arch 
manifest with support for various platforms. This can be used as a base for other kubernetes
components.

For example instead of running kube-apiserver like this:
```bash
"/bin/sh",
  "-c",
  "exec /usr/local/bin/kube-apiserver {{params}} --allow-privileged={{pillar['allow_privileged']}} 1>>/var/log/kube-apiserver.log 2>&1"
```  

we would use go-runner like so:
```bash
"/go-runner", "--log-file=/var/log/kube-apiserver.log", "--also-stdout=false", "--redirect-stderr=true",
  "/usr/local/bin/kube-apiserver",
  "--allow-privileged={{pillar['allow_privileged']}}",
  {{params}}
```

The go-runner would then ensure that we run the `/usr/local/bin/kube-apiserver` with the 
specified parameters and redirect stdout ONLY to the log file specified and ensure anything 
logged to stderr also ends up in the log file.
