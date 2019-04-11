# Agnhost

## Overview

There are significant differences between Linux and Windows, especially in the way
something can be obtained or tested. For example, the DNS suffix list can be found in
`/etc/resolv.conf` on Linux, but on Windows, such file does not exist, the same
information could retrieved through other means. To combat those differences,
`agnhost` was created.

`agnhost` is an extendable CLI that behaves and outputs the same expected content,
no matter the underlying OS. The name itself reflects this idea, being a portmanteau
word of the words agnost and host.

The image was created for testing purposes, reducing the need for having different test
cases for the same tested behaviour.

## Usage

The `agnhost` binary is a CLI with the following subcommands:

- `dns-suffix`: It will output the host's configured DNS suffix list, separated by commas.
- `dns-server-list`: It will output the host's configured DNS servers, separated by commas.
- `etc-hosts`: It will output the contents of host's `hosts` file. This file's location
  is `/etc/hosts` on Linux, while on Windows it is `C:/Windows/System32/drivers/etc/hosts`.
- `pause`: It will pause the execution of the binary. This can be used for containers
  which have to be kept in a `Running` state for various purposes, including executing
  other `agnhost` commands.
- `help`: Prints the binary's help menu. Additionally, it can be followed by another
  subcommand in order to get more information about that subcommand, including its
  possible arguments.

For example, let's consider the following `pod.yaml` file:

```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-agnhost
    spec:
      containers:
      - args:
        - dns-suffix
        image: gcr.io/kubernetes-e2e-test-images/agnhost:1.0
        name: agnhost
      dnsConfig:
        nameservers:
        - 1.1.1.1
        searches:
        - resolv.conf.local
      dnsPolicy: None
```

After we've used it to create a pod:

```console
    kubectl create -f pod.yaml
```

We can then check the container's output to see what is DNS suffix list the Pod was
configured with:

```console
    kubectl logs pod/test-agnhost
```

The output will be `resolv.conf.local`, as expected. Alternatively, the Pod could be
created with the `pause` argument instead, allowing us execute multiple commands:

```console
    kubectl exec test-agnhost -- /agnhost dns-suffix
    kubectl exec test-agnhost -- /agnhost dns-server-list
```

## Image

The image can be found at `gcr.io/kubernetes-e2e-test-images/agnhost:1.0` for Linux
containers, and `e2eteam/agnhost:1.0` for Windows containers. In the future, the same
repository can be used for both OSes.
