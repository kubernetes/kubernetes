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
        image: gcr.io/kubernetes-e2e-test-images/agnhost:2.1
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

The `agnhost` binary is a CLI with the following subcommands:

### net

The goal of this Go project is to consolidate all low-level
network testing "daemons" into one place. In network testing we
frequently have need of simple daemons (common/Runner) that perform
some "trivial" set of actions on a socket.

Usage:

* A package for each general area that is being tested, for example
  `nat/` will contain Runners that test various NAT features.
* Every runner should be registered via `main.go:makeRunnerMap()`.
* Runners receive a JSON options structure as to their configuration. `Run()`
  should return the disposition of the test.

Runners can be executed into two different ways, either through the
command-line or via an HTTP request:

Command-line:

```console
    kubectl exec test-agnhost -- /agnhost net --runner <runner> --options <json>
    kubectl exec test-agnhost -- /agnhost net \
        --runner nat-closewait-client \
        --options '{"RemoteAddr":"127.0.0.1:9999"}'
```

HTTP server:

```console
    kubectl exec test-agnhost -- /agnhost net --serve :8889
    kubectl exec test-agnhost -- curl -v -X POST localhost:8889/run/nat-closewait-server \
        -d '{"LocalAddr":"127.0.0.1:9999"}'
```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/images/net/README.md?pixel)]()

### netexec

Starts a HTTP server on given TCP / UDP ports with the following endpoints:

- `/`: Returns the request's timestamp.
- `/clientip`: Returns the request's IP address.
- `/dial`: Creates a given number of requests to the given host and port using the given protocol,
  and returns a JSON with the fields `responses` (successful request responses) and `errors` (
  failed request responses). Returns `200 OK` status code if the last request succeeded,
  `417 Expectation Failed` if it did not, or `400 Bad Request` if any of the endpoint's parameters
  is invalid. The endpoint's parameters are:
  - `host`: The host that will be dialed.
  - `port`: The port that will be dialed.
  - `request`: The HTTP endpoint or data to be sent through UDP. If not specified, it will result
    in a `400 Bad Request` status code being returned.
  - `protocol`: The protocol which will be used when making the request. Default value: `http`.
    Acceptable values: `http`, `udp`.
  - `tries`: The number of times the request will be performed. Default value: `1`.
- `/echo`: Returns the given `msg` (`/echo?msg=echoed_msg`)
- `/exit`: Closes the server with the given code (`/exit?code=some-code`). The `code`
  is expected to be an integer [0-127] or empty; if it is not, it will return an error message.
- `/healthz`: Returns `200 OK` if the server is ready, `412 Status Precondition Failed`
  otherwise. The server is considered not ready if the UDP server did not start yet or
  it exited.
- `/hostname`: Returns the server's hostname.
- `/hostName`: Returns the server's hostname.
- `/shell`: Executes the given `shellCommand` or `cmd` (`/shell?cmd=some-command`) and
  returns a JSON containing the fields `output` (command's output) and `error` (command's
  error message). Returns `200 OK` if the command succeeded, `417 Expectation Failed` if not.
- `/shutdown`: Closes the server with the exit code 0.
- `/upload`: Accepts a file to be uploaded, writing it in the `/uploads` folder on the host.
  Returns a JSON with the fields `output` (containing the file's name on the server) and
  `error` containing any potential server side errors.

Usage:

```console
    kubectl exec test-agnhost -- /agnhost netexec [--http-port <http-port>] [--udp-port <udp-port>]
```

### nettest

A tiny web server for checking networking connectivity.

Will dial out to, and expect to hear from, every pod that is a member of the service
passed in the flag `--service`.

Will serve a webserver on given `--port`, and will create the following endpoints:

- `/read`: to see the current state, or `/quit` to shut down.

- `/status`: to see `pass/running/fail` determination. (literally, it will return
one of those words.)

- `/write`: is used by other network test pods to register connectivity.

Usage:

```console
    kubectl exec test-agnhost -- /agnhost nettest [--port <port>] [--peers <peers>] [--service <service>] [--namespace <namespace>] [--delay-shutdown <delay>]
```

### webhook (Kubernetes External Admission Webhook)

The subcommand tests MutatingAdmissionWebhook and ValidatingAdmissionWebhook. After deploying
it to kubernetes cluster, administrator needs to create a ValidatingWebhookConfiguration
in kubernetes cluster to register remote webhook admission controllers.

TODO: add the reference when the document for admission webhook v1beta1 API is done.

Check the [MutatingAdmissionWebhook](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.14/#mutatingwebhookconfiguration-v1beta1-admissionregistration-k8s-io) and [ValidatingAdmissionWebhook](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.14/#validatingwebhookconfiguration-v1beta1-admissionregistration-k8s-io) documentations for more information about them.

Usage:

```console
    kubectl exec test-agnhost -- /agnhost webhook [--tls-cert-file <key-file>] [--tls-private-key-file <cert-file>]
```

## Image

The image can be found at `gcr.io/kubernetes-e2e-test-images/agnhost:2.1` for Linux
containers, and `e2eteam/agnhost:2.1` for Windows containers. In the future, the same
repository can be used for both OSes.
