# rkt Commands

Work in progress.
Please contribute if you see an area that needs more detail.

## Downloading Images (ACIs)

rkt runs applications packaged according to the open-source [App Container Image][aci-images] specification.
ACIs consist of the root filesystem of the application container, a manifest, and an optional signature.

ACIs are named with a URL-like structure.
This naming scheme allows for a decentralized discovery of ACIs, related signatures and public keys.
rkt uses these hints to execute [meta discovery][appc-discovery].

* [trust](subcommands/trust.md)
* [fetch](subcommands/fetch.md)

## Running Pods

rkt can execute ACIs identified by name, hash, local file path, or URL.
If an ACI hasn't been cached on disk, rkt will attempt to find and download it.
To use rkt's [metadata service][metadata-spec], enable registration with the `--mds-register` flag when [invoking it][rkt-mds].

* [run](subcommands/run.md)
* [stop](subcommands/stop.md)
* [enter](subcommands/enter.md)
* [prepare](subcommands/prepare.md)
* [run-prepared](subcommands/run-prepared.md)

## Pod inspection and management

rkt provides subcommands to list, get status, and clean its pods.

* [list](subcommands/list.md)
* [status](subcommands/status.md)
* [export](subcommands/export.md)
* [gc](subcommands/gc.md)
* [rm](subcommands/rm.md)
* [cat-manifest](subcommands/cat-manifest.md)

## Interacting with the local image store

rkt provides subcommands to list, inspect and export images in its local store.

* [image](subcommands/image.md)

## Metadata Service

The metadata service helps running apps introspect their execution environment and assert their pod identity.

* [metadata-service](subcommands/metadata-service.md)

## API Service

The API service allows clients to list and inspect pods and images running under rkt.

* [api-service](subcommands/api-service.md)

## Misc

* [version](subcommands/version.md)
* [config](subcommands/config.md)

## Global Options

In addition to the flags used by individual `rkt` commands, `rkt` has a set of global options that are applicable to all commands.

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--cpuprofile (hidden flag)` | ''  | A file path | Write CPU profile to the file |
| `--debug` |  `false` | `true` or `false` | Prints out more debug information to `stderr` |
| `--dir` | `/var/lib/rkt` | A directory path | Path to the `rkt` data directory |
| `--insecure-options` |  none | **none**, **http**, **image**, **tls**, **ondisk**, **pubkey**, **capabilities**, **paths**, **seccomp**, **all-fetch**, **all-run**, **all** <br/> More information below. | Comma-separated list of security features to disable |
| `--local-config` |  `/etc/rkt` | A directory path | Path to the local configuration directory |
| `--memprofile (hidden flag)` | '' | A file path | Write memory profile to the file |
| `--system-config` |  `/usr/lib/rkt` | A directory path | Path to the system configuration directory |
| `--trust-keys-from-https` |  `false` | `true` or `false` | Automatically trust gpg keys fetched from HTTPS (or HTTP if the insecure `pubkey` option is also specified) |
| `--user-config` | '' | A directory path | Path to the user configuration directory |

### `--insecure-options`

- **none**: All security features are enabled
- **http**: Allow HTTP connections. Be warned that this will send any credentials as clear text, allowing anybody with access to your network to obtain them. It will also perform no validation of the remote server, making it possible for an attacker to impersonate the remote server. This applies specifically to fetching images, signatures, and gpg pubkeys.
- **image**: Disables verifying image signatures. If someone is able to replace the image on the server with a modified one or is in a position to impersonate the server, they will be able to force you to run arbitrary code.
- **tls**: Accept any certificate from the server and any host name in that certificate. This will make it possible for attackers to spoof the remote server and provide malicious images.
- **ondisk**: Disables verifying the integrity of the on-disk, rendered image before running. This significantly speeds up start time. If an attacker is able to modify the contents of your local filesystem, this will allow them to cause you to run arbitrary malicious code.
- **pubkey**: Allow fetching pubkeys via insecure connections (via HTTP connections or from servers with unverified certificates). This slightly extends the meaning of the `--trust-keys-from-https` flag. This will make it possible for an attacker to spoof the remote server, potentially providing fake keys and allowing them to provide container images that have been tampered with.
- **capabilities**: Gives all [capabilities][capabilities] to apps. This allows an attacker that is able to execute code in the container to trivially escalate to root privileges on the host. 
- **paths**: Disables inaccessible and read-only paths. This makes it easier for an attacker who can gain control over a single container to execute code in the host system, potentially allowing them to escape from the container. This also leaks additional information.
- **seccomp**: Disables [seccomp][seccomp]. This increases the attack surface available to an attacker who can gain control over a single container, potentially making it easier for them to escape from the container.
- **all-fetch**: Disables the following security checks: image, tls, http
- **all-run**: Disables the following security checks: ondisk, capabilities, paths, seccomp
- **all**: Disables all security checks

## Logging

By default, rkt will send logs directly to stdout/stderr, allowing them to be captured by the invoking process.
On host systems running systemd, rkt will attempt to integrate with journald on the host.
In this case, the logs can be accessed directly via journalctl.

#### Accessing logs via journalctl

To read the logs of a running pod, get the pod's machine name from `machinectl`:

```
$ machinectl
MACHINE                                  CLASS     SERVICE
rkt-bc3c1451-2e81-45c6-aeb0-807db44e31b4 container rkt

1 machines listed.
```

or `rkt list --full`

```
$ rkt list --full
UUID                                  APP    IMAGE NAME                              IMAGE ID             STATE    CREATED                             STARTED                             NETWORKS
bc3c1451-2e81-45c6-aeb0-807db44e31b4  etcd   coreos.com/etcd:v2.3.4                  sha512-7f05a10f6d2c  running  2016-05-18 10:07:35.312 +0200 CEST  2016-05-18 10:07:35.405 +0200 CEST  default:ip4=172.16.28.83
                                      redis  registry-1.docker.io/library/redis:3.2  sha512-6eaaf936bc76
```

The pod's machine name will be the pod's UUID prefixed with `rkt-`.
Given this machine name, logs can be retrieved by `journalctl`:

```
$ journalctl -M rkt-bc3c1451-2e81-45c6-aeb0-807db44e31b4
[...]
```

To get logs from one app in the pod:

```
$ journalctl -M rkt-bc3c1451-2e81-45c6-aeb0-807db44e31b4 -t etcd
[...]
$ journalctl -M rkt-bc3c1451-2e81-45c6-aeb0-807db44e31b4 -t redis
[...]
```

Additionaly, logs can be programmatically accessed via the [sd-journal API][sd-journal].

Currently there are two known main issues with logging in rkt:
* In some rare situations when an application inside the pod is writing to `/dev/stdout` and `/dev/stderr` (i.e. nginx) there is no way to obtain logs.
 The app should be modified so it will write to `stdout` or `syslog`. In the case of nginx:
 ```
 error_log stderr;
 
 http {
     access_log syslog:server=unix:/dev/log main;
     [...]
 }
 ```
 should be added to ```/etc/nginx/nginx.conf```

* Some applications, like etcd 3.0, write directly to journald. Such log entries will not be written to stdout or stderr.
 These logs can be retrieved by passing the machine ID to journalctl:

 ```
 $ journalctl -M rkt-bc3c1451-2e81-45c6-aeb0-807db44e31b4
 ```

 For the specific etcd case, since release 3.1.0-rc.1 it is possible to force emitting logs to stdout via a `--log-output=stdout` command-line option.

##### Stopped pod

To read the logs of a stopped pod, use:

```
journalctl -m _MACHINE_ID=132f9d560e3f4d1eba8668efd488bb62

[...]
```

On some distributions such as Ubuntu, persistent journal storage is not enabled by default. In this case, it is not possible to get the logs of a stopped pod. Persistent journal storage can be enabled with `sudo mkdir /var/log/journal` before starting the pods.


[aci-images]: https://github.com/appc/spec/blob/master/spec/aci.md#app-container-image
[appc-discovery]: https://github.com/appc/spec/blob/master/spec/discovery.md#app-container-image-discovery
[etcd-5449]: https://github.com/coreos/etcd/issues/5449
[metadata-spec]: https://github.com/appc/spec/blob/master/spec/ace.md#app-container-metadata-service
[rkt-mds]: subcommands/metadata-service.md
[sd-journal]: https://www.freedesktop.org/software/systemd/man/sd-journal.html
[capabilities]: capabilities-guide.md
[seccomp]: seccomp-guide.md
