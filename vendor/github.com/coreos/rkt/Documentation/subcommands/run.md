# rkt run

## Image Addressing

Images can be run by either their name, their hash, an explicit transport address, or a Docker registry URL.
rkt will automatically [fetch][fetch] them if they're not present in the local store.

### Run By Name
```
# rkt run coreos.com/etcd:v2.0.0
```

### Run By Hash
```
# rkt run sha512-fa1cb92dc276b0f9bedf87981e61ecde
```

### Run By ACI Address
```
# rkt run https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.aci
```

### Run From a Docker Registry
```
# rkt --insecure-options=image run docker://quay.io/coreos/etcd:v2.0.0
```

## Run multiple applications in the same pod

Multiple applications can be run in a pod by passing multiple images to the run command:
```
# rkt run example.com/app1 example.com/app2
```

## Overriding the app's name

Be default, the image's name will be used as the app's name.
It can be overridden by rkt using the `--name` flag.
This comes handy when we want to run multiple apps using the same image:

```
# rkt --insecure-options=image run docker://busybox --name=busybox1 docker://busybox --name=busybox2
```

## Overriding Executable to launch

Application images include an `exec` field that specifies the executable to launch.
This executable can be overridden by rkt using the `--exec` flag:

```
# rkt --insecure-options=image run docker://busybox --exec /bin/date
```

## Overriding Isolators

Application images can include per-app isolators and some of them can be overridden by rkt.
The units come from [the Kubernetes resource model][k8s-resources].
In the following example, the CPU isolator is defined to 750 milli-cores and the memory isolator limits the memory usage to 128MB.

```
# rkt run coreos.com/etcd:v2.0.0 --cpu=750m --memory=128M
```

## Overriding User/Group

Application images must specify the username/group or the UID/GID the app is to be run as as specified in the [Image Manifest Schema][image-manifest-schema]. The user/group can be overridden by rkt using the `--user` and `--group` flags:

```
# rkt --insecure-options=image run docker://busybox --user=1000 --group=100 --exec id
```

## Passing Arguments

To pass additional arguments to images use the pattern of `image1 -- [image1 flags] --- image2 -- [image2 flags]`.
For example:

```
# rkt run example.com/worker -- --loglevel verbose --- example.com/syncer -- --interval 30s
```

This can be combined with overridden executables:

```
# rkt run example.com/worker --exec /bin/ov -- --loglevel verbose --- example.com/syncer --exec /bin/syncer2 -- --interval 30s
```

## Adding user annotations and user labels

Additional annotations and labels can be added to the app by using `--user-annotation` and `--user-label` flag.
The annotations and labels will appear in the app's `UserAnnotations` and `UserLabels` field.

```
# rkt run example.com/example --user-annotation=foo=bar --user-label=hello=world
```

## Influencing Environment Variables

To inherit all environment variables from the parent, use the `--inherit-env` flag.

To explicitly set environment variables for all apps, use the `--set-env` flag.

To explicitly set environment variables for all apps from a file, use the `--set-env-file` flag.
Variables are expected to be in the format `VAR_NAME=VALUE` separated by the new line character `\n`.
Lines starting with `#` or `;` and empty ones will be ignored.

To explicitly set environment variables for each app individually, use the `--environment` flag.

The precedence is as follows with the last item replacing previous environment entries:

- Parent environment
- App image environment
- Explicitly set environment variables for all apps from file (`--set-env-file`)
- Explicitly set environment variables for all apps on command line (`--set-env`)
- Explicitly set environment variables for each app on command line (`--environment`)

```
# export EXAMPLE_ENV=hello
# export EXAMPLE_OVERRIDE=under
# rkt run --inherit-env --set-env=FOO=bar --set-env=EXAMPLE_OVERRIDE=over example.com/env-printer
EXAMPLE_ENV=hello
FOO=bar
EXAMPLE_OVERRIDE=over

# export EXAMPLE_ENV=hello
# export EXAMPLE_OVERRIDE=under
# rkt run --inherit-env --set-env=FOO=bar --set-env=EXAMPLE_OVERRIDE=over example.com/env-printer --environment=EXAMPLE_OVERRIDE=ride
EXAMPLE_ENV=hello
FOO=bar
EXAMPLE_OVERRIDE=ride
```

## Disable Signature Verification

If desired, `--insecure-options=image` can be used to disable this security check:

```
# rkt --insecure-options=image run coreos.com/etcd:v2.0.0
rkt: searching for app image coreos.com/etcd:v2.0.0
rkt: fetching image from https://github.com/coreos/etcd/releases/download/v2.0.0/etcd-v2.0.0-linux-amd64.aci
rkt: warning: signature verification has been disabled
...
```

## Mount Volumes into a Pod

Each ACI can define a [list of mount points][image-manifest-schema] that the app is expecting external data to be mounted into:

```json
{
    "acKind": "ImageManifest",
    "name": "example.com/app1",
    ...
    "app": {
        ...
        "mountPoints": [
            {
                "name": "data",
                "path": "/var/data",
                "readOnly": false,
                "recursive": true
            }
        ]
    }
    ...
}
```

To fulfill these mount points, volumes are used.
A volume is assigned to a mount point if they both have the same name.
There are today two kinds of volumes:

- `host` volumes that can expose a directory or a file from the host to the pod.
- `empty` volumes that initialize an empty storage to be accessed locally within the pod. When the pod is garbage collected, it will be removed.

Each volume can be selectively mounted into each application at differing mount points.
Note that any volumes that are specified but do not have a matching mount point (or [`--mount` flag](#mounting-volumes-without-mount-points)) will be silently ignored.

If a mount point is specified in the image manifest but no matching volume is found, an implicit `empty` volume will be created automatically.

### Mounting Volumes

Volumes are defined via the `--volume` flag, the volume is then mounted into each app running in the pod based on information defined in the ACI manifest.

There are two kinds of volumes, `host` and `empty`.

#### Host Volumes

For `host` volumes, the `--volume` flag allows you to specify the volume name, the location on the host, and whether the volume is read-only or not.
The volume name and location on the host are mandatory.
The read-only parameter is false by default.

Syntax:

```
--volume NAME,kind=host,source=SOURCE_PATH,readOnly=BOOL
```

In the following example, we make the host's `/srv/data` accessible to app1 on `/var/data`:

```
# rkt run --volume data,kind=host,source=/srv/data,readOnly=false example.com/app1
```

If you don't intend to persist the data and you just want to have a volume shared between all the apps in the pod, you can use an `empty` volume:

```
# rkt run --volume data,kind=empty,readOnly=false example.com/app1
```

#### Empty Volumes

For `empty` volumes, the `--volume` flag allows you to specify the volume name, and the mode, UID and GID of the generated volume.
The volume name is mandatory.
By default, `mode` is `0755`, UID is `0` and GID is `0`.

Syntax:

 `--volume NAME,kind=empty,mode=MODE,uid=UID,gid=GID`

 In the following example, we create an empty volume for app1's `/var/data`:

 ```
 # rkt run --volume data,kind=empty,mode=0700,uid=0,gid=0
 ```

### Mounting Volumes without Mount Points

If the ACI doesn't have any mount points defined in its manifest, you can still mount volumes using the `--mount` flag.

With `--mount` you define a mapping between volumes and a path in the app.
This will supplement and override any mount points in the image manifest.
In the following example, the `--mount` option is positioned after the app name; it defines the mount only in that app:

```
# rkt run --volume logs,kind=host,source=/srv/logs \
        example.com/app1 --mount volume=logs,target=/var/log \
        example.com/app2 --mount volume=logs,target=/opt/log
```

In the following example, the `--mount` option is positioned before the app names.
It defines mounts on all apps: both app1 and app2 will have `/srv/logs` accessible on `/var/log`.

```
# rkt run --volume logs,kind=host,source=/srv/logs \
       --mount volume=data,target=/var/log \
        example.com/app1 example.com/app2
```

### MapReduce Example

Let's say we want to read data from the host directory `/opt/tenant1/work` to power a MapReduce-style worker.
We'll call this app `example.com/reduce-worker`.

We also want this data to be available to a backup application that runs alongside the worker (in the same pod).
We'll call this app `example.com/worker-backup`.
The backup application only needs read-only access to the data.

Below we show the abbreviated manifests for the respective applications (recall that the manifest is bundled into the application's ACI):

```json
{
    "acKind": "ImageManifest",
    "name": "example.com/reduce-worker",
    ...
    "app": {
        ...
        "mountPoints": [
            {
                "name": "work",
                "path": "/var/lib/work",
                "readOnly": false
            }
        ],
        ...
    }
    ...
}
```

```json
{
    "acKind": "ImageManifest",
    "name": "example.com/worker-backup",
    ...
    "app": {
        ...
        "mountPoints": [
            {
                "name": "work",
                "path": "/backup",
                "readOnly": true
            }
        ],
        ...
    }
    ...
}
```

In this case, both apps reference a volume they call "work", and expect it to be made available at `/var/lib/work` and `/backup` within their respective root filesystems.

Since they reference the volume using an abstract name rather than a specific source path, the same image can be used on a variety of different hosts without being coupled to the host's filesystem layout.

To tie it all together, we use the `rkt run` command-line to provide them with a volume by this name. Here's what it looks like:

```
# rkt run --volume=work,kind=host,source=/opt/tenant1/work \
  example.com/reduce-worker \
  example.com/worker-backup
```

If the image didn't have any mount points, you can achieve a similar effect with the `--mount` flag (note that both would be read-write though):

```
# rkt run --volume=work,kind=host,source=/opt/tenant1/work \
  example.com/reduce-worker --mount volume=work,target=/var/lib/work \
  example.com/worker-backup --mount volume=work,target=/backup
```

Now when the pod is running, the two apps will see the host's `/opt/tenant1/work` directory made available at their expected locations.

## Enabling metadata service registration

By default, `rkt run` will not register the pod with the [metadata service][metadata-service].
You can enable registration with the `--mds-register` command line option.

## Pod Networking

The `run` subcommand features the `--net` argument which takes options to configure the pod's network.

### Default contained networking

When the argument is not given, `--net=default` is automatically assumed and the default contained network will be loaded.

### Host networking

Simplified, with `--net=host` the apps within the pod will share the network stack and the interfaces with the host machine.

```
# rkt run --net=host coreos.com/etcd:v2.0.0
```

Strictly seen, this is only true when `rkt run` is invoked on the host directly, because the network stack will be inherited from the process that is invoking the `rkt run` command.

### Other Networking Examples

More details about rkt's networking options and examples can be found in the [networking documentation][net-overview].

## Run rkt as a Daemon

rkt doesn't include any built-in support for running as a daemon.
However, since it is a regular process, you can use your init system to achieve the same effect.

For example, if you use systemd, you can [run rkt using `systemd-run`][systemd-run].

If you don't use systemd, you can use [daemon][daemon] as an alternative.

## Use a Custom Stage1

rkt is designed and intended to be modular, using a [staged architecture][rkt-arch].

You can use a custom stage1 by using the `--stage1-{url,path,name,hash,from-dir}` flags.

```
# rkt --stage1-path=/tmp/stage1.aci run coreos.com/etcd:v2.0.0
```

rkt expects stage1 images to be signed except in the following cases:

* it is the default stage1 image and it's in the same directory as the rkt binary
* `--stage1-{name,hash}` is used and the image is already in the store
* `--stage1-{url,path,from-dir}` is used and the image is in the default directory configured at build time

For more details see the [hacking documentation][rkt-hacking].

## Disabling overlay

rkt uses overlayfs by default when running application containers. This provides immense benefits to performance and efficiency: start times for large containers are much faster, and multiple pods using the same images will consume less disk space and can share page cache entries.

This feature will be disabled automatically if the underlying filesystem does not support overlay fs, see the [prepare][prepare] subcommand for details. This feature can also be explicitly disabled with the `--no-overlay` option:

```
# rkt run --no-overlay=true --insecure-options=image coreos.com/etcd:v2.0.0
```

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--user-annotation` | none | annotation add to the app's UserAnnotations field | Set the app's annotations (example: '--user-annotation=foo=bar'). |
| `--caps-remove` | none | capability to remove (e.g. `--caps-remove=CAP_SYS_CHROOT,CAP_MKNOD`) | Capabilities to remove from the process's capabilities bounding set; all others from the default set will be included. |
| `--caps-retain` | none | capability to retain (e.g. `--caps-retain=CAP_SYS_ADMIN,CAP_NET_ADMIN`) | Capabilities to retain in the process's capabilities bounding set; all others will be removed. |
| `--cpu` | none | CPU units (e.g. `--cpu=500m`) | CPU limit for the preceding image in [Kubernetes resource model][k8s-resources] format. |
| `--dns` | none | IP Address | Name server to write in `/etc/resolv.conf`. It can be specified several times. |
| `--dns-opt` | none | DNS option | DNS option from resolv.conf(5) to write in `/etc/resolv.conf`. It can be specified several times. |
| `--dns-search` | none | Domain name | DNS search domain to write in `/etc/resolv.conf`. It can be specified several times. |
| `--environment` | none | environment variables add to the app's environment variables | Set the app's environment variables (example: '--environment=foo=bar'). |
| `--exec` | none | Path to executable | Override the exec command for the preceding image. |
| `--group` | root | gid, groupname or file path (e.g. `--group=core`) | Group override for the preceding image. |
| `--hostname` | `rkt-$PODUUID` | A host name | Set pod's host name. |
| `--inherit-env` | `false` | `true` or `false` | Inherit all environment variables not set by apps. |
| `--interactive` | `false` | `true` or `false` | Run pod interactively. If true, only one image may be supplied. |
| `--user-label` | none | label add to the apps' UserLabels field | Set the app's labels (example: '--user-label=foo=bar'). |
| `--mds-register` | `false` | `true` or `false` | Register pod with metadata service. It needs network connectivity to the host (`--net` as `default`, `default-restricted`, or `host`). |
| `--memory` | none | Memory units (e.g. `--memory=50M`) | Memory limit for the preceding image in [Kubernetes resource model][k8s-resources] format. |
| `--mount` | none | Mount syntax (e.g. `--mount volume=NAME,target=PATH`) | Mount point binding a volume to a path within an app. See [Mounting Volumes without Mount Points](#mounting-volumes-without-mount-points). |
| `--name` | none | Name of the app | Set the name of the app (example: '--name=foo'). If not set, then the app name default to the image's name |
| `--net` | `default` | A comma-separated list of networks. (e.g. `--net[=n[:args], ...]`) | Configure the pod's networking. Optionally, pass a list of user-configured networks to load and set arguments to pass to each network, respectively. |
| `--no-overlay` | `false` | `true` or `false` | Disable the overlay filesystem. |
| `--pod-manifest` | none | A path | The path to the pod manifest. If it's non-empty, then only `--net`, `--no-overlay` and `--interactive` will have effect. |
| `--port` | none | A port name and number pair | Container port name to expose through host port number. Requires [contained network][contained]. Syntax: `--port=NAME:HOSTPORT` The NAME is that given in the ACI. By convention, Docker containers' EXPOSEd ports are given a name formed from the port number, a hyphen, and the protocol, e.g., `80-tcp`, giving something like `--port=80-tcp:8080`. |
| `--private-users` | `false` | `true` or `false` | Run within user namespaces. |
| `--set-env` | none | An environment variable (e.g. `--set-env=NAME=VALUE`) | An environment variable to set for apps. |
| `--set-env-file` | none | Path of an environment variables file (e.g. `--set-env-file=/path/to/env/file`) | Environment variables to set for apps. |
| `--signature` | none | A file path | Local signature file to use in validating the preceding image. |
| `--stage1-from-dir` | none | Image name (e.g. `--stage1-name=coreos.com/rkt/stage1-coreos`) | A stage1 image file name to search for inside the default stage1 images directory. |
| `--stage1-hash` | none | Image hash (e.g. `--stage1-hash=sha512-dedce9f5ea50`) | A hash of a stage1 image. The image must exist in the store. |
| `--stage1-name` | none | Image name (e.g. `--stage1-name=coreos.com/rkt/stage1-coreos`) | A name of a stage1 image. Will perform a discovery if the image is not in the store. |
| `--stage1-path` | none | Absolute or relative path | A path to a stage1 image. |
| `--stage1-url` | none | URL with protocol | A URL to a stage1 image. HTTP/HTTPS/File/Docker URLs are supported. |
| `--pull-policy` | `new` | `never`, `new`, or `update` | Sets the policy for when to fetch an image. See [image fetching behavior][img-fetch] |
| `--user` | none | uid, username or file path (e.g. `--user=core`) | User override for the preceding image. |
| `--uuid-file-save` | none | A file path | Write out the pod UUID to a file. |
| `--volume` | none | Volume syntax (e.g. `--volume NAME,kind=KIND,source=PATH,readOnly=BOOL`) | Volumes to make available in the pod. See [Mount Volumes into a Pod](#mount-volumes-into-a-pod). |

## Per-application options

| Flag | Default | Options | Description |
| ---  | ---     | ---     | ---         |
| `--stdin`  | "null" | "null", "tty", "stream"        | Mode for this application stdin.  |
| `--stdout` | "log"  | "null", "tty", "stream", "log" | Mode for this application stdout. |
| `--stderr` | "log"  | "null", "tty", "stream", "log" | Mode for this application stderr. |

## Global options

See the table with [global options in general commands documentation][global-options].


[contained]: ../networking/overview.md#contained-mode
[daemon]: http://www.libslack.org/daemon/
[fetch]: fetch.md
[global-options]: ../commands.md#global-options
[image-manifest-schema]: https://github.com/appc/spec/blob/master/spec/aci.md#image-manifest-schema
[img-fetch]: ../image-fetching-behavior.md
[k8s-resources]: https://github.com/kubernetes/kubernetes/blob/release-1.2/docs/design/resources.md
[metadata-service]: metadata-service.md
[net-overview]: ../networking/overview.md
[prepare]: prepare.md
[rkt-arch]: ../devel/architecture.md
[rkt-hacking]: ../hacking.md
[systemd-run]: ../using-rkt-with-systemd.md#systemd-run
