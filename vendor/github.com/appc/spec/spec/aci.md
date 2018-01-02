## App Container Image

An *App Container Image* (ACI) contains all files and metadata needed to execute a given app.
In some ways an ACI can be thought of as equivalent to a static binary.

### Image Layout

The on-disk layout of an App Container Image is straightforward.
It includes a *rootfs* directory with all of the files that will exist in the root of the app, and an *app image manifest* file describing the contents of the image and how to execute the app.

```
/manifest
/rootfs
/rootfs/usr/bin/data-downloader
/rootfs/usr/bin/reduce-worker
```

### Image Archives

The ACI file format ("image archive") aims for flexibility and relies on standard and common technologies: HTTP, gpg, tar and gzip.
This set of formats makes it easy to build, host and secure an ACI using technologies that are widely available and battle-tested.

- Image archives MUST be named with the suffix `.aci`, irrespective of compression/encryption (see below).
- Image archives MUST be a tar formatted file with no duplicate entries.
- Image archives MUST have only two top-level pathnames, `manifest` (a regular file) and `rootfs` (a directory). Image archives with additional files outside of `rootfs` are not valid.
- All files in the image MUST maintain all of their original properties, including timestamps, Unix modes, and extended attributes (xattrs).
- Image archives MAY be compressed with `gzip`, `bzip2`, or `xz`.
- Image archives MAY be encrypted using PGP symmetric encryption with AES cipher, after optional compression.
- Image archives SHOULD be signed using PGP, the format MUST be ascii-armored detached signature mode.
- Image signatures MUST be named with the suffix `.aci.asc`.

**Security Note**: Compressing the image will often change the image size significantly. When encryption is applied after compression, it could lead to information leakage, that would not have been revealed without compression, due to such observable difference. Implementations supporting encryption SHOULD include an option to disable compression.

The following example demonstrates the creation of a simple ACI using common command-line tools.
In this case, the ACI is compressed, encrypted, and signed.

```bash
tar cvf reduce-worker.tar manifest rootfs
gzip reduce-worker.tar -c > reduce-worker.aci
gpg --output reduce-worker.aci.out --digest-algo sha256 --cipher-algo AES256 --passphrase sekr3t --symmetric reduce-worker.aci
mv reduce-worker.aci.out reduce-worker.aci
gpg --armor --output reduce-worker.aci.asc --detach-sig reduce-worker.aci
```

**Note**: the key distribution mechanism to facilitate image signature validation is not defined here.
Implementations of the App Container spec will need to provide a mechanism for users to configure the list of signing keys to trust, or use the key discovery described in [App Container Image Discovery](discovery.md).

An example App Container Image builder is [actool](https://github.com/appc/spec/tree/master/actool).

### Image ID

An image is addressed and verified against the hash of its uncompressed tar file, known as its _image ID_.
The image ID provides a way to uniquely and globally reference an image, and verify its integrity at any point.
An image ID is canonically represented as a string prefixed by the algorithm used (e.g. sha512-a83...): this format and the allowed hash algorithms are defined by the [Image ID Type](types.md#image-id-type).

```bash
echo sha512-$(sha512sum reduce-worker.tar | awk '{print $1}')
```

### Image Manifest

The [image manifest](#image-manifest-schema) is a [JSON](https://tools.ietf.org/html/rfc4627) file that includes details about the contents of the ACI, and optionally information about how to execute a process inside the ACI's rootfs.
If included, execution details include mount points that must exist, the user, the command args, default cgroup settings and more.
The manifest MAY also define binaries to execute in response to lifecycle events of the main process such as *pre-start* and *post-stop*.

Image manifests MUST be valid JSON located in the file `manifest` in the root of the image archive.
Image manifests MAY specify dependencies, which describe how to assemble the final rootfs from a collection of other images.
As an example, an app might require special certificates to be layered into its filesystem.
In this case, the app can reference the name "example.com/trusted-certificate-authority" as a dependency in the image manifest.
The dependencies are applied in the order defined in the [Filesystem Setup specification](ace.md#filesystem-setup) and each image dependency can overwrite files from the previous dependency.
Execution details specified in image dependencies are ignored.
An optional *path whitelist* can be provided, in which case all non-specified files from all dependencies will be omitted in the final, assembled rootfs.

### Image Manifest Schema

JSON Schema for the Image Manifest (app image manifest, ACI manifest), conforming to [RFC4627](https://tools.ietf.org/html/rfc4627)

```json
{
    "acKind": "ImageManifest",
    "acVersion": "0.8.9",
    "name": "example.com/reduce-worker",
    "labels": [
        {
            "name": "version",
            "value": "1.0.0"
        },
        {
            "name": "arch",
            "value": "amd64"
        },
        {
            "name": "os",
            "value": "linux"
        }
    ],
    "app": {
        "exec": [
            "/usr/bin/reduce-worker",
            "--quiet"
        ],
        "user": "100",
        "group": "300",
        "supplementaryGids": [
                400,
                500
        ],
        "eventHandlers": [
            {
                "exec": [
                    "/usr/bin/data-downloader"
                ],
                "name": "pre-start"
            },
            {
                "exec": [
                    "/usr/bin/deregister-worker",
                    "--verbose"
                ],
                "name": "post-stop"
            }
        ],
        "workingDirectory": "/opt/work",
        "environment": [
            {
                "name": "REDUCE_WORKER_DEBUG",
                "value": "true"
            }
        ],
        "isolators": [
            {
                "name": "resource/cpu",
                "value": {
                    "request": "250m",
                    "limit": "500m"
                }
            },
            {
                "name": "resource/memory",
                "value": {
                    "request": "1G",
                    "limit": "2G"
                }
            },
            {
                "name": "os/linux/capabilities-retain-set",
                "value": {
                    "set": ["CAP_NET_BIND_SERVICE"]
                }
            }
        ],
        "mountPoints": [
            {
                "name": "work",
                "path": "/var/lib/work",
                "readOnly": false
            }
        ],
        "ports": [
            {
                "name": "health",
                "port": 4000,
                "protocol": "tcp",
                "socketActivated": true
            },
            {
                "name": "ftp-data",
                "port": 20000,
                "count": 1000,
                "protocol": "tcp"
            }
        ]
    },
    "dependencies": [
        {
            "imageName": "example.com/reduce-worker-base",
            "imageID": "sha512-...",
            "labels": [
                {
                    "name": "os",
                    "value": "linux"
                },
                {
                    "name": "env",
                    "value": "canary"
                }
            ],
            "size": 22017258
        }
    ],
    "pathWhitelist": [
        "/etc/ca/example.com/crt",
        "/usr/bin/map-reduce-worker",
        "/opt/libs/reduce-toolkit.so",
        "/etc/reduce-worker.conf",
        "/etc/systemd/system/"
    ],
    "annotations": [
        {
            "name": "authors",
            "value": "Carly Container <carly@example.com>, Nat Network <[nat@example.com](mailto:nat@example.com)>"
        },
        {
            "name": "created",
            "value": "2014-10-27T19:32:27.67021798Z"
        },
        {
            "name": "documentation",
            "value": "https://example.com/docs"
        },
        {
            "name": "homepage",
            "value": "https://example.com"
        },
        {
            "name": "appc.io/executor/supports-systemd-notify",
            "value": false
        }
    ]
}
```

* **acKind** (string, required) must be an [AC Kind](types.md#ac-kind-type) of value "ImageManifest"
* **acVersion** (string, required) represents the version of the schema specification [AC Version Type](types.md#ac-version-type)
* **name** (string, required) a human-readable name for this App Container Image (string, restricted to the [AC Identifier](types.md#ac-identifier-type) formatting). This is not expected to be unique (see the **version** label) but SHOULD have a URL-like structure to facilitate **[App Container Image Discovery](discovery.md)**. If this image is resolved through the discovery process, this field MUST match the name used for discovery.
* **labels** (list of objects, optional) used during image discovery and dependency resolution. The listed objects must have two key-value pairs: *name* is restricted to the [AC Identifier](types.md#ac-identifier-type) formatting and *value* is an arbitrary string. Label names must be unique within the list, and (to avoid confusion with the image's name) cannot be "name". Several well-known labels are defined:
    * **version** when combined with "name", this SHOULD be unique for every build of an app (on a given "os"/"arch" combination).
    * **os**, **arch** can together be considered to describe the syscall ABI this image requires. **arch** is meaningful only if **os** is provided. If one or both values are not provided, the image is assumed to be OS- and/or architecture-independent. Currently supported combinations are listed in the [`types.ValidOSArch`](../schema/types/labels.go) variable, which can be updated by an implementation that supports other combinations. The combinations whitelisted by default are (in format `os/arch`): `linux/amd64`, `linux/i386`, `freebsd/amd64`, `freebsd/i386`, `freebsd/arm`, `darwin/x86_64`, `darwin/i386`. See the [Operating System spec](OS-SPEC.md) for the environment apps can expect to run in given a known **os** label.
* **app** (object, optional) if present, defines the default parameters that can be used to execute this image as an application.
    * **exec** (list of strings, optional) executable to launch and any flags. ACE duplicates the actions of the shell in searching for the executable file. If the specified filename does not contain a slash (`/`) character, the executable is sought according to the `PATH` environment variable. If `PATH` isn't defined, the executable is sought in the current directory followed by the list of directories returned by `confstr(_CS_PATH)`. If the specified filename includes a slash character, then `PATH` is ignored, and the file at the specified pathname is executed. (See `man exec(3)` for more details). ACE MAY append or override the list. These strings are not evaluated in any way and environment variables are not substituted.
    * **user**, **group** (string, required) indicates either the username/group name or the UID/GID the app is to be run as (freeform string). The user and group values may be all numbers to indicate a UID/GID, however it is possible on some systems (POSIX) to have usernames that are all numerical. The user and group values will first be resolved using the image's own `/etc/passwd` or `/etc/group`. If no valid matches are found, then if the string is all numerical, it shall be converted to an integer and used as the UID/GID. If the user or group field begins with a "/", the owner and group of the file found at that absolute path inside the rootfs is used as the UID/GID of the process. Example values for the fields include `root`, `1000`, or `/usr/bin/ping`.
    * **supplementaryGIDs** (list of unsigned integers, optional) indicates additional (supplementary) group IDs (GIDs) as which the app's processes should run.
    * **eventHandlers** (list of objects, optional) allows the app to have several hooks based on lifecycle events. For example, you may want to execute a script before the main process starts up to download a dataset or backup onto the filesystem. An eventHandler is a simple object with two fields - an **exec** (array of strings, ACE can append or override), and a **name** (there may be only one eventHandler of a given name), which must be one of:
        * **pre-start** - executed and must exit before the long running main **exec** binary is launched
        * **post-stop** - executed if the main **exec** process is killed. This can be used to cleanup resources in the case of clean application shutdown, but cannot be relied upon in the face of machine failure.
    * **workingDirectory** (string, optional) working directory of the launched application, relative to the application image's root (must be an absolute path, defaults to "/", ACE can override). If the directory does not exist in the application's assembled rootfs (including any dependent images and mounted volumes), the ACE must fail execution.
    * **environment** (list of objects, optional) represents the app's environment variables (ACE can append). The listed objects must have two key-value pairs: **name** and **value**. The **name** must consist solely of letters, digits, and underscores '_' as outlined in [IEEE Std 1003.1-2001](http://pubs.opengroup.org/onlinepubs/009695399/basedefs/xbd_chap08.html). The **value** is an arbitrary string. These values are not evaluated in any way, and no substitutions are made.
    * **isolators** (list of objects of type [Isolator](types.md#isolator-type), optional) list of isolation steps that SHOULD be applied to the app.
    * **mountPoints** (list of objects, optional) locations where an app is expecting external data to be mounted. The listed objects contain the following key-value pairs: the **name** indicates a label to refer to a mount point (which may be used by the executor when resolving a mount to a volume in the PodManifest), and the **path** stipulates where it is to be mounted inside the rootfs. The name is restricted to the [AC Name](types.md#ac-name-type) Type formatting. **readOnly** is a boolean indicating whether or not the mount point will be read-only (defaults to "false" if unsupplied).
    * **ports** (list of objects, optional) ports that this app will be listening on once started. This field is informational: example uses include helping users to discover the listening ports of the application, or indicating to executors ports that should be exposed on the host. This information could also optionally be used to limit the inbound connections to the container via firewall rules to only ports that are explicitly exposed. Each object can represent either a single port or a port range (contiguous set of ports).
        * **name** (string, required, restricted to the [AC Name](#ac-name-type) formatting) descriptive name for this port; for example, "http" or "database". This field is used as a key in the [Pod Manifest](pods.md#pod-manifest-schema) when specifying ports to be forwarded from the host.
        * **protocol** (string, required) protocol that will be used on this port. MAY be any value, but typically SHOULD be a transport layer (Layer 4) protocol - for example, "tcp" or "udp". The executor MAY refuse to execute if this field contains an unrecognized value.
        * **port** (integer, required) port number that will be used; see also **count**. Must be >=1 and <=65535.
        * **count** (integer, optional, defaults to 1 if unset) specifies a range of ports, starting with "port" and ending with "port" + "count" - 1. Must be >=1.
        * **socketActivated** (boolean, optional, defaults to "false" if unsupplied) if set to true, the application expects to be [socket activated](http://www.freedesktop.org/software/systemd/man/sd_listen_fds.html) on these ports. To perform socket activation, the ACE MUST pass file descriptors using the [socket activation protocol](http://www.freedesktop.org/software/systemd/man/sd_listen_fds.html) that are listening on these ports when starting this app. If multiple apps in the same pod are using socket activation then the ACE must match the sockets to the correct apps using getsockopt() and getsockname().
	* **userAnnotations** (object, optional) map of arbitrary key-value data for end-user use. All values in the object must be strings. Unlike annotations, user annotations allow arbitrary strings for keys, and MUST NOT affect ACE runtime behavior.
	* **userLabels** (object, optional) map of arbitrary key-value data for end-user use. All values in the object must be strings. User labels MUST NOT affect ACE runtime behavior.
* **dependencies** (list of objects, optional) dependent application images that need to be placed down into the rootfs before the files from this image (if any). The ordering is significant. See [Dependency Matching](#dependency-matching) for how dependencies are retrieved.
    * **imageName** (string of type [AC Identifier](types.md#ac-identifier-type), required) name of the dependent App Container Image.
    * **imageID** (string of type [Image ID](types.md#image-id-type), optional) content hash of the dependency. If provided, the retrieved dependency must match the hash. This can be used to produce deterministic, repeatable builds of an App Container Image that has dependencies.
    * **labels** (list of objects, optional) a list of the very same form as the aforementioned label objects in the top level ImageManifest. See [Dependency Matching](#dependency-matching) for how these are used.
    * **size** (integer, optional) the size of the image referenced dependency, in bytes. This field is optional; if it is present, the ACE SHOULD ensure it matches when retrieving a dependency, to mitigate "endless data" attacks.
* **pathWhitelist** (list of strings, optional) whitelist of absolute paths that will exist in the app's rootfs after rendering. This must be a complete and absolute set. An empty list is equivalent to an absent value and means that all files in this image and any dependencies will be available in the rootfs.
* **annotations** (list of objects, optional) any extra metadata you wish to add to the image. Each object has two key-value pairs: the *name* is restricted to the [AC Identifier](types.md#ac-identifier-type) formatting and *value* is an arbitrary string. Annotation names must be unique within the list. Annotations MAY be used by systems outside of the ACE, and MAY affect ACE behaviour (for example, particular runtime extensions). An ACE MAY override annotations. If you are defining new annotations, please consider submitting them to the specification. If you intend for your field to remain special to your application please be a good citizen and prefix an appropriate namespace to your key names. Recognized annotations include:
    * **created** date on which the image was built (string, [timestamps type](types.md#timestamps-type))
    * **authors** contact details of the people or organization responsible for the image (freeform string)
    * **homepage** URL to find more information on the image (string, must be a URL with scheme HTTP or HTTPS)
    * **documentation** URL to get documentation on the image (string, must be a URL with scheme HTTP or HTTPS)
    * **appc.io/executor/supports-systemd-notify** (boolean, optional, defaults to "false" if unset) if set to true, the application SHOULD use the sd\_notify mechanism to signal when it is ready. Also it SHOULD be able to detect if the executor had not set up the sd\_notify mechanism and skip the notification without error ([sd_notify()](https://www.freedesktop.org/software/systemd/man/sd_notify.html) from libsystemd does that automatically).

#### Dependency Matching

Dependency matching is based on a combination of the three different fields of the dependency - **imageName**, **imageID**, and **labels**.
First, the image discovery mechanism is used to locate a dependency based on the **imageName** and **labels** (see [App Container Image Discovery](discovery.md)).

If the image discovery process successfully returns an image and the dependency specification has an image ID, it will be compared against the hash of image returned, and MUST match.

This facilitates "wildcard" matching and a variety of common usage patterns, like "noarch" or "latest" dependencies.
For example, an ACI containing a set of bash scripts might omit both "os" and "arch", and hence could be used as a dependency by a variety of different ACIs.
Alternatively, an ACI might specify a dependency with no image ID and no "version" label, and the image discovery mechanism could always retrieve the latest version of an ACI.
