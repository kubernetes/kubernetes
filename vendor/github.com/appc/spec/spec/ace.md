## App Container Executor

The **App Container Executor** defines the process by which applications contained in ACIs are executed.
There are two "perspectives" in this process.
The "*executor*" perspective consists of the steps that the App Container Executor (ACE) must take to set up the environment for the pod and applications.
The "*app*" perspective is how the app processes inside the pod see the environment.

This example pod will use a set of three apps:

| Name                               | Version | Image hash                                      |
|------------------------------------|---------|-------------------------------------------------|
| example.com/reduce-worker          | 1.0.0   | sha512-277205b3ae3eb3a8e042a62ae46934b470e43... |
| example.com/worker-backup          | 1.0.0   | sha512-3e86b59982e49066c5d813af1c2e2579cbf57... |
| example.com/reduce-worker-register | 1.0.0   | sha512-86298e1fdb95ec9a45b5935504e26ec29b8fe... |

#### Pod UUID

Each pod much be assigned an [RFC4122 UUID](http://www.ietf.org/rfc/rfc4122.txt). 
The UUID serves as a canonical reference to a pod within a given administrative domain.
In this context, an administrative domain is linked to the scope of the associated [Metadata Service](#app-container-metadata-service).
For example, given a metadata service that is federated across a geographical cluster of systems, the pod UUID is uniquely scoped to the same cluster.
This UUID is exposed to the pod through the [Metadata Service](#app-container-metadata-service).

#### Filesystem Setup

Each app in a pod will start chrooted into its own unique read-write filesystem before execution.

An app's filesystem must be *rendered* in an empty directory by the following process (or equivalent):

1. If the ACI contains a non-empty `dependencies` field in its `ImageManifest`, the `rootfs` of each dependent image is extracted into the target directory, in the order in which they are listed.
2. The `rootfs` contained in the ACI is extracted into the target directory
3. If the ACI contains a non-empty `pathWhitelist` field in its `ImageManifest`, *all* paths not in the whitelist must be removed from the target directory

If during rootfs extraction a path is already present in the target directory from an earlier dependency, the previously extracted path MUST be overwritten. If the existing path is a symbolic link to a directory, the link MUST NOT be followed and it MUST be removed and replaced with the new path.

Every execution of an app MUST start from a clean copy of this rendered filesystem.

The simplest implementation will take an ACI (with no dependencies) and extract it into a new directory:

```bash
cd $(mktemp -d -t temp.XXXX)
mkdir hello
tar xzvf /var/lib/pce/hello.aci -C hello
```

Other implementations could increase performance and de-duplicate data by building on top of overlay filesystems, copy-on-write block devices, or a content-addressed file store.
These details are orthogonal to the runtime environment.

#### Volume Setup

Volumes that are specified in the Pod Manifest are mounted into each of the apps via a bind mount (or equivalent).
For example, say that the worker-backup and reduce-worker both have a `mountPoint` named "work".
In this case, the executor will bind mount the host's `/opt/tenant1/work` directory into the `path` of each of the matching "work" `mountPoint`s of the two app filesystems.

If the target `path` does not exist in the rendered filesystem, it SHOULD be created, including any missing parent directories.

If the target `path` is a non-empty directory, its contents SHOULD be discarded (e.g. obscured by the bind mount).
If the target `path` refers to a file, the ACE SHOULD remove that file and create a directory in its place.
In either of these cases, the ACE SHOULD warn the user that existing files are being masked by a volume.

If multiple targets have overlapping target `path`s (for example, if one is nested within another), the ACE SHOULD consider this an error.

The target `path` directories that the ACE creates SHOULD be owned by UID 0 and GID 0, and have access mode `0755` (`rwxr-xr-xr-x`).
The ACE implementation MAY provide a method for administrator to specify different permissions on a per-pod basis.

The ACE SHOULD NOT create any paths in the host file system, and MUST consider missing volume source paths an error.
If the ACE does modify the host file system, it SHOULD be possible to disable this behaviour.
The ACE MAY implement single-file volumes if the underlying operating system supports it.

If the host volume's `source` path is a symbolic link, the ACE SHOULD consider it an error, and SHOULD NOT attempt to use this link's target as volume.
The ACE SHOULD also consider it an error if any intermediate directory in volume's `source` path is a symbolic link.
If the ACE chooses to support symbolic links as volume sources, it SHOULD provide a way to enable or disable this behaviour on a per-pod basis (e.g. as a boolean isolator or a command line switch).

#### Network Setup

A Pod must have a loopback network interface and zero or more [layer 3](http://en.wikipedia.org/wiki/Network_layer) (commonly called the IP layer) network interfaces, which can be instantiated in any number of ways (e.g. veth, macvlan, ipvlan, device pass-through).
Each network interface MUST be configured with one or more IPv4 and/or IPv6 addresses.

#### Logging

Apps SHOULD log to stdout and stderr.  The ACE is responsible for capturing and persisting this output.

If the application detects other logging options, such as the `/run/systemd/system/journal` socket, it may optionally upgrade to using those mechanisms.
Note that logging mechanisms other than stdout and stderr are not required by this specification (and are not tested during compliancy verifications).

### Apps Perspective

#### Execution Environment

The following environment variables MUST be set for each application's main process and any lifecycle processes:

* **PATH** `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`
* **AC_APP_NAME** name of the application, as defined in the image manifest
* **AC_METADATA_URL** URL where the [metadata service](#app-container-metadata-service) for this pod can be found.
* **container** name of the [App Container Executor](#app-container-executor) as a concise indicator the environment is in a container (free-form string)

An executor MAY set additional environment variables for the application processes.

Additionally, processes must have their **working directory** set to the value of the application's **workingDirectory** option, if specified, or the root of the application image by default.

### Isolators

Isolators enforce resource constraints rather than namespacing.
Isolators may be scoped to individual applications, to whole pods, or to both.
Any isolators applied to the pod will _bound_ any individual isolators applied to applications within the pod.

Some well known isolators can be verified by the specification.
Additional isolators will be added to this specification over time.

An executor MAY ignore isolators that it does not understand and run the pod without them.
But, an executor MUST make information about which isolators were ignored, enforced or modified available to the user.
An executor MAY implement a "strict mode" where an image cannot run unless all isolators are in place.

### Linux Isolators

These isolators are specific to the Linux kernel and are impossible to represent as a 1-to-1 mapping on other kernels.
The first example is "capabilities" but this will be expanded to include things such as SELinux, SMACK or AppArmor.

#### os/linux/capabilities-remove-set

* Scope: app

**Parameters:**

* **set** list of capabilities that will be removed from the process's capabilities bounding set, all others will be included.

```json
"name": "os/linux/capabilities-remove-set",
"value": {
  "set": [
    "CAP_SYS_PTRACE"
  ]
}
```

#### os/linux/capabilities-retain-set

* Scope: app

**Parameters:**

* **set** list of capabilities that will be retained in the process's capabilities bounding set, all others will be removed

```json
"name": "os/linux/capabilities-retain-set",
"value": {
  "set": [
    "CAP_KILL",
    "CAP_CHOWN"
  ]
}
```

### Resource Isolators

A _resource_ is something that can be consumed by an application (app) or group of applications (pod), such as memory (RAM), CPU, and network bandwidth.
Resource isolators have a *request* and *limit* quantity:

- **request** is the minimum amount of a resource guaranteed to be available to the app/pod.
If the app/pod attempts to consume a resource in excess of its request, it may be throttled or denied.
If **request** is omitted, it defaults to the value of **limit**.

- **limit** is the maximum amount of a resource available to the app/pod.
If the app/pod consumes a resource in excess of its limit, it must be terminated or throttled to no more than the limit.

Limit and request quantities must always be represented internally (i.e. for encoding and any processing) as an integer value (i.e. NOT floating point) in a resource type's natural base units (e.g., bytes, not megabytes or gigabytes).
For convenience, when specified by users quantities may either be unsuffixed, have metric suffices (E, P, T, G, M, K) or binary (power-of-two) suffices (Ei, Pi, Ti, Gi, Mi, Ki).
For example, the following strings represent the same value: "128974848", "125952Ki", "123Mi".
Small quantities can be represented directly as decimals (e.g., 0.3), or using milli-units (e.g., "300m").

#### resource/block-bandwidth

* Scope: app/pod

**Parameters:**

* **default** must be set to true and means that this limit applies to all block devices by default
* **limit** read/write bytes per second

```json
"name": "resource/block-bandwidth",
"value": {
  "default": true,
  "limit": "2M"
}
```

#### resource/block-iops

* Scope: app/pod

**Parameters:**

* **default** must be set to true and means that this limit applies to all block devices by default
* **limit** read/write input/output operations per second

```json
"name": "resource/block-iops",
"value": {
  "default": true,
  "limit": "1000"
}
```

#### resource/cpu

* Scope: app/pod

**Parameters:**

* **request** cores that are requested
* **limit** cores that can be consumed before the kernel temporarily throttles the process

```json
"name": "resource/cpu",
"value": {
  "request": "250m",
  "limit": "500m"
}
```

**Note**: a core is the seconds/second that the app/pod will be able to run. e.g. 1 (or 1000m for 1000 milli-seconds) would represent full use of a single CPU core every second.

#### resource/memory

* Scope: app/pod

**Parameters:**

* **request** bytes of memory that the app/pod is requesting to use and allocations over this request will be reclaimed in case of contention
* **limit** bytes of memory that the app can allocate before the kernel considers the app/pod out of memory and stops allowing allocations.

```json
"name": "resource/memory",
"value": {
  "request": "1G",
  "limit": "2G"
}
```

#### resource/network-bandwidth

* Scope: app/pod

**Parameters:**

* **default** must be set to true and means that this bandwidth limit applies to all interfaces (except localhost) by default.
* **limit** read/write bytes per second

```json
"name": "resource/network-bandwidth",
"value": {
  "default": true,
  "limit": "1G"
}
```

**NOTE**: Network limits MUST NOT apply to localhost communication between apps in a pod.


## App Container Metadata Service

For a variety of reasons, it is desirable to not write files to the filesystem in order to run an App Container:
* Secrets can be kept outside of the app
* The app can be run on top of a cryptographically secured read-only filesystem
* Metadata is a proven system for virtual machines

The App Container specification defines an HTTP-based metadata service for providing metadata to applications, as well as an [identity endpoint](#identity-endpoint).

### Metadata Service

The ACE SHOULD provide a Metadata service on the address given to the applications via the `AC_METADATA_URL` [environment variable](#execution-environment).
This URL must reference an endpoint which conforms to the HTTP protocol with TLS optionally enabled.

ACE implementations SHOULD embed an authorization token in `AC_METADATA_URL`, which provides a means for the metadata service to uniquely and securely identify a pod.
For example, `AC_METADATA_URL` passed to a pod could be set to `https://10.0.0.1:8888/Y4vFeVZzKM2T9rwkpWHfqXuGsNjS6O5c` with the path portion acting as a token.
Since the token is used by the Metadata Service to authenticate the pod's identity, it SHOULD have no fewer than 128 bits of entropy (i.e. size of UUID), and SHOULD NOT be easily guessable (e.g. the pod UUID should not be used).

For the following endpoints, unless otherwise specified, the media type of `application/json` must be specified and the body must conform to [RFC4627](http://www.ietf.org/rfc/rfc4627.txt).

[UUIDs](#pod-uuid) assigned to pods MUST be unique for the administrative domain of the metadata service.

### Pod Metadata

Information about the pod that this app is executing in.

Retrievable at `$AC_METADATA_URL/acMetadata/v1/pod`

| Entry       | Description |
|-------------|-------------|
|annotations | Top level annotations from Pod Manifest. Response body should conform to the sub-schema of the annotations property from the Pod specification (e.g. ```[ { "name": "ip-address", "value": "10.1.2.3" } ]```). |
|manifest     | Fully-reified Pod Manifest JSON. |
|uuid         | Pod UUID. The metadata service must return the `Content-Type` of `text/plain; charset=us-ascii` and the body of the response must be the pod UUID in canonical form. |

### App Metadata

Every running process will be able to introspect its App Name via the `AC_APP_NAME` environment variable.
This is necessary to query for the correct endpoint metadata.

Retrievable at `$AC_METADATA_URL/acMetadata/v1/apps/$AC_APP_NAME/`

| Entry         | Description |
|---------------|-------------|
|annotations   | Annotations from Image Manifest merged with app annotations from Pod Manifest. Response body should conform to the sub-schema of the annotations property from the ACE and Pod specifications (e.g. ```[ { "name": "ip-address", "value": "10.1.2.3" } ]```). |
|image/manifest | Original Image Manifest of the app. |
|image/id       | Image ID (digest) this app is contained in. The metadata service must return the `Content-Type` of `text/plain; charset=us-ascii` and the body of the response must be the image ID as described in the ACI specification.|

### Identity Endpoint

As a basic building block for building a secure identity system, the metadata service must provide an HMAC (described in [RFC2104](https://www.ietf.org/rfc/rfc2104.txt)) endpoint for use by the apps in the pod.
This gives a cryptographically verifiable identity to the pod based on its unique ID and the pod HMAC key, which is held securely by the ACE.

Accessible at `$AC_METADATA_URL/acMetadata/v1/pod/hmac`

| Entry | Description |
|-------|-------------|
|sign   | Client applications must POST a form with content=&lt;object to sign&gt;. The response must specify a `Content-Type` header of `text/plain; charset=us-ascii` and the body must be a base64 encoded hmac-sha512 signature based on an HMAC key maintained by the Metadata Service. |
|verify | Verify a signature from another pod. POST a form with content=&lt;object that was signed&gt;, uuid=&lt;uuid of the pod that generated the signature&gt;, signature=&lt;base64 encoded signature&gt;. Returns 200 OK if the signature passes and 403 Forbidden if the signature check fails. |
