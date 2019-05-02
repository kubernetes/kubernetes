# Kubernetes test images

## Overview

All the images found here are used in Kubernetes tests that ensure its features and functionality.
The images are built and published as manifest lists, allowing multiarch and cross platform support.

This guide will provide information on how to: make changes to images, bump their version, build the
new images, test the changes made, promote the newly built staging images.


## Prerequisites

In order to build the docker test images, a Linux node is required. The node will require `make`
and `docker (version 18.06.0 or newer)`. Manifest lists were introduced in 18.03.0, but 18.06.0
is recommended in order to avoid certain issues.

The node must be able to push the images to the desired container registry, make sure you are
authenticated with the registry you're pushing to.

Windows Container images are not built by default, since they cannot be built on Linux. For
that, a Windows node with Docker installed and configured for remote management is required.


### Windows node(s) setup

In order to build the Windows container images, a node with Windows 10 or Windows Server 2019
with the latest updates installed is required. The node will have to have Docker installed,
preferably version 18.06.0 or newer.

Keep in mind that the Windows node might not be able to build container images for newer OS versions
than itself (even with `--isolation=hyperv`), so keeping the node up to date and / or upgrading it
to the latest Windows Server edition is ideal.

Windows test images must be built for Windows Server 2019 (1809) and Windows Server 1903, thus,
if the node does not have Hyper-V enabled, or it is not supported, multiple Windows nodes are required,
one per OS version.

Additionally, remote management must be configured for the node's Docker daemon. Exposing the
Docker daemon without requiring any authentication is not recommended, and thus, it must be
configured with TLS to ensure that only authorised people can interact with it. For this, the
following `powershell` script can be executed:

```powershell
mkdir .docker
docker run --isolation=hyperv --user=ContainerAdministrator --rm `
  -e SERVER_NAME=$(hostname) `
  -e IP_ADDRESSES=127.0.0.1,YOUR_WINDOWS_BUILD_NODE_IP `
  -v "c:\programdata\docker:c:\programdata\docker" `
  -v "$env:USERPROFILE\.docker:c:\users\containeradministrator\.docker" stefanscherer/dockertls-windows:2.5.5
# restart the Docker daemon.
Restart-Service docker
```

For more information about the above commands, you can check [here](https://hub.docker.com/r/stefanscherer/dockertls-windows/).

A firewall rule to allow connections to the Docker daemon is necessary:

```powershell
New-NetFirewallRule -DisplayName 'Docker SSL Inbound' -Profile @('Domain', 'Public', 'Private') -Direction Inbound -Action Allow -Protocol TCP -LocalPort 2376
```

If your Windows build node is hosted by a cloud provider, make sure the port `2376` is open for the node.
For example, in Azure, this is done by running the following command:

```console
az vm open-port -g GROUP-NAME -n NODE-NAME --port 2376
```

The `ca.pem`, `cert.pem`, and `key.pem` files that can be found in `$env:USERPROFILE\.docker`
will have to copied to the `~/.docker-${os_version)/` on the Linux build node, where `${os_version}`
is `1809` or `1903`.

```powershell
scp.exe -r $env:USERPROFILE\.docker ubuntu@YOUR_LINUX_BUILD_NODE:/home/ubuntu/.docker-$os_version
```

After all this, the Linux build node should be able to connect to the Windows build node:

```bash
docker --tlsverify --tlscacert ~/.docker-${os_version}/ca.pem --tlscert ~/.docker-${os_version}/cert.pem --tlskey ~/.docker-${os_version}/key.pem -H "$REMOTE_DOCKER_URL" version
```

For more information and troubleshooting about enabling Docker remote management, see
[here](https://docs.microsoft.com/en-us/virtualization/windowscontainers/management/manage_remotehost)

Finally, the node must be able to push the images to the desired container registry, make sure you are
authenticated with the registry you're pushing to.


## Making changes to images

There are several thousands of tests in Kubernetes E2E testing. Not all of them are being run on
new PRs, and thus, not all images are used, especially those that are not used by Conformance tests.

So, in order to prevent regressions in the images and failing jobs, any changes made to the image
itself or its binaries will require the image's version to be bumped. In the case of a regression
which cannot be immediately resolved, the image version used in E2E tests will be reverted to the
last known stable version.

The version can easily be bumped by modifying the file `test/images/${IMAGE_NAME}/VERSION`, which will
be used when building the image. Additionally, for the `agnhost` image, also bump the `Version` in
`test/images/agnhost/agnhost.go`.

The typical image used in E2E testing is the `agnhost` image. It contains several subcommands with
different [functionalities](agnhost/README.md), used to validate different Kubernetes behaviours. If
a new functionality needs testing, consider adding an `agnhost` subcommand for it first, before
creating an entirely separate test image.

Some test images (`agnhost`) are used as bases for other images (`kitten`, `nautilus`). If the parent
image's `VERSION` has been bumped, also bump the version in the children's `BASEIMAGE` files in order
for base image changes to be reflected in the child images as well.

Keep in mind that the Kubernetes CI will not run with the image changes you've made. It is a good idea
to build the image and push it to your own registry first, and run some tests that are using that image.
For these steps, see the sections below.

After the desired changes have been made, the affected images will have to be built and published,
and then tested. After the pull request with those changes has been approved, the new images will be
built and published to the `gcr.io/kubernetes-e2e-test-images` registry as well.

Currently, the image building process has been automated with the Image Promoter, but *only* for the
Conformance images (`agnhost`, `jessie-dnsutils`, `kitten`, `nautilus`, `nonewprivs`, `resource-consumer`,
`sample-apiserver`).  After the pull request merges, a postsubmit job will be started with the new changes,
which can be tracked [here](https://testgrid.k8s.io/sig-testing-images#post-kubernetes-push-images).
After it passes successfully, the new image will reside in the `gcr.io/k8s-staging-e2e-test-images/${IMAGE_NAME}:${VERSION}`
registry, from which it will have to be promoted by adding a line for it
[here](https://github.com/kubernetes/k8s.io/blob/master/k8s.gcr.io/images/k8s-staging-e2e-test-images/images.yaml).
For this, you will need the image manifest list's digest, which can be obtained by running:

```bash
manifest-tool inspect --raw gcr.io/k8s-staging-e2e-test-images/${IMAGE_NAME}:${VERSION} | jq '.[0].Digest'
```

The images are built through `make`. Since some images (e.g.: `busybox`) are used as a base for
other images, it is recommended to build them first, if needed.


## Building images

The images are built through `make`. Since some images (`agnhost`) are used as a base for other images,
it is recommended to build them first, if needed.

An image can be built by simply running the command:

```bash
make all WHAT=agnhost
```

To build AND push an image, the following command can be used:

```bash
make all-push WHAT=agnhost
```

By default, the images will be tagged and pushed under the `gcr.io/kubernetes-e2e-test-images`
registry. That can changed by running this command instead:

```bash
REGISTRY=foo_registry make all-push WHAT=agnhost
```

In order to also include Windows Container images into the final manifest lists, the `REMOTE_DOCKER_URL` argument
in the form `tcp://[host]:[port][path]` (for more details, see [here]([https://docs.docker.com/engine/reference/commandline/dockerd/#daemon-socket-option]/))
will also have to be specified:

```bash
REMOTE_DOCKER_URL_1909=remote_docker_url_1909 REMOTE_DOCKER_URL_1903=remote_docker_url_1903 REMOTE_DOCKER_URL_1809=remote_docker_url_1809 REGISTRY=foo_registry make all-push WHAT=test-webserver
```

*NOTE* (for test `gcr.io` image publishers): Some tests (e.g.: `should serve a basic image on each replica with a private image`)
require the `agnhost` image to be published in an authenticated repo as well:

```bash
REGISTRY=gcr.io/kubernetes-e2e-test-images make all-push WHAT=agnhost
REGISTRY=gcr.io/k8s-authenticated-test make all-push WHAT=agnhost
```


## Testing the new image

Once the image has been built and pushed to an accesible registry, you can run the tests using that image
by having the environment variable `KUBE_TEST_REPO_LIST` set before running the tests that are using the
image:

```bash
export KUBE_TEST_REPO_LIST=/path/to/repo_list.yaml
```

`repo_list.yaml` is a configuration file used by the E2E tests, in which you can set alternative registries
to pull the images from. Sample file:

```yaml
dockerLibraryRegistry: your-awesome-registry
e2eRegistry: your-awesome-registry
gcRegistry: your-awesome-registry
sampleRegistry: your-awesome-registry
```

Keep in mind that some tests are using multiple images, so it is a good idea to also build and push those images.

Finally, make sure to bump the image version used in E2E testing by modifying the file `test/utils/image/manifest.go`, and recompile afterwards:

```bash
./build/run.sh make WHAT=test/e2e/e2e.test
```

After all the above has been done, run the desired tests.


## Known issues and workarounds

`docker manifest create` fails due to permission denied on `/etc/docker/certs.d/gcr.io` (https://github.com/docker/for-linux/issues/396). This issue can be resolved by running:

```bash
sudo chmod o+x /etc/docker
```

`nc` is being used by some E2E tests, which is why we are including a Linux-like `nc.exe` into the Windows `busybox` image. The image could fail to build during that step with an error that looks like this:

```console
re-exec error: exit status 1: output: time="..." level=error msg="hcsshim::ImportLayer failed in Win32: The system cannot find the path specified. (0x3) path=\\\\?\\C:\\ProgramData\\...
```

The issue is caused by the Windows Defender which is removing the `nc.exe` binary from the filesystem. For more details on this issue, see [here](https://github.com/diegocr/netcat/issues/6). To fix this, you can simply run the following powershell command to temporarily disable Windows Defender:

```powershell
Set-MpPreference -DisableRealtimeMonitoring $true
```
