# Kubernetes test images

## Overview

All the images found here are used in Kubernetes tests that ensure its features and functionality.
The images are built and published as manifest lists, allowing multiarch and cross platform support.

This guide will provide information on how to: make changes to images, bump their version, build the
new images, test the changes made.


## Prerequisites

In order to build the docker test images, a Linux node is required. The node will require `make`
and `docker (version 18.06.0 or newer)`. Manifest lists were introduced in 18.03.0, but 18.06.0
is recommended in order to avoid certain issues.

The node must be able to push the images to the desired container registry, make sure you are
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

Some test images (`mounttest`, `test-webserver`) are used as bases for other images (`mounttest-user`,
`kitten`, `nautilus`). If the parent image's `VERSION` has been bumped, also bump the version in the
children's `BASEIMAGE` files in order for base image changes to be reflected in the child images as well.

TODO: Once [Centralization part 4](https://github.com/kubernetes/kubernetes/pull/81226) merges, the paragraph
above will have to be updated, as those images will be included into `agnhost`.

After the desired changes have been made, the affected images will have to be built and published,
and then tested. After the pull request with those changes has been approved, the new images will be
built and published to the `gcr.io/kubernetes-e2e-test-images` registry as well.


## Building images

The images are built through `make`. Since some images (`mounttest`, `test-webserver`)
are used as a base for other images, it is recommended to build them first, if needed.

TODO: Once [Centralization part 4](https://github.com/kubernetes/kubernetes/pull/81226) merges, the paragraph
above will have to be updated, as those images will be included into `agnhost`.

An image can be built by simply running the command:

```bash
make all WHAT=test-webserver
```

To build AND push an image, the following command can be used:

```bash
make all-push WHAT=test-webserver
```

By default, the images will be tagged and pushed under the `gcr.io/kubernetes-e2e-test-images`
registry. That can changed by running this command instead:

```bash
REGISTRY=foo_registry make all-push WHAT=test-webserver
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
