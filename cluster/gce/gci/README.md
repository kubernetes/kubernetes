# Container-Optimized OS

[Container-Optimized OS](https://cloud.google.com/container-optimized-os/docs),
(previously Google Container-VM image a.k.a GCI) is a container-optimized OS image for the Google Cloud Platform (GCP). It is
primarily for running Google services on GCP. Container-Optimized OS is an open
source OS based on
the open source [ChromiumOS project](https://www.chromium.org/chromium-os), allowing us greater control over the build management,
security compliance, and customizations for GCP.

Container-Optimized OS is [open source](https://cos.googlesource.com), and is released on milestones. Example milestones are
81, 85. Each milestone will experience three release channels -- dev, beta and stable to reach
stability. The promotion between those channels are about six weeks.
Starting milestone 69, for
every 4 milestones, the last milestone will be promoted into LTS image after it
becomes stable.
For details, please see COS's [Release Channels](https://cloud.google.com/container-optimized-os/docs/concepts/release-channels) and [Support
Policy](https://cloud.google.com/container-optimized-os/docs/resources/support-policy).

## COS in End-to-End tests

Container-Optimized OS images are used by kubernetes [End-to-End tests](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-testing/e2e-tests.md) and
[Node End-to-End tests](https://github.com/kubernetes/community/tree/master/contributors/devel/sig-node). To see current
active releases, please refer to COS's [Release
Notes](https://cloud.google.com/container-optimized-os/docs/release-notes).

### How to choose an image in configuration file

There are three ways to specify an image used by each testing suite: `image`,
`image_regex` or `image_family`.

  * `image` is preferred, but manual updates are needed to use a newly released
    COS image, so the test suites don't use deprecated images. This will result
    to frequent yaml configuration file update everytime COS releases new
    image.One future option is to use an autobumper robot to update COS image
    automatically. e.g:
```
  cos-stable:
    image: cos-77-12371-274-0
    project: cos-cloud
    metadata: "user-data</go/src/github.com/containerd/cri/test/e2e_node/init.yaml,containerd-configure-sh</go/src/github.com/containerd/cri/cluster/gce/configure.sh,containerd-extra-init-sh</go/src/github.com/containerd/cri/test/e2e_node/gci-init.sh,containerd-env</workspace/test-infra/jobs/e2e_node/containerd/cri-master/env,gci-update-strategy=update_disabled"
```

  * `image_family` should be used if you always want to use latest image in the
    same family. Tests will start to use new images once COS releases
    new image. This is not predictable and test can potentially be broken because of this. The probability of a
    breakage due to the OS itself is low for LTS or stable image, but high for dev or beta image.
    If things went wrong, it will be hard to rollback
    images using `image_regex` and `image_family`. e.g:
```
  cos-stable:
    image_family: cos-77-lts
    project: cos-cloud
    metadata: "user-data</workspace/test-infra/jobs/e2e_node/containerd/init.yaml,cni-template</workspace/test-infra/jobs/e2e_node/containerd/cni.template,containerd-config</workspace/test-infra/jobs/e2e_node/containerd/config.toml"
```

  * `image_regex` can also
    be used if you want image with the same naming pattern. Latest image will be
    chosen when multiple images match the regular expression. However, this
    option is rarely seen in the test code.

  * To update the images, using image in the same channel is preferred. Keep in
    mind, 69 is the first LTS image. Before that, COS only has dev, beta and stable
    images. That is why stable images are used quite frequently in current testing.
    For now, images should slowly migrate from stable to LTS if possible. For
    testing using dev or beta, we need to consider the original intention and
    keep using image in existing channel unless we understand the underlying reason.

### What image is needed for your test

Consider the importance of tests and the stability of Container-Optimized OS, the
following guidelines are proposed for image choice in E2E testing.

  * To run release blocking tests, the latest LTS images are preferred.
    'image' should be used to specify the image.

  * To run presubmit, postsubmit or periodic tests, the latest LTS images are
    preferred. If tests need two images, you can use the latest two LTS images.
    LTS images are stable and usually include latest bug and security fix.
    'image' should be used to specify the image.

  * To integrate continuously with other container
    related technologies like runc, containerd, docker and kubernetes, the
    latest LTS or stable images are preferred. 'image_family' should be used to
    specify the image.

  * To try out latest COS features, the latest dev or beta or stable images are preferred.
    'image' or 'image_family' should be used to specify the image.

### How to find current COS image in each channel

To find the current COS image, use the following command:

```shell
$ gcloud compute images list --project=cos-cloud | grep cos-cloud
cos-69-10895-385-0                                    cos-cloud          cos-69-lts                                    READY
cos-73-11647-534-0                                    cos-cloud          cos-73-lts                                    READY
cos-77-12371-274-0                                    cos-cloud          cos-77-lts                                    READY
cos-81-12871-119-0                                    cos-cloud          cos-81-lts                                    READY
cos-beta-81-12871-117-0                               cos-cloud          cos-beta                                      READY
cos-dev-84-13078-0-0                                  cos-cloud          cos-dev                                       READY
cos-stable-81-12871-119-0                             cos-cloud          cos-stable                                    READY
```

COS image will experience dev, beta, stable and LTS stage. Before LTS stage, image is named with its
family as a prefix, e.g cos-dev, cos-beta, cos-stable. However, the milestone
number in those families may change when channel promotions happen. Only when a milestone becomes LTS, the
image will have a new family, and the milestone number in the image name stays the same. The image
will be always there even after the milestone is deprecated.
