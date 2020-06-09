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
For details, please see COS's [Release Channels](https://cloud.google.com/container-optimized-os/docs/concepts/release-channels)and [Support
Policy](https://cloud.google.com/container-optimized-os/docs/resources/support-policy).

## COS in End-to-End tests

Container-Optimized OS images are used by kubernetes [End-to-End tests](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-testing/e2e-tests.md) and
[Node End-to-End tests](https://github.com/kubernetes/community/tree/master/contributors/devel/sig-node). To see current
active releases, please refer to COS's [Release
Notes](https://cloud.google.com/container-optimized-os/docs/release-notes). When
choose images for testing, the latest LTS images is suggested for running
presubmit, postsubmit or periodics tests since the image is stable and has
latest bug and security fixes. For testing requiring new features or continuous integration,
the latest dev or beta or stable channel can be used.
are needed, the latest LTS images or stables are preferred.

To find the current COS image, use the following command:
```shell
$ gcloud compute images list --project=cos-cloud |grep cos-cloud
cos-69-10895-385-0                                    cos-cloud          cos-69-lts                                    READY
cos-73-11647-534-0                                    cos-cloud          cos-73-lts                                    READY
cos-77-12371-274-0                                    cos-cloud          cos-77-lts                                    READY
cos-81-12871-119-0                                    cos-cloud          cos-81-lts                                    READY
cos-beta-81-12871-117-0                               cos-cloud          cos-beta                                      READY
cos-dev-84-13078-0-0                                  cos-cloud          cos-dev                                       READY
cos-stable-81-12871-119-0                             cos-cloud          cos-stable                                    READY
```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/gce/gci/README.md?pixel)]()
