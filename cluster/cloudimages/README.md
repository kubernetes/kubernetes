## Kubernetes-optimized images

This directory contains manifests for building Kubernetes-optimized images for
various clouds (currently just AWS).  It is currently highly experimental, and
these images are not used by default (though you can pass `AWS_IMAGE` to the
AWS kube-up script if you're feeling brave).

Advantages of an optimized image:

* We can preinstall packages that would otherwise require a download.  Great
  for speed, and also for reliability (in case the source repository is down)
* We can make kernel configuration changes that might otherwise require a
  reboot, or even apply kernel patches if we really want to.  For example,
  Debian requires a kernel boot parameter to enable the cgroup memory
  controller, which we require.
* The more configuration we can do in advance, the easier it is for people that
  don't want to use kube-up to get a cluster up and running.

Advantages of a harmonized image:

* All the platforms can test with the same versions of software, rather than
  relying on whatever image happens to be optimal on that cloud.

## bootstrap-vz

Currently images are built using
[bootstrap-vz](https://github.com/andsens/bootstrap-vz), because this is
default builder for the official Debian images, and because it supports
multiple clouds including AWS, Azure & GCE.  It also supports KVM, which should
support OpenStack.

## Building an image

A go program/script to build images in
[in progress](https://github.com/kubernetes/contrib/pull/486), in the contrib
project.



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/cloudimages/README.md?pixel)]()
