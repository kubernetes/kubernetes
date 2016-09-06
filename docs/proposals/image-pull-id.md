<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.4/docs/proposals/image-provenance.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Image Identification
====================

Currently to identify the image for a running container, Kubelet sets the
`ImageID` field in the container status information.  For Docker, the
`ImageID` field contains the ID of the image as returned by Docker on the
system.  This value is the actually the digest of the config blob for the
image.

Unfortunately, the Docker registry does not use this digest to refer to
the image.  Instead, it uses the digest of the image *manifest*.
Therefore, the image ID alone cannot be used to easily associate the image
from a running container with a particular image on the registry.  Being
able to determine and pull the image used by a particular container is
useful for auditing tooling (e.g. warn if I'm using "known bad" images on
my cluster), as well as general administration.  Image tags cannot be used
for this purpose, since tags are mutable.

Container Status Changes
------------------------

In order to make it possible to discover the image for a running
container, a new field, `CanonicalImageID` would be introduced.
This would contain an absolute, immutable reference to the image for
a particular running container.  In the case of Docker containers, this
would be the image manifest ID.  The existing `ImageID` field remains in
place to preserve backwards compatibility.

Obtaining an the Immutable Image Pull Information
-------------------------------------------------

When pulling by digest, Docker currently populates the `RepoDigests`
field.  However, when pulling by tag, Docker 1.10 and 1.11 leave this
field empty (Docker 1.12 always attempts to populate the field:
docker/docker@d81ed3eb4c6c4e23956ac075a2df714b2249b50e). However, Docker
knows the image manifest digest when pulling the image either way -- it
simply does not appear to store the information in case of pulling by tag.

Thus, we have two paths for getting the appropriate information.

### Always Pull by Digest ###

One option would be to have Kubelet always pull by digest.  Kubelet would
contact the relevant Docker registry directly in order to resolve the
manifest digest for a given tag, and then pull that instead of pulling by
tag.  This would be mostly transparent to the user and admin (images would
show up in `docker images` by digest instead of tag).  The advantage of
this approach is that it is doable today without requiring changes to any
other projects.  The downside is that it means Kubelet would have to know
how to speak to registries directly instead of simply communicating with
the Docker daemon, and would have to gain equivalents to the registry
flags on the Docker daemon (`--insecure-registry`, etc).

### Wait for Docker 1.12 ###

This entails waiting for Docker 1.12, but does not require actually
changing Kubelet or adding much to Kubelet itself, so is thus simpler to
implement.

Potential short-term solutions
------------------------------

It may be desirable to preform a quick fix in the short term.  We could
simply populate the existing `ImageID` field, or an annoation, with the
canonical image ID when available (anyone with running Docker with the
aforementioned commit included).  Then, in the long-term, we can
transition to a solution which always populates the new field, regardless
of Docker version. 

Path Forward
------------

TBD
