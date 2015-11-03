<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/devel/how-to-doc.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Document Conventions
====================

Updated: 11/3/2015

*This document is oriented at users and developers who want to write documents for Kubernetes.*

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

  - [What Are Mungers?](#what-are-mungers)
  - [Table of Contents](#table-of-contents)
  - [Writing Examples](#writing-examples)
  - [Adding Links](#adding-links)
  - [Auto-added Mungers](#auto-added-mungers)
    - [Unversioned Warning](#unversioned-warning)
    - [Is Versioned](#is-versioned)
    - [Generate Analytics](#generate-analytics)

<!-- END MUNGE: GENERATED_TOC -->

## What Are Mungers?

Mungers are like gofmt for md docs which we use to format documents. To use it, simply place

```
<!-- BEGIN MUNGE: xxxx -->
<!-- END MUNGE: xxxx -->
```

in your md files. Note that xxxx is the placeholder for a specific munger. Appropriate content will be generated and inserted between two brackets after you run `hack/update-generated-docs.sh`. See [munger document](../../cmd/mungedocs/) for more details.


## Table of Contents

Instead of writing table of contents by hand, use the TOC munger:

```
<!-- BEGIN MUNGE: GENERATED_TOC -->
<!-- END MUNGE: GENERATED_TOC -->
```

## Writing Examples

Sometimes you may want to show the content of certain example files. Use EXAMPLE munger whenever possible:

```
<!-- BEGIN MUNGE: EXAMPLE path/to/file -->
<!-- END MUNGE: EXAMPLE path/to/file -->
```

This way, you save the time to do the copy-and-paste; what's better, the content won't become out-of-date everytime you update the example file.

For example, the following munger:

```
<!-- BEGIN MUNGE: EXAMPLE ../user-guide/pod.yaml -->
<!-- END MUNGE: EXAMPLE ../user-guide/pod.yaml -->
```

generates
<!-- BEGIN MUNGE: EXAMPLE ../user-guide/pod.yaml -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
```

[Download example](../user-guide/pod.yaml?raw=true)
<!-- END MUNGE: EXAMPLE ../user-guide/pod.yaml -->

## Adding Links

Use inline link instead of url at all times. When you add internal links from `docs/` to `docs/` or `examples/`, use relative links; otherwise, use `http://releases.k8s.io/HEAD/<path/to/link>`. For example, use:

```
[GCE](../getting-started-guides/gce.md)                 # note that it's under docs/
[Kubernetes package](http://releases.k8s.io/HEAD/pkg/)  # note that it's under pkg/
[Kubernetes](http://kubernetes.io/)
```

and avoid using:

```
[GCE](https://github.com/kubernetes/kubernetes/blob/master/docs/getting-started-guides/gce.md)
[Kubernetes package](../../pkg/)
http://kubernetes.io/
```

## Auto-added Mungers

Some mungers are auto-added. You don't have to add them manually, and `hack/update-generated-docs.sh` does that for you. It's recommended to just read this section as a reference instead of messing up with the following mungers.

### Unversioned Warning

UNVERSIONED_WARNING munger inserts unversioned warning which warns the users when they're reading the document from HEAD and informs them where to find the corresponding document for a specific release.

```
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->
<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
```

### Is Versioned

IS_VERSIONED munger inserts `IS_VERSIONED` tag in documents in each release, which stops UNVERSIONED_WARNING munger from inserting warning messages.

```
<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->
```

### Generate Analytics

ANALYTICS munger inserts a Google Anaylytics link for this page.

```
<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
<!-- END MUNGE: GENERATED_ANALYTICS -->
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/how-to-doc.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
