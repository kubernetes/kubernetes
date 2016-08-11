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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Go Tools and Tips

Kubernetes is one of the largest open source Go projects, so good tooling a solid understanding of
Go is critical to Kubernetes development. This document provides a collection of resources, tools
and tips that our developers have found useful.

## Recommended Reading

- [Kubernetes Go development environment](development.md#go-development-environment)
- [The Go Spec](https://golang.org/ref/spec) - The Go Programming Language
  Specification.
- [Go Tour](https://tour.golang.org/welcome/2) - Official Go tutorial.
- [Effective Go](https://golang.org/doc/effective_go.html) - A good collection of Go advice.
- [Kubernetes Code conventions](coding-conventions.md) - Style guide for Kubernetes code.
- [Three Go Landmines](https://gist.github.com/lavalamp/4bd23295a9f32706a48f) - Surprising behavior in the Go language. These have caused real bugs!

## Recommended Tools

- [godep](https://github.com/tools/godep) - Used for Kubernetes dependency management. See also [Kubernetes godep and dependency management](development.md#godep-and-dependency-management)
- [Go Version Manager](https://github.com/moovweb/gvm) - A handy tool for managing Go versions.
- [godepq](https://github.com/google/godepq) - A tool for analyzing go import trees.

## Go Tips

- [Godoc bookmarklet](https://gist.github.com/timstclair/c891fb8aeb24d663026371d91dcdb3fc) - navigate from a github page to the corresponding godoc page.
- Consider making a separate Go tree for each project, which can make overlapping dependency management much easier. Remember to set the `$GOPATH` correctly! Consider [scripting](https://gist.github.com/timstclair/17ca792a20e0d83b06dddef7d77b1ea0) this.
- Emacs users - setup [go-mode](https://github.com/dominikh/go-mode.el)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/go-code.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
