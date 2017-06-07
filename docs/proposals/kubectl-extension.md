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
[here](http://releases.k8s.io/release-1.3/docs/proposals/kubectl-extension.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubectl Extension

Abstract
--------

Allow `kubectl` to be extended to include other commands that can provide new functionality without recompiling Kubectl


Motivation and Background
-------------------------

Kubernetes is designed to be a composable and extensible system, with the ability to add new APIs and features via Third Party Resources
or API federation, by making the server provide functionality that eases writing generic clients, and by supporting other authentication
systems. Given that `kubectl` is the primary method for interacting with the server, some new extensions are difficult to make usable
for end users without recompiling that command. In addition, it is difficult to prototype new functionality for kubectl outside of the
Kubernetes source tree.

Ecosystem tools like OpenShift, Deis, and Helm add additional workflow around kubectl targeted at the end user. It is beneficial
to encourage workflows to develop around Kubernetes without requiring them to be part of Kubernetes to both the end user community
and the Kubernetes developer community.

There are many tools that currently offer CLI extension for the same reasons - [Git](https://www.kernel.org/pub/software/scm/git/docs/howto/new-command.html) and
[Heroku](https://devcenter.heroku.com/articles/developing-cli-plug-ins#creating-the-package) are two relevant examples in the space.


Proposal
--------

Define a system for `kubectl` that allows new subcommands and subcommand trees to be added by placing an executable in a specific
location on disk, like Git.  Allow third parties to extend kubectl by placing their extensions in that directory. Ensure that help
and other logic correctly includes those extensions.

A kubectl command extension would be an executable located in `EXEC_PATH` (an arbitrary directory to be defined that follows similar
conventions in Linux) with a name pattern like `kubectl-COMMAND[-SUBCOMMAND[...]]` with one or many sub parts. The presence of
a command extension overrides any built in command.

A key requirement is that the lookup be fast (since it would be invoked on every execution of `kubectl`) and so some true extension
behavior (such as complex inference of commands) may not be supported in order to reduce the complexity of the lookup.

Kubectl would lazily include the appropriate commands in preference to the internal command structure if detected (a user asking for
`kubectl a b c` would *first* check for `kubectl-a-b-c`, `kubectl-a-b`, or `kubectl-a` before loading the internal command).

All kubectl command extensions MUST:

* Support the `-h` and `--help` flags to display a help page
* Respect the semantics of KUBECONFIG lookup (to be further specified)

All kubectl command extensions SHOULD:

* Follow the display and output conventions of normal kubectl commands.

