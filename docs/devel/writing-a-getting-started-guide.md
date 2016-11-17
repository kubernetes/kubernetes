# Writing a Getting Started Guide

This page gives some advice for anyone planning to write or update a Getting Started Guide for Kubernetes.
It also gives some guidelines which reviewers should follow when reviewing a pull request for a
guide.

A Getting Started Guide is instructions on how to create a Kubernetes cluster on top of a particular
type(s) of infrastructure.  Infrastructure includes: the IaaS provider for VMs;
the node OS; inter-node networking; and node Configuration Management system.
A guide refers to scripts, Configuration Management files, and/or binary assets such as RPMs.  We call
the combination of all these things needed to run on a particular type of infrastructure a
**distro**.

[The Matrix](../../docs/getting-started-guides/README.md) lists the distros.  If there is already a guide
which is similar to the one you have planned, consider improving that one.


Distros fall into two categories:
  - **versioned distros** are tested to work with a particular binary release of Kubernetes.  These
    come in a wide variety, reflecting a wide range of ideas and preferences in how to run a cluster.
  - **development distros** are tested work with the latest Kubernetes source code.  But, there are
    relatively few of these and the bar is much higher for creating one.  They must support
    fully automated cluster creation, deletion, and upgrade.

There are different guidelines for each.

## Versioned Distro Guidelines

These guidelines say *what* to do.  See the Rationale section for *why*.
 - Send us a PR.
 - Put the instructions in `docs/getting-started-guides/...`. Scripts go there too.  This helps devs easily
   search for uses of flags by guides.
 - We may ask that you host binary assets or large amounts of code in our `contrib` directory or on your
   own repo.
 - Add or update a row in [The Matrix](../../docs/getting-started-guides/README.md).
 - State the binary version of Kubernetes that you tested clearly in your Guide doc.
 - Setup a cluster and run the [conformance tests](e2e-tests.md#conformance-tests) against it, and report the
   results in your PR.
 - Versioned distros should typically not modify or add code in `cluster/`.  That is just scripts for developer
   distros.
 - When a new major or minor release of Kubernetes comes out, we may also release a new
   conformance test, and require a new conformance test run to earn a conformance checkmark.

If you have a cluster partially working, but doing all the above steps seems like too much work,
we still want to hear from you.  We suggest you write a blog post or a Gist, and we will link to it on our wiki page.
Just file an issue or chat us on [Slack](http://slack.kubernetes.io) and one of the committers will link to it from the wiki.

## Development Distro Guidelines

These guidelines say *what* to do.  See the Rationale section for *why*.
  - the main reason to add a new development distro is to support a new IaaS provider (VM and
    network management).  This means implementing a new `pkg/cloudprovider/providers/$IAAS_NAME`.
  - Development distros should use Saltstack for Configuration Management.
  - development distros need to support automated cluster creation, deletion, upgrading, etc.
    This mean writing scripts in `cluster/$IAAS_NAME`.
  - all commits to the tip of this repo need to not break any of the development distros
    - the author of the change is responsible for making changes necessary on all the cloud-providers if the
      change affects any of them, and reverting the change if it breaks any of the CIs.
  - a development distro needs to have an organization which owns it.  This organization needs to:
    - Setting up and maintaining Continuous Integration that runs e2e frequently (multiple times per day) against the
      Distro at head,  and which notifies all devs of breakage.
    - being reasonably available for questions and assisting with
      refactoring and feature additions that affect code for their IaaS.

## Rationale

 - We want people to create Kubernetes clusters with whatever IaaS, Node OS,
   configuration management tools, and so on, which they are familiar with.  The
   guidelines for **versioned distros** are designed for flexibility.
 - We want developers to be able to work without understanding all the permutations of
   IaaS, NodeOS, and configuration management.  The guidelines for **developer distros** are designed
   for consistency.
 - We want users to have a uniform experience with Kubernetes whenever they follow instructions anywhere
   in our Github repository.  So, we ask that versioned distros pass a **conformance test** to make sure
   really work.
 - We want to **limit the number of development distros** for several reasons.  Developers should
   only have to change a limited number of places to add a new feature.  Also, since we will
   gate commits on passing CI for all distros, and since end-to-end tests are typically somewhat
   flaky, it would be highly likely for there to be false positives and CI backlogs with many CI pipelines.
 - We do not require versioned distros to do **CI** for several reasons.  It is a steep
   learning curve to understand our automated testing scripts.  And it is considerable effort
   to fully automate setup and teardown of a cluster, which is needed for CI.  And, not everyone
   has the time and money to run CI.  We do not want to
   discourage people from writing and sharing guides because of this.
 - Versioned distro authors are free to run their own CI and let us know if there is breakage, but we
   will not include them as commit hooks -- there cannot be so many commit checks that it is impossible
   to pass them all.
 - We prefer a single Configuration Management tool for development distros.  If there were more
   than one, the core developers would have to learn multiple tools and update config in multiple
   places.  **Saltstack** happens to be the one we picked when we started the project.  We
   welcome versioned distros that use any tool; there are already examples of
   CoreOS Fleet, Ansible, and others.
 - You can still run code from head or your own branch
   if you use another Configuration Management tool -- you just have to do some manual steps
   during testing and deployment.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/writing-a-getting-started-guide.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
