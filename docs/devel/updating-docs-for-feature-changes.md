<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# How to update docs for new kubernetes features

This document describes things to consider when updating Kubernetes docs for new features or changes to existing features (including removing features).

## Who should read this doc?

Anyone making user facing changes to kubernetes.  This is especially important for Api changes or anything impacting the getting started experience.

## What docs changes are needed when adding or updating a feature in kubernetes?

### When making Api changes

*e.g. adding Deployments*
* Always make sure docs for downstream effects are updated *(PetSet -> PVC, Deployment -> ReplicationController)*
* Add or update the corresponding *[Glossary](http://kubernetes.io/docs/reference/)* item
* Verify the guides / walkthroughs do not require any changes:
  * **If your change will be recommended over the approaches shown in these guides, then they must be updated to reflect your change**
  * [Hello Node](http://kubernetes.io/docs/hellonode/)
  * [K8s101](http://kubernetes.io/docs/user-guide/walkthrough/)
  * [K8S201](http://kubernetes.io/docs/user-guide/walkthrough/k8s201/)
  * [Guest-book](https://github.com/kubernetes/kubernetes/tree/release-1.2/examples/guestbook)
  * [Thorough-walkthrough](http://kubernetes.io/docs/user-guide/)
* Verify the [landing page examples](http://kubernetes.io/docs/samples/) do not require any changes (those under "Recently updated samples")
  * **If your change will be recommended over the approaches shown in the "Updated" examples, then they must be updated to reflect your change**
  * If you are aware that your change will be recommended over the approaches shown in non-"Updated" examples, create an Issue
* Verify the collection of docs under the "Guides" section do not require updates (may need to use grep for this until are docs are more organized)

### When making Tools changes

*e.g. updating kube-dash or kubectl*
* If changing kubectl, verify the guides / walkthroughs do not require any changes:
  * **If your change will be recommended over the approaches shown in these guides, then they must be updated to reflect your change**
  * [Hello Node](http://kubernetes.io/docs/hellonode/)
  * [K8s101](http://kubernetes.io/docs/user-guide/walkthrough/)
  * [K8S201](http://kubernetes.io/docs/user-guide/walkthrough/k8s201/)
  * [Guest-book](https://github.com/kubernetes/kubernetes/tree/release-1.2/examples/guestbook)
  * [Thorough-walkthrough](http://kubernetes.io/docs/user-guide/)
* If updating an existing tool
  * Search for any docs about the tool and update them
* If adding a new tool for end users
  * Add a new page under [Guides](http://kubernetes.io/docs/)
* **If removing a tool (kube-ui), make sure documentation that references it is updated appropriately!**

### When making cluster setup changes

*e.g. adding Multi-AZ support*
* Update the relevant [Administering Clusters](http://kubernetes.io/docs/) pages

### When making Kubernetes binary changes

*e.g. adding a flag, changing Pod GC behavior, etc*
* Add or update a page under [Configuring Kubernetes](http://kubernetes.io/docs/)

## Where do the docs live?

1. Most external user facing docs live in the [kubernetes/docs](https://github.com/kubernetes/kubernetes.github.io) repo
  * Also see the *[general instructions](http://kubernetes.io/editdocs/)* for making changes to the docs website
2. Internal design and development docs live in the [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes) repo

## Who should help review docs changes?

* cc *@kubernetes/docs*
* Changes to [kubernetes/docs](https://github.com/kubernetes/kubernetes.github.io) repo must have both a Technical Review and a Docs Review

## Tips for writing new docs

* Try to keep new docs small and focused
* Document pre-requisites (if they exist)
* Document what concepts will be covered in the document
* Include screen shots or pictures in documents for GUIs
* *TODO once we have a standard widget set we are happy with* - include diagrams to help describe complex ideas (not required yet)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/updating-docs-for-feature-changes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
