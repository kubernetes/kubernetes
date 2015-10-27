<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes API and Release Versioning

Legend:

* **Kube &lt;major&gt;.&lt;minor&gt;.&lt;patch&gt;** refers to the version of Kubernetes that is released. This versions all components: apiserver, kubelet, kubectl, etc.
* **API vX[betaY]** refers to the version of the HTTP API.

## Release Timeline

### Minor version scheme and timeline

* Kube 1.0.0, 1.0.1 -- DONE!
* Kube 1.0.X (X>1): Standard operating procedure. We patch the release-1.0 branch as needed and increment the patch number.
* Kube 1.1.0-alpha.X: Released roughly every two weeks by cutting from HEAD. No cherrypick releases. If there is a critical bugfix, a new release from HEAD can be created ahead of schedule.
* Kube 1.1.0-beta: When HEAD is feature-complete, we will cut the release-1.1.0 branch 2 weeks prior to the desired 1.1.0 date and only merge PRs essential to 1.1.  This cut will be marked as 1.1.0-beta, and HEAD will be revved to 1.2.0-alpha.0.
* Kube 1.1.0: Final release, cut from the release-1.1.0 branch cut two weeks prior. Should occur between 3 and 4 months after 1.0.  1.1.1-beta will be tagged at the same time on the same branch.

### Major version timeline

There is no mandated timeline for major versions. They only occur when we need to start the clock on deprecating features. A given major version should be the latest major version for at least one year from its original release date.

## Release versions as related to API versions

Here is an example major release cycle:

* **Kube 1.0 should have API v1 without v1beta\* API versions**
 * The last version of Kube before 1.0 (e.g. 0.14 or whatever it is) will have the stable v1 API. This enables you to migrate all your objects off of the beta API versions of the API and allows us to remove those beta API versions in Kube 1.0 with no effect. There will be tooling to help you detect and migrate any v1beta\* data versions or calls to v1 before you do the upgrade.
* **Kube 1.x may have API v2beta***
 * The first incarnation of a new (backwards-incompatible) API in HEAD is v2beta1. By default this will be unregistered in apiserver, so it can change freely. Once it is available by default in apiserver (which may not happen for several minor releases), it cannot change ever again because we serialize objects in versioned form, and we always need to be able to deserialize any objects that are saved in etcd, even between alpha versions. If further changes to v2beta1 need to be made, v2beta2 is created, and so on, in subsequent 1.x versions.
* **Kube 1.y (where y is the last version of the 1.x series) must have final API v2**
 * Before Kube 2.0 is cut, API v2 must be released in 1.x. This enables two things: (1) users can upgrade to API v2 when running Kube 1.x and then switch over to Kube 2.x transparently, and (2) in the Kube 2.0 release itself we can cleanup and remove all API v2beta\* versions because no one should have v2beta\* objects left in their database. As mentioned above, tooling will exist to make sure there are no calls or references to a given API version anywhere inside someone's kube installation before someone upgrades.
 * Kube 2.0 must include the v1 API, but Kube 3.0 must include the v2 API only. It *may* include the v1 API as well if the burden is not high - this will be determined on a per-major-version basis.

## Rationale for API v2 being complete before v2.0's release

It may seem a bit strange to complete the v2 API before v2.0 is released, but *adding* a v2 API is not a breaking change. *Removing* the v2beta\* APIs *is* a breaking change, which is what necessitates the major version bump. There are other ways to do this, but having the major release be the fresh start of that release's API without the baggage of its beta versions seems most intuitive out of the available options.

# Patches

Patch releases are intended for critical bug fixes to the latest minor version, such as addressing security vulnerabilities, fixes to problems affecting a large number of users, severe problems with no workaround, and blockers for products based on Kubernetes.

They should not contain miscellaneous feature additions or improvements, and especially no incompatibilities should be introduced between patch versions of the same minor version (or even major version).

Dependencies, such as Docker or Etcd, should also not be changed unless absolutely necessary, and also just to fix critical bugs (so, at most patch version changes, not new major nor minor versions).

# Upgrades

* Users can upgrade from any Kube 1.x release to any other Kube 1.x release as a rolling upgrade across their cluster. (Rolling upgrade means being able to upgrade the master first, then one node at a time. See #4855 for details.)
* No hard breaking changes over version boundaries.
 * For example, if a user is at Kube 1.x, we may require them to upgrade to Kube 1.x+y before upgrading to Kube 2.x. In others words, an upgrade across major versions (e.g. Kube 1.x to Kube 2.x) should effectively be a no-op and as graceful as an upgrade from Kube 1.x to Kube 1.x+1. But you can require someone to go from 1.x to 1.x+y before they go to 2.x.

There is a separate question of how to track the capabilities of a kubelet to facilitate rolling upgrades. That is not addressed here.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/versioning.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
