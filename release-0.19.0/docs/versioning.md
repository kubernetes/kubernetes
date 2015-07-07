# Kubernetes API and Release Versioning

Legend:

* **Kube &lt;major&gt;.&lt;minor&gt;.&lt;patch&gt;** refers to the version of Kubernetes that is released. This versions all components: apiserver, kubelet, kubectl, etc.
* **API vX[betaY]** refers to the version of the HTTP API.

## Release Timeline

### Minor version timeline

* Kube 1.0.0
* Kube 1.0.x: We create a 1.0-patch branch and backport critical bugs and security issues to it. Patch releases occur as needed.
* Kube 1.1-alpha1: Cut from HEAD, smoke tested and released two weeks after Kube 1.0's release. Roughly every two weeks a new alpha is released from HEAD. The timeline is flexible; for example, if there is a critical bugfix, a new alpha can be released ahead of schedule. (This applies to the beta and rc releases as well.)
* Kube 1.1-beta1: When HEAD is feature complete, we create a 1.1-snapshot branch and release it as a beta. (The 1.1-snapshot branch may be created earlier if something that definitely won't be in 1.1 needs to be merged to HEAD.) This should occur 6-8 weeks after Kube 1.0. Development continues at HEAD and only fixes are backported to 1.1-snapshot.
* Kube 1.1-rc1: Released from 1.1-snapshot when it is considered stable and ready for testing. Most users should be able to upgrade to this version in production.
* Kube 1.1: Final release. Should occur between 3 and 4 months after 1.0.

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

# Upgrades

* Users can upgrade from any Kube 1.x release to any other Kube 1.x release as a rolling upgrade across their cluster. (Rolling upgrade means being able to upgrade the master first, then one node at a time. See #4855 for details.)
* No hard breaking changes over version boundaries.
 * For example, if a user is at Kube 1.x, we may require them to upgrade to Kube 1.x+y before upgrading to Kube 2.x. In others words, an upgrade across major versions (e.g. Kube 1.x to Kube 2.x) should effectively be a no-op and as graceful as an upgrade from Kube 1.x to Kube 1.x+1. But you can require someone to go from 1.x to 1.x+y before they go to 2.x.

There is a separate question of how to track the capabilities of a kubelet to facilitate rolling upgrades. That is not addressed here.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/versioning.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/versioning.md?pixel)]()
