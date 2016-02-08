# Docs Process

GitHub Pages and K8s Docs

[johndylan@google.com](mailto:johndylan@google.com)

Executive summary

When a community is empowered to be effective contributors, great things happen. That’s why I’m proposing that we move the documentation from the docs/ folder in the core Kubernetes repo in GitHub into the GitHub pages repo itself, and migrate that repo into one named kubernetes.github.io so that it will be compatible with the GitHub Pages auto-update system.

Benefits:

* Onboard contributors quickly with tech they are familiar with that is easy to use

* Have in-the-cloud staging of our site on *username*.github.io

* Have documentation and website material located in one place

* Reduce dependency on scripts and tools that only a few people can use/understand

Cost:

* Migration of current head/stable release system that is done with build/integration scripts into GitHub branches

* Updating of any doc generation tools so that they produce Jekyll-ready files

# Github.io staging

GitHub has a technology called GitHub Pages that we currently use to construct the site at kubernetes.io. GitHub Pages is essentially a system for managing and hosting Jekyll sites. When you check in a Jekyll site into a repo that is named username.github.io, then GitHub will automatically stage its contents at [http://kubernetes.github.io](http://username.github.io), and automatically update it whenever you push new changes -- no publishing required.

This highly lubricates the ability for external users to make changes, as they can immediately see the consequences of their actions in their forked environment. 

## Why migrate to kubernetes.github.io instead of using a gh-pages branch?

GitHub Pages also suppots using branches of an existing repo to host Jekyll sites; in this case you would reuse the repo, but check in the Jekyll site to a branch named ‘gh-pages.’ While you still get auto-building and staging, it will build at [http://kubernetes.github.io/](http://username.github.io/repo)**[rep**o](http://username.github.io/repo) -- and in our case that is problematic because absolute links to any URI within the site will not work the same as they will on kubernetes.io. If we use the username.github.io repo instead of the gh-pages branch, then including an image from "/images/example.png" will work the same on [http://kubernetes.github.io](http://username.github.io) as it will on [http://kubernetes.io](http://kubernetes.io), becuase the staging is not publishing that image at [http://kubernetes.github.io/](http://username.github.io/repo/images/example.png)**[rep**o](http://username.github.io/repo/images/example.png)[/images/example.png](http://username.github.io/repo/images/example.png). 

## Overview of docs migration and how release process will change

Right now, the subdirectory of kubernetes/docs acts as the HEAD for the docs, which is used to populate new releases, as well as the basis for cherrypicking changes into docs for old releases. However, consider a "roaming HEAD" situation, such that the new release is always treated as the basis for populating new releases and  cherrypicks. In otherwords, whatever information is [http://kubernetes.io/vNEWEST/](http://kubernetes.io/vX.Y/) is HEAD, and we no longer have a dependency on an underlying “master” docset. 

New releases can be developed under [http://kubernetets.io/vNEXT](http://kubernetets.io/vNEXT). We can do this because there is an extensive set of steps to get any one particular folder full of docs to be the default. Or, at worst, we can make a vNEXT branch of the kubernetes/kubernetes.github.io repo and do vNEXT work there, culminating in a pull request at release time. But this would only be necessary if we don’t want users to ever see what vNEXT docs look like. I personally vote for doing our work out in public. We can always mark those docs "beta" -- and keeping them in the master branch means that we get to use all the great github.io staging tools. 

## Adapting the current docs process for releases

Nuts and bolts: How would a pure-GitHub Pages process work? 

Let’s rewrite this doc that we provide to the community, which covers [how to contribute to our docs for a K8s release](https://github.com/kubernetes/kubernetes/blob/master/docs/devel/update-release-docs.md). Changes are highlighted in **green**. Steps no longer necessary have been stricken.

### Adding a new docs collection for a release

Whenever a new release series (release-X.Y) is cut from master, we push the corresponding set of docs to http://kubernetes.io/vX.Y/docs. The steps are as follows:

* Create a _vX.Y folder in **the kubernetes.github.io repo.** 

* Add vX.Y as a valid collection in [_config.yml](https://github.com/kubernetes/kubernetes/blob/gh-pages/_config.yml)

* Create a new _includes/nav_vX.Y.html file with the navigation menu. This can be a copy of _includes/nav_vX.Y-1.html with links to new docs added and links to deleted docs removed. Update [_layouts/docwithnav.html](https://github.com/kubernetes/kubernetes/blob/gh-pages/_layouts/docwithnav.html) to include this new navigation html file. Example PR: [#16143](https://github.com/kubernetes/kubernetes/pull/16143).

* [Pull docs from release branch](https://github.com/kubernetes/kubernetes/blob/master/docs/devel/update-release-docs.md#updating-docs-in-gh-pages-branch) in _vX.Y folder.

Once these changes have been submitted, you should be able to reach the docs at http://kubernetes.io/vX.Y/docs/where you can test them.

To make X.Y the default version of docs:

* Update [_config.yml](https://github.com/kubernetes/kubernetes/blob/gh-pages/_config.yml) and /kubernetes/**kubernetes.github.io/blob/master/docs/index.md** to point to the new version. Example PR:[#16416](https://github.com/kubernetes/kubernetes/pull/16416).

* Update [_includes/docversionselector.html](https://github.com/kubernetes/kubernetes/blob/gh-pages/_includes/docversionselector.html) to make vX.Y the default version.

* Add "Disallow: /vX.Y-1/" to existing [robots.txt](https://github.com/kubernetes/kubernetes/blob/gh-pages/robots.txt) file to hide old content from web crawlers and focus SEO on new docs. Example PR: [#16388](https://github.com/kubernetes/kubernetes/pull/16388).

* Regenerate [sitemaps.xml](https://github.com/kubernetes/kubernetes/blob/gh-pages/sitemap.xml) so that it now contains vX.Y links. Sitemap can be regenerated using [https://www.xml-sitemaps.com](https://www.xml-sitemaps.com/). Example PR: [#17126](https://github.com/kubernetes/kubernetes/pull/17126).

* Resubmit the updated sitemaps file to [Google webmasters](https://www.google.com/webmasters/tools/sitemap-list?siteUrl=http://kubernetes.io/) for google to index the new links.

* Update [_layouts/docwithnav.html](https://github.com/kubernetes/kubernetes/blob/gh-pages/_layouts/docwithnav.html) to include [_includes/archivedocnotice.html](https://github.com/kubernetes/kubernetes/blob/gh-pages/_includes/archivedocnotice.html) for vX.Y-1 docs which need to be archived.

* Ping **@johndylan** to update docs.k8s.io to redirect to http://kubernetes.io/vX.Y/. [#18788](https://github.com/kubernetes/kubernetes/issues/18788).

[http://kubernetes.io/docs/](http://kubernetes.io/docs/) should now be redirecting to http://kubernetes.io/vX.Y/.

### Updating docs in an existing collection

The high level steps to update docs in an existing collection are:

1. Update docs on HEAD (master branch) of **kubernetes/kubernetes.github.io**

2. Cherrypick the change in **by copying those changes from the newest docs folder to the desired collection.**

3. Update docs on gh-pages.

## Updating docs in gh-pages branch

Once release branch has all the relevant changes, we can pull in the latest docs in gh-pages branch. Run the following 2 commands in gh-pages branch to update docs for release X.Y:

_tools/import_docs vX.Y _vX.Y release-X.Y release-X.Y

For ex: to pull in docs for release 1.1, run:

_tools/import_docs v1.1 _v1.1 release-1.1 release-1.1

Apart from copying over the docs, _tools/release_docs also does some post processing (like updating the links to docs to point to [http://kubernetes.io/docs/](http://kubernetes.io/docs/) instead of pointing to github repo). Note that we always pull in the docs from release branch and not from master (pulling docs from master requires some extra processing like versionizing the links and removing unversioned warnings).

We delete all existing docs before pulling in new ones to ensure that deleted docs go away.

If the change added or deleted a doc, then update the corresponding _includes/nav_vX.Y.html file as well.

This process is eliminated by making vLATEST head; the docs should already be in their final destination and have its links pointing to the kubernetes.io URLs. If versioning the links is necessary, then vX.Y folders should use {% variables %} to {{ print }} the version inside the link URLs, like so:

**Click here for [hello world](/{{ version }}/hello-world).**

