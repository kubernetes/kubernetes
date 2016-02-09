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

Let’s rewrite this doc that we provide to the community, which covers [how to contribute to our docs for a K8s release](https://github.com/kubernetes/kubernetes/blob/master/docs/devel/update-release-docs.md).

### Adding a new docs collection for a release

Whenever a new release series (release-X.Y) is cut from master, we push the corresponding set of docs to http://kubernetes.io/vX.Y/docs. The steps are as follows:

* Create a _vX.Y folder in the kubernetes.github.io repo.

* Add vX.Y as a valid collection in `_config.yml`.

* Create a new `_includes/nav_vX.Y.html` file with the navigation menu. This can be a copy of `_includes/nav_vX.Y-1.html` with links to new docs added and links to deleted docs removed. Update _layouts/docwithnav.html to include this new navigation html file. Example PR: [#16143](https://github.com/kubernetes/kubernetes/pull/16143).

* [Pull docs from release branch](https://github.com/kubernetes/kubernetes/blob/master/docs/devel/update-release-docs.md#updating-docs-in-gh-pages-branch) in _vX.Y folder.

Once these changes have been submitted, you should be able to reach the docs at http://kubernetes.io/vX.Y/docs/where you can test them.

To make X.Y the default version of docs:

* Update `_config.yml` and `index.md` to point to the new version. Example PR:[#16416](https://github.com/kubernetes/kubernetes/pull/16416).

* Update `_includes/docversionselector.html` to make vX.Y the default version.

* Add `Disallow: /vX.Y-1/` to existing robots.txt file to hide old content from web crawlers and focus SEO on new docs. Example PR: [#16388](https://github.com/kubernetes/kubernetes/pull/16388).

* Regenerate `sitemaps.xml` so that it now contains vX.Y links. Sitemap can be regenerated using [https://www.xml-sitemaps.com](https://www.xml-sitemaps.com/). Example PR: [#17126](https://github.com/kubernetes/kubernetes/pull/17126).

* Resubmit the updated sitemaps file to [Google webmasters](https://www.google.com/webmasters/tools/sitemap-list?siteUrl=http://kubernetes.io/) for google to index the new links.

* Update `_layouts/docwithnav.html` to include `_includes/archivedocnotice.html` for vX.Y-1 docs which need to be archived.

* Ping @johndmulhausen to update docs.k8s.io to redirect to http://kubernetes.io/vX.Y/. [#18788](https://github.com/kubernetes/kubernetes/issues/18788).

[http://kubernetes.io/docs/](http://kubernetes.io/docs/) should now be redirecting to http://kubernetes.io/vX.Y/.

## Where is <HEAD> for the docs?

Essentially, the kubernetes.github.io/vX.Y folder for vLATEST operates as head. The docs should already be in their final destination and have its links pointing to the kubernetes.io URLs.

If versioning the links is necessary, then vX.Y folders should use {% variables %} to {{ print }} the version inside the link URLs, like so:

    Click here for [hello world](/{{ version }}/hello-world)

## Maintaining old docs

Updates to the docs for past versions of Kubernetes are unlikely; we're aiming to keep <HEAD> up-to-date, and when a new release comes out, copy <HEAD> into the folder for the new release and start working there, leaving that version of the docs archived/stale. 

