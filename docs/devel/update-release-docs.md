<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Table of Contents

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Adding a new docs collection for a release](#adding-a-new-docs-collection-for-a-release)
- [Updating docs in an existing collection](#updating-docs-in-an-existing-collection)
  - [Updating docs on HEAD](#updating-docs-on-head)
  - [Updating docs in release branch](#updating-docs-in-release-branch)
  - [Updating docs in gh-pages branch](#updating-docs-in-gh-pages-branch)

<!-- END MUNGE: GENERATED_TOC -->

# Overview

This document explains how to update kubernetes release docs hosted at http://kubernetes.io/docs/.

http://kubernetes.io is served using the [gh-pages
branch](https://github.com/kubernetes/kubernetes/tree/gh-pages) of kubernetes repo on github.
Updating docs in that branch will update http://kubernetes.io

There are 2 scenarios which require updating docs:
* Adding a new docs collection for a release.
* Updating docs in an existing collection.

# Adding a new docs collection for a release

Whenever a new release series (`release-X.Y`) is cut from `master`, we push the
corresponding set of docs to `http://kubernetes.io/vX.Y/docs`. The steps are as follows:

* Create a `_vX.Y` folder in `gh-pages` branch.
* Add `vX.Y` as a valid collection in [_config.yml](https://github.com/kubernetes/kubernetes/blob/gh-pages/_config.yml)
* Create a new `_includes/nav_vX.Y.html` file with the navigation menu. This can
  be a copy of `_includes/nav_vX.Y-1.html` with links to new docs added and links
  to deleted docs removed. Update [_layouts/docwithnav.html]
  (https://github.com/kubernetes/kubernetes/blob/gh-pages/_layouts/docwithnav.html)
  to include this new navigation html file. Example PR: [#16143](https://github.com/kubernetes/kubernetes/pull/16143).
* [Pull docs from release branch](#updating-docs-in-gh-pages-branch) in `_vX.Y`
  folder.

Once these changes have been submitted, you should be able to reach the docs at
`http://kubernetes.io/vX.Y/docs/` where you can test them.

To make `X.Y` the default version of docs:

* Update [_config.yml](https://github.com/kubernetes/kubernetes/blob/gh-pages/_config.yml)
  and [/kubernetes/kubernetes/blob/gh-pages/_docs/index.md](https://github.com/kubernetes/kubernetes/blob/gh-pages/_docs/index.md)
  to point to the new version. Example PR: [#16416](https://github.com/kubernetes/kubernetes/pull/16416).
* Update [_includes/docversionselector.html](https://github.com/kubernetes/kubernetes/blob/gh-pages/_includes/docversionselector.html)
  to make `vX.Y` the default version.
* Add "Disallow: /vX.Y-1/" to existing [robots.txt](https://github.com/kubernetes/kubernetes/blob/gh-pages/robots.txt)
  file to hide old content from web crawlers and focus SEO on new docs. Example PR:
  [#16388](https://github.com/kubernetes/kubernetes/pull/16388).
* Regenerate [sitemaps.xml](https://github.com/kubernetes/kubernetes/blob/gh-pages/sitemap.xml)
  so that it now contains `vX.Y` links. Sitemap can be regenerated using
  https://www.xml-sitemaps.com. Example PR: [#17126](https://github.com/kubernetes/kubernetes/pull/17126).
* Resubmit the updated sitemaps file to [Google
  webmasters](https://www.google.com/webmasters/tools/sitemap-list?siteUrl=http://kubernetes.io/) for google to index the new links.
* Update [_layouts/docwithnav.html] (https://github.com/kubernetes/kubernetes/blob/gh-pages/_layouts/docwithnav.html)
  to include [_includes/archivedocnotice.html](https://github.com/kubernetes/kubernetes/blob/gh-pages/_includes/archivedocnotice.html)
  for `vX.Y-1` docs which need to be archived.
* Ping @thockin to update docs.k8s.io to redirect to `http://kubernetes.io/vX.Y/`. [#18788](https://github.com/kubernetes/kubernetes/issues/18788).

http://kubernetes.io/docs/ should now be redirecting to `http://kubernetes.io/vX.Y/`.

# Updating docs in an existing collection

The high level steps to update docs in an existing collection are:

1. Update docs on `HEAD` (master branch)
2. Cherryick the change in relevant release branch.
3. Update docs on `gh-pages`.

## Updating docs on HEAD

[Development guide](development.md) provides general instructions on how to contribute to kubernetes github repo.
[Docs how to guide](how-to-doc.md) provides conventions to follow while writing docs.

## Updating docs in release branch

Once docs have been updated in the master branch, the changes need to be
cherrypicked in the latest release branch.
[Cherrypick guide](cherry-picks.md) has more details on how to cherrypick your change.

## Updating docs in gh-pages branch

Once release branch has all the relevant changes, we can pull in the latest docs
in `gh-pages` branch.
Run the following 2 commands in `gh-pages` branch to update docs for release `X.Y`:

```
_tools/import_docs vX.Y _vX.Y release-X.Y release-X.Y
```

For ex: to pull in docs for release 1.1, run:

```
_tools/import_docs v1.1 _v1.1 release-1.1 release-1.1
```

Apart from copying over the docs, `_tools/release_docs` also does some post processing
(like updating the links to docs to point to http://kubernetes.io/docs/ instead of pointing to github repo).
Note that we always pull in the docs from release branch and not from master (pulling docs
from master requires some extra processing like versionizing the links and removing unversioned warnings).

We delete all existing docs before pulling in new ones to ensure that deleted
docs go away.

If the change added or deleted a doc, then update the corresponding `_includes/nav_vX.Y.html` file as well.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/update-release-docs.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
