# Contributing to the reference library

## Community help

If you need help, please ask in the [#distribution](https://cloud-native.slack.com/archives/C01GVR8SY4R) channel on CNCF community slack.
[Click here for an invite to the CNCF community slack](https://slack.cncf.io/)

## Reporting security issues

The maintainers take security seriously. If you discover a security
issue, please bring it to their attention right away!

Please **DO NOT** file a public issue, instead send your report privately to
[cncf-distribution-security@lists.cncf.io](mailto:cncf-distribution-security@lists.cncf.io).

## Reporting an issue properly

By following these simple rules you will get better and faster feedback on your issue.

 - search the bugtracker for an already reported issue

### If you found an issue that describes your problem:

 - please read other user comments first, and confirm this is the same issue: a given error condition might be indicative of different problems - you may also find a workaround in the comments
 - please refrain from adding "same thing here" or "+1" comments
 - you don't need to comment on an issue to get notified of updates: just hit the "subscribe" button
 - comment if you have some new, technical and relevant information to add to the case
 - __DO NOT__ comment on closed issues or merged PRs. If you think you have a related problem, open up a new issue and reference the PR or issue.

### If you have not found an existing issue that describes your problem:

 1. create a new issue, with a succinct title that describes your issue:
   - bad title: "It doesn't work with my docker"
   - good title: "Private registry push fail: 400 error with E_INVALID_DIGEST"
 2. copy the output of (or similar for other container tools):
   - `docker version`
   - `docker info`
   - `docker exec <registry-container> registry --version`
 3. copy the command line you used to launch your Registry
 4. restart your docker daemon in debug mode (add `-D` to the daemon launch arguments)
 5. reproduce your problem and get your docker daemon logs showing the error
 6. if relevant, copy your registry logs that show the error
 7. provide any relevant detail about your specific Registry configuration (e.g., storage backend used)
 8. indicate if you are using an enterprise proxy, Nginx, or anything else between you and your Registry

## Contributing Code

Contributions should be made via pull requests. Pull requests will be reviewed
by one or more maintainers or reviewers and merged when acceptable.

You should follow the basic GitHub workflow:

 1. Use your own [fork](https://help.github.com/en/articles/about-forks)
 2. Create your [change](https://github.com/containerd/project/blob/master/CONTRIBUTING.md#successful-changes)
 3. Test your code
 4. [Commit](https://github.com/containerd/project/blob/master/CONTRIBUTING.md#commit-messages) your work, always [sign your commits](https://github.com/containerd/project/blob/master/CONTRIBUTING.md#commit-messages)
 5. Push your change to your fork and create a [Pull Request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)

Refer to [containerd's contribution guide](https://github.com/containerd/project/blob/master/CONTRIBUTING.md#successful-changes)
for tips on creating a successful contribution.

## Sign your work

The sign-off is a simple line at the end of the explanation for the patch. Your
signature certifies that you wrote the patch or otherwise have the right to pass
it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.
