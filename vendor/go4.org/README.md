# go4

[![travis badge](https://travis-ci.org/camlistore/go4.svg?branch=master)](https://travis-ci.org/camlistore/go4 "Travis CI")

[go4.org](http://go4.org) is a collection of packages for
Go programmers.

They started out living in [Camlistore](https://camlistore.org)'s repo
and elsewhere but they have nothing to do with Camlistore, so we're
moving them here.

## Details

* **single repo**. go4 is a single repo. That means things can be
    changed and rearranged globally atomically with ease and
    confidence.

* **no backwards compatibility**. go4 makes no backwards compatibility
    promises. If you want to use go4, vendor it. And next time you
    update your vendor tree, update to the latest API if things in go4
    changed. The plan is to eventually provide tools to make this
    easier.

* **forward progress** because we have no backwards compatibility,
    it's always okay to change things to make things better. That also
    means the bar for contributions is lower. We don't have to get the
    API 100% correct in the first commit.

* **code review** contributions must be code-reviewed. We're trying
    out Gerrithub, to see if we can find a mix of Github Pull Requests
    and Gerrit that works well for many people. We'll see.

* **CLA compliant** contributors must agree to the Google CLA (the
    same as Go itself). This ensures we can move things into Go as
    necessary in the future. It also makes lawyers at various
    companies happy.  The CLA is **not** a copyright *assignment*; you
    retain the copyright on your work. The CLA just says that your
    work is open source and you have permission to open source it. See
    https://golang.org/doc/contribute.html#tmp_6

* **docs, tests, portability** all code should be documented in the
    normal Go style, have tests, and be portable to different
    operating systems and architectures. We'll try to get builders in
    place to help run the tests on different OS/arches. For now we
    have Travis at least.

## Contributing

To add code to go4, send a pull request or push a change to Gerrithub.

We assume you already have your $GOPATH set and the go4 code cloned at
$GOPATH/src/go4.org. For example:

* `git clone https://review.gerrithub.io/camlistore/go4 $GOPATH/src/go4.org`

### To push a code review to Gerrithub directly:

* Sign in to [http://gerrithub.io](http://gerrithub.io "Gerrithub") with your Github account.

* Install the git hook that adds the magic "Change-Id" line to your commit messages:

  `curl -o $GOPATH/src/go4.org/.git/hooks/commit-msg https://camlistore.googlesource.com/camlistore/+/master/misc/commit-msg.githook`

* make changes

* commit (the unit of code review is a single commit identified by the Change-ID, **NOT** a series of commits on a branch)

* `git push ssh://$YOUR_GITHUB_USERNAME@review.gerrithub.io:29418/camlistore/go4 HEAD:refs/for/master`

### Using Github Pull Requests

* send a pull request with a single commit

* create a Gerrithub code review at https://review.gerrithub.io/plugins/github-plugin/static/pullrequests.html, selecting the pull request you just created.

### Problems contributing?

* Please file an issue or contact the [Camlistore mailing list](https://groups.google.com/forum/#!forum/camlistore) for any problems with the above.

See [https://review.gerrithub.io/Documentation/user-upload.html](https://review.gerrithub.io/Documentation/user-upload.html) for more generic documentation.

(TODO: more docs on Gerrit, integrate git-codereview, etc.)

