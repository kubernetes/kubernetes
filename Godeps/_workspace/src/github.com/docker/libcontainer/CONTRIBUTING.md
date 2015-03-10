# The libcontainer Contributors' Guide

Want to hack on libcontainer? Awesome! Here are instructions to get you
started. They are probably not perfect, please let us know if anything
feels wrong or incomplete.

## Reporting Issues

When reporting [issues](https://github.com/docker/libcontainer/issues)
on GitHub please include your host OS (Ubuntu 12.04, Fedora 19, etc),
the output of `uname -a`. Please include the steps required to reproduce
the problem if possible and applicable.
This information will help us review and fix your issue faster.

## Development Environment

### Requirements

For best results, use a Linux development environment.
The following packages are required to compile libcontainer natively.

- Golang 1.3
- GCC
- git
- cgutils

You can develop on OSX, but you are limited to Dockerfile-based builds only.

### Building libcontainer from Dockerfile

    make all

This is the easiest way of building libcontainer.
As this build is done using Docker, you can even run this from [OSX](https://github.com/boot2docker/boot2docker)

### Testing changes with "nsinit"

    make sh

This will create an container that runs `nsinit exec sh` on a busybox rootfs with the configuration from ['minimal.json'](https://github.com/docker/libcontainer/blob/master/sample_configs/minimal.json).
Like the previous command, you can run this on OSX too!

### Building libcontainer directly

> Note: You should add the `vendor` directory to your GOPATH to use the vendored libraries

    ./update-vendor.sh
    go get -d ./...
    make direct-build
    # Run the tests
    make direct-test-short | egrep --color 'FAIL|$'
    # Run all the test
    make direct-test | egrep --color 'FAIL|$'

### Testing Changes with "nsinit" directly

To test a change:

    # Install nsinit
    make direct-install

    # Optional, add a docker0 bridge
    ip link add docker0 type bridge
    ifconfig docker0 172.17.0.1/16 up

    mkdir testfs
    curl -sSL https://github.com/jpetazzo/docker-busybox/raw/buildroot-2014.02/rootfs.tar | tar -xC testfs
    cd testfs
    cp <your-sample-config.json> container.json
    nsinit exec sh

## Contribution Guidelines

### Pull requests are always welcome

We are always thrilled to receive pull requests, and do our best to
process them as fast as possible. Not sure if that typo is worth a pull
request? Do it! We will appreciate it.

If your pull request is not accepted on the first try, don't be
discouraged! If there's a problem with the implementation, hopefully you
received feedback on what to improve.

We're trying very hard to keep libcontainer lean and focused. We don't want it
to do everything for everybody. This means that we might decide against
incorporating a new feature. However, there might be a way to implement
that feature *on top of* libcontainer.

### Discuss your design on the mailing list

We recommend discussing your plans [on the mailing
list](https://groups.google.com/forum/?fromgroups#!forum/libcontainer)
before starting to code - especially for more ambitious contributions.
This gives other contributors a chance to point you in the right
direction, give feedback on your design, and maybe point out if someone
else is working on the same thing.

### Create issues...

Any significant improvement should be documented as [a GitHub
issue](https://github.com/docker/libcontainer/issues) before anybody
starts working on it.

### ...but check for existing issues first!

Please take a moment to check that an issue doesn't already exist
documenting your bug report or improvement proposal. If it does, it
never hurts to add a quick "+1" or "I have this problem too". This will
help prioritize the most common problems and requests.

### Conventions

Fork the repo and make changes on your fork in a feature branch:

- If it's a bugfix branch, name it XXX-something where XXX is the number of the
  issue
- If it's a feature branch, create an enhancement issue to announce your
  intentions, and name it XXX-something where XXX is the number of the issue.

Submit unit tests for your changes.  Go has a great test framework built in; use
it! Take a look at existing tests for inspiration. Run the full test suite on
your branch before submitting a pull request.

Update the documentation when creating or modifying features. Test
your documentation changes for clarity, concision, and correctness, as
well as a clean documentation build. See ``docs/README.md`` for more
information on building the docs and how docs get released.

Write clean code. Universally formatted code promotes ease of writing, reading,
and maintenance. Always run `gofmt -s -w file.go` on each changed file before
committing your changes. Most editors have plugins that do this automatically.

Pull requests descriptions should be as clear as possible and include a
reference to all the issues that they address.

Pull requests must not contain commits from other users or branches.

Commit messages must start with a capitalized and short summary (max. 50
chars) written in the imperative, followed by an optional, more detailed
explanatory text which is separated from the summary by an empty line.

Code review comments may be added to your pull request. Discuss, then make the
suggested modifications and push additional commits to your feature branch. Be
sure to post a comment after pushing. The new commits will show up in the pull
request automatically, but the reviewers will not be notified unless you
comment.

Before the pull request is merged, make sure that you squash your commits into
logical units of work using `git rebase -i` and `git push -f`. After every
commit the test suite should be passing. Include documentation changes in the
same commit so that a revert would remove all traces of the feature or fix.

Commits that fix or close an issue should include a reference like `Closes #XXX`
or `Fixes #XXX`, which will automatically close the issue when merged.

### Testing

Make sure you include suitable tests, preferably unit tests, in your pull request
and that all the tests pass.

*Instructions for running tests to be added.*

### Merge approval

libcontainer maintainers use LGTM (looks good to me) in comments on the code review
to indicate acceptance.

A change requires LGTMs from at lease two maintainers. One of those must come from
a maintainer of the component affected. For example, if a change affects `netlink/`
and `security`, it needs at least one LGTM from a maintainer of each. Maintainers
only need one LGTM as presumably they LGTM their own change.

For more details see [MAINTAINERS.md](MAINTAINERS.md)

### Sign your work

The sign-off is a simple line at the end of the explanation for the
patch, which certifies that you wrote it or otherwise have the right to
pass it on as an open-source patch.  The rules are pretty simple: if you
can certify the below (from
[developercertificate.org](http://developercertificate.org/)):

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

then you just add a line to every git commit message:

    Docker-DCO-1.1-Signed-off-by: Joe Smith <joe.smith@email.com> (github: github_handle)

using your real name (sorry, no pseudonyms or anonymous contributions.)

One way to automate this, is customise your get ``commit.template`` by adding
a ``prepare-commit-msg`` hook to your libcontainer checkout:

```
curl -o .git/hooks/prepare-commit-msg https://raw.githubusercontent.com/docker/docker/master/contrib/prepare-commit-msg.hook && chmod +x .git/hooks/prepare-commit-msg
```

* Note: the above script expects to find your GitHub user name in ``git config --get github.user``

#### Small patch exception

There are several exceptions to the signing requirement. Currently these are:

* Your patch fixes spelling or grammar errors.
* Your patch is a single line change to documentation contained in the
  `docs` directory.
* Your patch fixes Markdown formatting or syntax errors in the
  documentation contained in the `docs` directory.

If you have any questions, please refer to the FAQ in the [docs](to be written)

### How can I become a maintainer?

* Step 1: learn the component inside out
* Step 2: make yourself useful by contributing code, bugfixes, support etc.
* Step 3: volunteer on the irc channel (#libcontainer@freenode)

Don't forget: being a maintainer is a time investment. Make sure you will have time to make yourself available.
You don't have to be a maintainer to make a difference on the project!

