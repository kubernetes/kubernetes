# Contribution Guidelines

Development happens on GitHub.
Issues are used for bugs and actionable items and longer discussions can happen on the [mailing list](#mailing-list).

The content of this repository is licensed under the [Apache License, Version 2.0](LICENSE).

## Code of Conduct

Participation in the Open Container community is governed by [Open Container Code of Conduct][code-of-conduct].

## Meetings

The contributors and maintainers of all OCI projects have monthly meetings at 2:00 PM (USA Pacific) on the first Wednesday of every month.
There is an [iCalendar][rfc5545] format for the meetings [here][meeting.ics].
Everyone is welcome to participate via [UberConference web][UberConference] or audio-only: +1 415 968 0849 (no PIN needed).
An initial agenda will be posted to the [mailing list](#mailing-list) in the week before each meeting, and everyone is welcome to propose additional topics or suggest other agenda alterations there.
Minutes from past meetings are archived [here][minutes].

## Mailing list

You can subscribe and browse the mailing list on [Google Groups][mailing-list].

## IRC

OCI discussion happens on #opencontainers on [Freenode][] ([logs][irc-logs]).

## Git

### Security issues

If you are reporting a security issue, do not create an issue or file a pull
request on GitHub. Instead, disclose the issue responsibly by sending an email
to security@opencontainers.org (which is inhabited only by the maintainers of
the various OCI projects).

### Pull requests are always welcome

We are always thrilled to receive pull requests, and do our best to
process them as fast as possible. Not sure if that typo is worth a pull
request? Do it! We will appreciate it.

If your pull request is not accepted on the first try, don't be
discouraged! If there's a problem with the implementation, hopefully you
received feedback on what to improve.

We're trying very hard to keep the project lean and focused. We don't want it
to do everything for everybody. This means that we might decide against
incorporating a new feature.

### Conventions

Fork the repo and make changes on your fork in a feature branch.
For larger bugs and enhancements, consider filing a leader issue or mailing-list thread for discussion that is independent of the implementation.
Small changes or changes that have been discussed on the [project mailing list](#mailing-list) may be submitted without a leader issue.

If the project has a test suite, submit unit tests for your changes. Take a
look at existing tests for inspiration. Run the full test suite on your branch
before submitting a pull request.

Update the documentation when creating or modifying features. Test
your documentation changes for clarity, concision, and correctness, as
well as a clean documentation build.

Pull requests descriptions should be as clear as possible and include a
reference to all the issues that they address.

Commit messages must start with a capitalized and short summary
written in the imperative, followed by an optional, more detailed
explanatory text which is separated from the summary by an empty line.

Code review comments may be added to your pull request. Discuss, then make the
suggested modifications and push additional commits to your feature branch. Be
sure to post a comment after pushing. The new commits will show up in the pull
request automatically, but the reviewers will not be notified unless you
comment.

Before the pull request is merged, make sure that you squash your commits into
logical units of work using `git rebase -i` and `git push -f`. After every
commit the test suite (if any) should be passing. Include documentation changes
in the same commit so that a revert would remove all traces of the feature or
fix.

Commits that fix or close an issue should include a reference like `Closes #XXX`
or `Fixes #XXX`, which will automatically close the issue when merged.

### Sign your work

The sign-off is a simple line at the end of the explanation for the
patch, which certifies that you wrote it or otherwise have the right to
pass it on as an open-source patch.  The rules are pretty simple: if you
can certify the below (from [developercertificate.org][]):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

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

    Signed-off-by: Joe Smith <joe@gmail.com>

using your real name (sorry, no pseudonyms or anonymous contributions.)

You can add the sign off when creating the git commit via `git commit -s`.

[code-of-conduct]: https://github.com/opencontainers/tob/blob/d2f9d68c1332870e40693fe077d311e0742bc73d/code-of-conduct.md
[developercertificate.org]: http://developercertificate.org/
[Freenode]: https://freenode.net/
[irc-logs]: http://ircbot.wl.linuxfoundation.org/eavesdrop/%23opencontainers/
[mailing-list]: https://groups.google.com/a/opencontainers.org/forum/#!forum/dev
[meeting.ics]: https://github.com/opencontainers/runtime-spec/blob/master/meeting.ics
[minutes]: http://ircbot.wl.linuxfoundation.org/meetings/opencontainers/
[rfc5545]: https://tools.ietf.org/html/rfc5545
[UberConference]: https://www.uberconference.com/opencontainers
