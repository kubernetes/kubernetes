## Contributing

Thank you for your interest in go-toml! We appreciate you considering
contributing to go-toml!

The main goal is the project is to provide an easy-to-use TOML
implementation for Go that gets the job done and gets out of your way –
dealing with TOML is probably not the central piece of your project.

As the single maintainer of go-toml, time is scarce. All help, big or
small, is more than welcomed!

### Ask questions

Any question you may have, somebody else might have it too. Always feel
free to ask them on the [issues tracker][issues-tracker].  We will try to
answer them as clearly and quickly as possible, time permitting.

Asking questions also helps us identify areas where the documentation needs
improvement, or new features that weren't envisioned before. Sometimes, a
seemingly innocent question leads to the fix of a bug. Don't hesitate and
ask away!

### Improve the documentation

The best way to share your knowledge and experience with go-toml is to
improve the documentation. Fix a typo, clarify an interface, add an
example, anything goes!

The documentation is present in the [README][readme] and thorough the
source code. On release, it gets updated on [pkg.go.dev][pkg.go.dev]. To make a
change to the documentation, create a pull request with your proposed
changes. For simple changes like that, the easiest way to go is probably
the "Fork this project and edit the file" button on Github, displayed at
the top right of the file. Unless it's a trivial change (for example a
typo), provide a little bit of context in your pull request description or
commit message.

### Report a bug

Found a bug! Sorry to hear that :(. Help us and other track them down and
fix by reporting it. [File a new bug report][bug-report] on the [issues
tracker][issues-tracker]. The template should provide enough guidance on
what to include. When in doubt: add more details! By reducing ambiguity and
providing more information, it decreases back and forth and saves everyone
time.

### Code changes

Want to contribute a patch? Very happy to hear that!

First, some high-level rules:

* A short proposal with some POC code is better than a lengthy piece of
  text with no code. Code speaks louder than words.
* No backward-incompatible patch will be accepted unless discussed.
  Sometimes it's hard, and Go's lack of versioning by default does not
  help, but we try not to break people's programs unless we absolutely have
  to.
* If you are writing a new feature or extending an existing one, make sure
  to write some documentation.
* Bug fixes need to be accompanied with regression tests.
* New code needs to be tested.
* Your commit messages need to explain why the change is needed, even if
  already included in the PR description.

It does sound like a lot, but those best practices are here to save time
overall and continuously improve the quality of the project, which is
something everyone benefits from.

#### Get started

The fairly standard code contribution process looks like that:

1. [Fork the project][fork].
2. Make your changes, commit on any branch you like.
3. [Open up a pull request][pull-request]
4. Review, potential ask for changes.
5. Merge. You're in!

Feel free to ask for help! You can create draft pull requests to gather
some early feedback!

#### Run the tests

You can run tests for go-toml using Go's test tool: `go test ./...`.
When creating a pull requests, all tests will be ran on Linux on a few Go
versions (Travis CI), and on Windows using the latest Go version
(AppVeyor).

#### Style

Try to look around and follow the same format and structure as the rest of
the code. We enforce using `go fmt` on the whole code base.

---

### Maintainers-only

#### Merge pull request

Checklist:

* Passing CI.
* Does not introduce backward-incompatible changes (unless discussed).
* Has relevant doc changes.
* Has relevant unit tests.

1. Merge using "squash and merge".
2. Make sure to edit the commit message to keep all the useful information
   nice and clean.
3. Make sure the commit title is clear and contains the PR number (#123).

#### New release

1. Go to [releases][releases]. Click on "X commits to master since this
   release".
2. Make note of all the changes. Look for backward incompatible changes,
   new features, and bug fixes.
3. Pick the new version using the above and semver.
4. Create a [new release][new-release].
5. Follow the same format as [1.1.0][release-110].

[issues-tracker]: https://github.com/pelletier/go-toml/issues
[bug-report]: https://github.com/pelletier/go-toml/issues/new?template=bug_report.md
[pkg.go.dev]: https://pkg.go.dev/github.com/pelletier/go-toml
[readme]: ./README.md
[fork]: https://help.github.com/articles/fork-a-repo
[pull-request]: https://help.github.com/en/articles/creating-a-pull-request
[releases]: https://github.com/pelletier/go-toml/releases
[new-release]: https://github.com/pelletier/go-toml/releases/new
[release-110]: https://github.com/pelletier/go-toml/releases/tag/v1.1.0
