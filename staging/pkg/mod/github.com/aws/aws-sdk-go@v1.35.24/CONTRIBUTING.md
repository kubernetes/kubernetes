# Contributing to the AWS SDK for Go

Thank you for your interest in contributing to the AWS SDK for Go!
We work hard to provide a high-quality and useful SDK, and we greatly value
feedback and contributions from our community. Whether it's a bug report,
new feature, correction, or additional documentation, we welcome your issues
and pull requests. Please read through this document before submitting any
[issues] or [pull requests][pr] to ensure we have all the necessary information to
effectively respond to your bug report or contribution.

Jump To:

* [Bug Reports](#Bug-Reports)
* [Code Contributions](#Code-Contributions)

## How to contribute

*Before you send us a pull request, please be sure that:*

1. You're working from the latest source on the master branch.
2. You check existing open, and recently closed, pull requests to be sure 
   that someone else hasn't already addressed the problem.
3. You create an issue before working on a contribution that will take a 
   significant amount of your time.

*Creating a Pull Request*

1. Fork the repository.
2. In your fork, make your change in a branch that's based on this repo's master branch.
3. Commit the change to your fork, using a clear and descriptive commit message.
4. Create a pull request, answering any questions in the pull request form.

For contributions that will take a significant amount of time, open a new 
issue to pitch your idea before you get started. Explain the problem and 
describe the content you want to see added to the documentation. Let us know 
if you'll write it yourself or if you'd like us to help. We'll discuss your 
proposal with you and let you know whether we're likely to accept it.   

## Bug Reports

You can file bug reports against the SDK on the [GitHub issues][issues] page.

If you are filing a report for a bug or regression in the SDK, it's extremely
helpful to provide as much information as possible when opening the original
issue. This helps us reproduce and investigate the possible bug without having
to wait for this extra information to be provided. Please read the following
guidelines prior to filing a bug report.

1. Search through existing [issues][] to ensure that your specific issue has
   not yet been reported. If it is a common issue, it is likely there is
   already a bug report for your problem.

2. Ensure that you have tested the latest version of the SDK. Although you
   may have an issue against an older version of the SDK, we cannot provide
   bug fixes for old versions. It's also possible that the bug may have been
   fixed in the latest release.

3. Provide as much information about your environment, SDK version, and
   relevant dependencies as possible. For example, let us know what version
   of Go you are using, which and version of the operating system, and the
   environment your code is running in. e.g Container.

4. Provide a minimal test case that reproduces your issue or any error
   information you related to your problem. We can provide feedback much
   more quickly if we know what operations you are calling in the SDK. If
   you cannot provide a full test case, provide as much code as you can
   to help us diagnose the problem. Any relevant information should be provided
   as well, like whether this is a persistent issue, or if it only occurs
   some of the time.

## Code Contributions

We are always happy to receive code and documentation contributions to the SDK. 
Code contributions to the SDK are done through [Pull Requests][pr]. The list below are guidelines to use when submitting pull requests. These are the 
same set of guidelines that the core contributors use when submitting changes, and we ask the same of all community contributions as well:

1. The SDK is released under the [Apache license][license]. Any code you submit
   will be released under that license. For substantial contributions, we may
   ask you to sign a [Contributor License Agreement (CLA)][cla].

2. If you would like to implement support for a significant feature that is not
   yet available in the SDK, please talk to us beforehand to avoid any
   duplication of effort.

3. Wherever possible, pull requests should contain tests as appropriate.
   Bugfixes should contain tests that exercise the corrected behavior (i.e., the
   test should fail without the bugfix and pass with it), and new features
   should be accompanied by tests exercising the feature.

4. Pull requests that contain failing tests will not be merged until the test
   failures are addressed. Pull requests that cause a significant drop in the
   SDK's test coverage percentage are unlikely to be merged until tests have
   been added.

5. The JSON files under the SDK's `models` folder are sourced from outside the SDK.
   Such as `models/apis/ec2/2016-11-15/api.json`. We will not accept pull requests
   directly on these models. If you discover an issue with the models please
   create a [GitHub issue][issues] describing the issue.

### Testing

To run the tests locally, running the `make unit` command will `go get` the
SDK's testing dependencies, and run vet, link and unit tests for the SDK.

```
make unit
```

Standard go testing functionality is supported as well. To test SDK code that
is tagged with `codegen` you'll need to set the build tag in the go test
command. The `make unit` command will do this automatically.

```
go test -tags codegen ./private/...
```

See the `Makefile` for additional testing tags that can be used in testing.

To test on multiple platform the SDK includes several DockerFiles under the
`awstesting/sandbox` folder, and associated make recipes to execute
unit testing within environments configured for specific Go versions.

```
make sandbox-test-go18
```

To run all sandbox environments use the following make recipe

```
# Optionally update the Go tip that will be used during the batch testing
make update-aws-golang-tip

# Run all SDK tests for supported Go versions in sandboxes
make sandbox-test
```

In addition the sandbox environment include make recipes for interactive modes
so you can run command within the Docker container and context of the SDK.

```
make sandbox-go18
```

### Changelog Documents

You can see all release changes in the `CHANGELOG.md` file at the root of the
repository. The release notes added to this file will contain service client
updates, and major SDK changes. When submitting a pull request please include an entry in `CHANGELOG_PENDING.md` under the appropriate changelog type so your changelog entry is included on the following release.

#### Changelog Types

* `SDK Features` - For major additive features, internal changes that have
outward impact, or updates to the SDK foundations. This will result in a minor
version change.
* `SDK Enhancements` - For minor additive features or incremental sized changes.
This will result in a patch version change.
* `SDK Bugs` - For minor changes that resolve an issue. This will result in a
patch version change.

[issues]: https://github.com/aws/aws-sdk-go/issues
[pr]: https://github.com/aws/aws-sdk-go/pulls
[license]: http://aws.amazon.com/apache2.0/
[cla]: http://en.wikipedia.org/wiki/Contributor_License_Agreement
[releasenotes]: https://github.com/aws/aws-sdk-go/releases

