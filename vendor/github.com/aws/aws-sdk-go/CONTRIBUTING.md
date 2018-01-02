Contributing to the AWS SDK for Go

We work hard to provide a high-quality and useful SDK, and we greatly value
feedback and contributions from our community. Whether it's a bug report,
new feature, correction, or additional documentation, we welcome your issues
and pull requests. Please read through this document before submitting any
issues or pull requests to ensure we have all the necessary information to
effectively respond to your bug report or contribution.


## Filing Bug Reports

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
   the environment your code is running in. e.g Container.

4. Provide a minimal test case that reproduces your issue or any error
   information you related to your problem. We can provide feedback much
   more quickly if we know what operations you are calling in the SDK. If
   you cannot provide a full test case, provide as much code as you can
   to help us diagnose the problem. Any relevant information should be provided
   as well, like whether this is a persistent issue, or if it only occurs
   some of the time.


## Submitting Pull Requests

We are always happy to receive code and documentation contributions to the SDK.
Please be aware of the following notes prior to opening a pull request:

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
   create a Github [issue](issues) describing the issue.

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
`awstesting/sandbox` folder, and associated make recipes to to execute
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

### Changelog

You can see all release changes in the `CHANGELOG.md` file at the root of the
repository. The release notes added to this file will contain service client
updates, and major SDK changes.

[issues]: https://github.com/aws/aws-sdk-go/issues
[pr]: https://github.com/aws/aws-sdk-go/pulls
[license]: http://aws.amazon.com/apache2.0/
[cla]: http://en.wikipedia.org/wiki/Contributor_License_Agreement
[releasenotes]: https://github.com/aws/aws-sdk-go/releases

