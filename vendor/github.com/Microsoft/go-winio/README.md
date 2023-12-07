# go-winio [![Build Status](https://github.com/microsoft/go-winio/actions/workflows/ci.yml/badge.svg)](https://github.com/microsoft/go-winio/actions/workflows/ci.yml)

This repository contains utilities for efficiently performing Win32 IO operations in
Go. Currently, this is focused on accessing named pipes and other file handles, and
for using named pipes as a net transport.

This code relies on IO completion ports to avoid blocking IO on system threads, allowing Go
to reuse the thread to schedule another goroutine. This limits support to Windows Vista and
newer operating systems. This is similar to the implementation of network sockets in Go's net
package.

Please see the LICENSE file for licensing information.

## Contributing

This project welcomes contributions and suggestions.
Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that
you have the right to, and actually do, grant us the rights to use your contribution.
For details, visit [Microsoft CLA](https://cla.microsoft.com).

When you submit a pull request, a CLA-bot will automatically determine whether you need to
provide a CLA and decorate the PR appropriately (e.g., label, comment).
Simply follow the instructions provided by the bot.
You will only need to do this once across all repos using our CLA.

Additionally, the pull request pipeline requires the following steps to be performed before
mergining.

### Code Sign-Off

We require that contributors sign their commits using [`git commit --signoff`][git-commit-s]
to certify they either authored the work themselves or otherwise have permission to use it in this project.

A range of commits can be signed off using [`git rebase --signoff`][git-rebase-s].

Please see [the developer certificate](https://developercertificate.org) for more info,
as well as to make sure that you can attest to the rules listed.
Our CI uses the DCO Github app to ensure that all commits in a given PR are signed-off.

### Linting

Code must pass a linting stage, which uses [`golangci-lint`][lint].
The linting settings are stored in [`.golangci.yaml`](./.golangci.yaml), and can be run
automatically with VSCode by adding the following to your workspace or folder settings:

```json
    "go.lintTool": "golangci-lint",
    "go.lintOnSave": "package",
```

Additional editor [integrations options are also available][lint-ide].

Alternatively, `golangci-lint` can be [installed locally][lint-install] and run from the repo root:

```shell
# use . or specify a path to only lint a package
# to show all lint errors, use flags "--max-issues-per-linter=0 --max-same-issues=0"
> golangci-lint run ./...
```

### Go Generate

The pipeline checks that auto-generated code, via `go generate`, are up to date.

This can be done for the entire repo:

```shell
> go generate ./...
```

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Special Thanks

Thanks to [natefinch][natefinch] for the inspiration for this library.
See [npipe](https://github.com/natefinch/npipe) for another named pipe implementation.

[lint]: https://golangci-lint.run/
[lint-ide]: https://golangci-lint.run/usage/integrations/#editor-integration
[lint-install]: https://golangci-lint.run/usage/install/#local-installation

[git-commit-s]: https://git-scm.com/docs/git-commit#Documentation/git-commit.txt--s
[git-rebase-s]: https://git-scm.com/docs/git-rebase#Documentation/git-rebase.txt---signoff

[natefinch]: https://github.com/natefinch
