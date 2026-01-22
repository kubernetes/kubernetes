# hnslib

This package provides the Golang interface for accessing Windows HCN APIs to manage entities within the Host Network Service (HNS), which serves as the server container networking component in Windows. While it is mainly utilized by the Windows KubeProxy component in Kubernetes, it is also available for use in other projects, such as Azure CNI, Windows CNI, Calico CNI, Flannel CNI, Azure NPM, and more.

## Building

This project is imported by KubeProxy, Kubelet etc. for building binaries.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Microsoft CLA](https://cla.microsoft.com).

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

We require that contributors sign their commits
to certify they either authored the work themselves or otherwise have permission to use it in this project.

We also require that contributors sign their commits using  using [`git commit --signoff`][git-commit-s]
to certify they either authored the work themselves or otherwise have permission to use it in this project.
A range of commits can be signed off using [`git rebase --signoff`][git-rebase-s].

Please see  [the developer certificate](https://developercertificate.org) for more info,
as well as to make sure that you can attest to the rules listed.
Our CI uses the [DCO Github app](https://github.com/apps/dco) to ensure that all commits in a given PR are signed-off.

### Linting

Code must pass a linting stage, which uses [`golangci-lint`][lint].
Since `./test` is a separate Go module, the linter is run from both the root and the
`test` directories. Additionally, the linter is run with `GOOS` set to both `windows` and
`linux`.

The linting settings are stored in [`.golangci.yaml`](./.golangci.yaml), and can be run
automatically with VSCode by adding the following to your workspace or folder settings:

```json
    "go.lintTool": "golangci-lint",
    "go.lintOnSave": "package",
```

Additional editor [integrations options are also available][lint-ide].

Alternatively, `golangci-lint` can be [installed][lint-install] and run locally:

```shell
# use . or specify a path to only lint a package
# to show all lint errors, use flags "--max-issues-per-linter=0 --max-same-issues=0"
> golangci-lint run
```

To run across the entire repo for both `GOOS=windows` and `linux`:

```powershell
> foreach ( $goos in ('windows', 'linux') ) {
    foreach ( $repo in ('.', 'test') ) {
        pwsh -Command "cd $repo && go env -w GOOS=$goos && golangci-lint.exe run --verbose"
    }
}
```

### Go Generate

The pipeline checks that auto-generated code, via `go generate`, are up to date.
Similar to the [linting stage](#linting), `go generate` is run in root Go modules.

This can be done via:

```shell
> go generate ./...
> cd test && go generate ./...
```

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Dependencies

This project requires Golang 1.22 or newer to build.

For system requirements to run this project, see the Microsoft docs on [Windows Container requirements](https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/system-requirements).

## Reporting Security Issues

Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).

For additional details, see [Report a Computer Security Vulnerability](https://technet.microsoft.com/en-us/security/ff852094.aspx) on Technet

---------------
Copyright (c) 2018 Microsoft Corp.  All rights reserved.

[lint]: https://golangci-lint.run/
[lint-ide]: https://golangci-lint.run/usage/integrations/#editor-integration
[lint-install]: https://golangci-lint.run/usage/install/#local-installation

[git-commit-s]: https://git-scm.com/docs/git-commit#Documentation/git-commit.txt--s
[git-rebase-s]: https://git-scm.com/docs/git-rebase#Documentation/git-rebase.txt---signoff
