# Contributing guidelines

## Getting started

### Sign Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles. Follow the [CLA instructions](https://github.com/kubernetes/community/blob/master/CLA.md) for how to sign.

### Setting up development environment

The [development guide](https://github.com/kubernetes/community/blob/master/contributors/devel/development.md#building-kubernetes-on-a-local-osshell-environment) will help you to install requirements, set up your workspace and build binaries.

### Test your setup

*This only works if you already built binaries.*

```bash
cd $GOPATH/src/k8s.io/kubernetes
cluster/kubectl.sh
# if everything went well you are ready to run kubectl commands
```
### Learn about resource package

[Read resource package documentation](https://godoc.org/k8s.io/kubernetes/pkg/kubectl/resource).

## Contribution process

1. Pick up an issue or submit a [design proposal](https://github.com/kubernetes/community/blob/master/sig-cli/CONTRIBUTING.md#design-proposals).  
Comment on the issue that you would like to work on it, be sure to mention the author of the issue as well as `@seans3` and `@mengqiy`.
1. Fork the [`kubernetes/kubernetes`](https://github.com/kubernetes/kubernetes) repo, clone your fork and create a new branch.  
See [development workflow](https://github.com/kubernetes/community/blob/master/contributors/devel/development.md#workflow) for detailed instructions.
1. Develop and [test your code changes](https://github.com/kubernetes/community/blob/master/contributors/devel/development.md#test).  
Ensure that you follow the [code conventions](https://github.com/kubernetes/community/blob/master/contributors/devel/coding-conventions.md#code-conventions) to avoid common go style mistakes.
1. Submit a pull request.  
Follow the [pull request template](https://github.com/kubernetes/kubernetes/blob/master/.github/PULL_REQUEST_TEMPLATE.md).  
[Add release notes](https://github.com/kubernetes/kubernetes/blob/master/.github/PULL_REQUEST_TEMPLATE.md) if needed.  
See [pull request process](https://github.com/kubernetes/community/blob/master/contributors/devel/pull-requests.md) for best practices and additional information.
1. [Get a code review](https://github.com/kubernetes/community/blob/master/contributors/devel/development.md#get-a-code-review).  
Pull requests can be merged until at least one `/LGTM` by a member or colaborator, this indicate that a PR is acceptable.

## How to contribute?

### Improving documentation

Write `doc.go` with package overview and examples or add code comments to document existing types and functions with their purpose and usage.  
See [`godoc`](https://blog.golang.org/godoc-documenting-go-code) documentation tool for learn how to generate reference documentation from source code comments.

### Writing test

Write test for functions that are missing unit/integration tests or improve test coverage.  
See [testing conventions](https://github.com/kubernetes/community/blob/master/contributors/devel/coding-conventions.md#testing-conventions), and [testing guide](https://github.com/kubernetes/community/blob/master/contributors/devel/testing.md) for more details.

### Fixing bugs

Filter issue search for `is:open is:issue no:assignee label:sig/cli` and see if anything sounds interesting.  
Ensure that you follow the [code conventions](https://github.com/kubernetes/community/blob/master/contributors/devel/coding-conventions.md#code-conventions) to avoid common go style mistakes.

### Working on a new feature

It's necessary to [make a design proposal](https://github.com/kubernetes/community/blob/master/sig-cli/CONTRIBUTING.md#design-proposals) for propose a new feature and get it approved by a lead.  
Include test and if your patch depends on new packages, [add that package](https://github.com/kubernetes/community/blob/master/contributors/devel/godep.md) with [`godep`](https://github.com/tools/godep).  

## Special notes 

Build binaries can take several time, if you only want to build kubectl code run:

```bash
cd $GOPATH/src/k8s.io/kubernetes
make all WHAT=cmd/kubectl
```

## Support channels

* [**Mailing list**](https://groups.google.com/forum/#!forum/kubernetes-sig-cli)
* [**Slack**](http://slack.k8s.io/) join #sig-cli
    * [Slack archive](https://kubernetes.slackarchive.io/sig-cli/)