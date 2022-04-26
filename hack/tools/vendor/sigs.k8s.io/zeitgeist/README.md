# Zeitgeist

([/ˈzaɪtɡaɪst/](https://en.wikipedia.org/wiki/Help:IPA/English)) is a language-agnostic dependency checker that keeps track of external dependencies across your project and ensure they're up-to-date.

[![CircleCI](https://circleci.com/gh/Pluies/zeitgeist.svg?style=shield)](https://circleci.com/gh/Pluies/zeitgeist)
[![Go Report Card](https://goreportcard.com/badge/sigs.k8s.io/zeitgeist)](https://goreportcard.com/report/sigs.k8s.io/zeitgeist)
[![GoDoc](https://godoc.org/sigs.k8s.io/zeitgeist?status.svg)](https://godoc.org/sigs.k8s.io/zeitgeist)

- [Rationale](#rationale)
- [What is Zeitgeist](#what-is-zeitgeist)
- [When is Zeitgeist _not_ suggested](#when-is-zeitgeist-not-suggested)
- [Naming](#naming)
- [Releasing](#releasing)
- [Credit](#credit)
- [To do](#to-do)
- [Community, discussion, contribution, and support](#community-discussion-contribution-and-support)
  - [Code of conduct](#code-of-conduct)

## Rationale

More and more projects nowadays have external dependencies, and the best way to ensure stability and reproducibility is to pin these dependencies to a specific version.

However, this leads to a new problem: the world changes around us, and new versions of these dependencies are released _all the time_.

For a simple project with a couple of dependencies, a team can usually keep up to speed by following mailing lists or Slack channels, but for larger projects this becomes a daunting task.

This problem is pretty much solved by package managers in specific programming languages (see [_When is Zeitgeist _not_ suggested_](#when-is-zeitgeist-not-suggested) below), but it remains a big issue when:

- Your project relies on packages outside your programming language of choice
- You declare infrastructure-as-code, where the "build step" is usually bespoke and dependencies are managed manually
- Dependencies do not belong in a classical "package manager" (e.g. AMI images)

## What is Zeitgeist

Zeitgeist is a tool that takes a configuration file with a list of dependencies, and ensures that:

- These dependencies versions are consistent within your project
- These dependencies are up-to-date

A Zeitgeist configuration file (usually `dependencies.yaml`) is a list of _dependencies_, referenced in files, which may or may not have an _upstream_:

```yaml
dependencies:
- name: terraform
  version: 0.12.3
  upstream:
    flavour: github
    url: hashicorp/terraform
  refPaths:
  - path: helper-image/Dockerfile
    match: TERRAFORM_VERSION
- name: aws-eks-ami
  version: ami-09bbefc07310f7914
  scheme: random
  upstream:
    flavour: ami
    owner: amazon
    name: "amazon-eks-node-1.13-*"
  refPaths:
  - path: clusters.yaml
    match: workers_ami
- name: package
  version: 3.99.99
  upstream:
    flavour: gitlab
    url: honk/honk
    constraints: <3.0.0
```

Use `zeitgeist local` to verify that the dependency version is correct in all files referenced in _`refPaths`_:

![zeigeist local](/docs/local.png)

Use `zeitgeist validate` to also check with defined `upstreams` whether a new version is available for the given dependencies:

![zeigeist validate](/docs/validate.png)


When using `zeitgeist validate --remote` with flavour `github` or `gitlab` you will need to set some environment variables.

For `GitHub` you will need to set:

```console
export GITHUB_TOKEN=<YOUR_GITHUB_TOKEN>
```

For `GitLab`:

When using the public `GitLab` instance at https://gitlab.com/ :

```console
export GITLAB_TOKEN=<YOUR_GITLAB_TOKEN>
```

When using the self-hosted `GitLab` instance, ie. https://my-gitlab.company.com/ :

```console
export GITLAB_PRIVATE_TOKEN=<YOUR_GITLAB_PRIVATE_TOKEN>
```

You can use in the `dependencies.yaml` both public and private GitLab instances. The only limitation today is that
you can only use one private GitLab at the moment.

See the [full documentation](https://godoc.org/sigs.k8s.io/zeitgeist/dependencies#Dependency) to see configuration options.

## When is Zeitgeist _not_ suggested

While Zeitgeist aims to be a great cross-language solution for tracking external dependencies, it won't be as well integrated as native package managers.

If your project is mainly written in one single language with a well-known and supported package manager (e.g. [`npm`](https://www.npmjs.com/), [`maven`](https://maven.apache.org/), [`rubygems`](https://rubygems.org/), [`pip`](https://pypi.org/project/pip/), [`cargo`](https://crates.io/)...), you definitely should use your package manager rather than Zeitgeist.

## Naming

[Zeitgeist](https://en.wikipedia.org/wiki/Zeitgeist), a German compound word, can be translated as "spirit of the times" and refers to _a schema of fashions or fads which prescribes what is considered to be acceptable or tasteful for an era_.

## Releasing

Releases are generated with [goreleaser](https://goreleaser.com/).

```bash
git tag v0.0.0 # Use the correct version here
git push --tags
export GPG_TTY=$(tty)
goreleaser release --rm-dist
```

## Credit

Zeitgeist is inspired by [Kubernetes' script to manage external dependencies](https://groups.google.com/forum/?pli=1#!topic/kubernetes-dev/cTaYyb1a18I) and extended to include checking with upstream sources to ensure dependencies are up-to-date.

## To do

- [x] Find a good name for the project
- [ ] Support `helm` upstream
- [ ] Support `eks` upstream
- [x] Support `ami` upstream
- [ ] support `docker` upstream
- [x] Cleanly separate various upstreams to make it easy to add new upstreams
- [x] Implement non-semver support (e.g. for AMI, but also for classic releases)
- [x] Write good docs :)
- [x] Write good tests!
- [x] Externalise the project into its own repo
- [x] Generate releases
- [x] Automate release generation from a tag

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community page](http://kubernetes.io/community/).

You can reach the maintainers of this project at:

- [`#release-management`](https://kubernetes.slack.com/archives/CJH2GBF7Y) channel on [Kubernetes Slack](http://slack.k8s.io/)
- [Mailing List](https://groups.google.com/forum/#!forum/kubernetes-sig-release)

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
