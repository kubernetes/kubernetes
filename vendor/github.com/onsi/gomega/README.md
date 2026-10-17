<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/gomega-dark.png">
  <img alt="Gomega: Ginkgo's Preferred Matcher Library" src="docs/images/gomega.png">
</picture>

[![test](https://github.com/onsi/gomega/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/onsi/gomega/actions/workflows/test.yml?query=branch%3Amaster) [![Gomega Docs](docs/images/docs-badge.svg)](https://onsi.github.io/gomega/) [![Sponsor onsi](docs/images/sponsor-badge.svg)](https://github.com/sponsors/onsi)

Jump straight to the [docs](http://onsi.github.io/gomega/) to learn about Gomega, including a list of [all available matchers](http://onsi.github.io/gomega/#provided-matchers).

If you have a question, comment, bug report, feature request, etc. please open a GitHub issue.

## Using Gomega with Claude Code

Gomega ships a set of [Claude Code](https://claude.com/claude-code) skills as a **plugin**, so an agent writing assertions in *your* suite has Gomega's idioms — the full matcher catalog, `Eventually`/`Consistently`, and the `gstruct`/`ghttp`/`gexec`/`gbytes`/`gleak`/`gmeasure` sub-libraries — on hand. The Gomega repo doubles as the plugin marketplace, so installation is two commands. From inside Claude Code:

```
/plugin marketplace add onsi/gomega
/plugin install gomega@gomega
```

(or non-interactively: `claude plugin marketplace add onsi/gomega` then `claude plugin install gomega@gomega`)

This installs a family of `gomega:*` skills that activate automatically while you write tests. See the [docs](http://onsi.github.io/gomega/#using-gomega-with-claude-code) for the full list.

## [Ginkgo](http://github.com/onsi/ginkgo): a BDD Testing Framework for Golang

Learn more about Ginkgo [here](http://onsi.github.io/ginkgo/)

## Community Matchers

A collection of community matchers is available on the [wiki](https://github.com/onsi/gomega/wiki).

## License

Gomega is MIT-Licensed

The `ConsistOf` matcher uses [goraph](https://github.com/amitkgupta/goraph) which is embedded in the source to simplify distribution.  goraph has an MIT license.
