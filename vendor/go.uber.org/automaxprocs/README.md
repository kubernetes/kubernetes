# automaxprocs [![GoDoc][doc-img]][doc] [![Build Status][ci-img]][ci] [![Coverage Status][cov-img]][cov]

Automatically set `GOMAXPROCS` to match Linux container CPU quota.

## Installation

`go get -u go.uber.org/automaxprocs`

## Quick Start

```go
import _ "go.uber.org/automaxprocs"

func main() {
  // Your application logic here.
}
```

# Performance
Data measured from Uber's internal load balancer. We ran the load balancer with 200% CPU quota (i.e., 2 cores):

| GOMAXPROCS         |  RPS      | P50 (ms) | P99.9 (ms) |
| ------------------ | --------- | -------- | ---------- |
| 1                  | 28,893.18 | 1.46     | 19.70      |
| 2 (equal to quota) | 44,715.07 | 0.84     | 26.38      |
| 3                  | 44,212.93 | 0.66     | 30.07      |
| 4                  | 41,071.15 | 0.57     | 42.94      |
| 8                  | 33,111.69 | 0.43     | 64.32      |
| Default (24)       | 22,191.40 | 0.45     | 76.19      |

When `GOMAXPROCS` is increased above the CPU quota, we see P50 decrease slightly, but see significant increases to P99. We also see that the total RPS handled also decreases.

When `GOMAXPROCS` is higher than the CPU quota allocated, we also saw significant throttling:

```
$ cat /sys/fs/cgroup/cpu,cpuacct/system.slice/[...]/cpu.stat
nr_periods 42227334
nr_throttled 131923
throttled_time 88613212216618
```

Once `GOMAXPROCS` was reduced to match the CPU quota, we saw no CPU throttling.

## Development Status: Stable

All APIs are finalized, and no breaking changes will be made in the 1.x series
of releases. Users of semver-aware dependency management systems should pin
automaxprocs to `^1`.

## Contributing

We encourage and support an active, healthy community of contributors &mdash;
including you! Details are in the [contribution guide](CONTRIBUTING.md) and
the [code of conduct](CODE_OF_CONDUCT.md). The automaxprocs maintainers keep
an eye on issues and pull requests, but you can also report any negative
conduct to oss-conduct@uber.com. That email list is a private, safe space;
even the automaxprocs maintainers don't have access, so don't hesitate to hold
us to a high standard.

<hr>

Released under the [MIT License](LICENSE).

[doc-img]: https://godoc.org/go.uber.org/automaxprocs?status.svg
[doc]: https://godoc.org/go.uber.org/automaxprocs
[ci-img]: https://github.com/uber-go/automaxprocs/actions/workflows/go.yml/badge.svg
[ci]: https://github.com/uber-go/automaxprocs/actions/workflows/go.yml
[cov-img]: https://codecov.io/gh/uber-go/automaxprocs/branch/master/graph/badge.svg
[cov]: https://codecov.io/gh/uber-go/automaxprocs


