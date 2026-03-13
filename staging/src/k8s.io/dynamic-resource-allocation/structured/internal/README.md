The allocator code exists in three variants which get picked depending on which
features are enabled:
- stable: for a "GA only" feature configuration, minimal changes going forward
- incubating: the default implementation, adds support for beta features
- experimental: under active development, including alpha features

This structure serves as a safety net because experimental changes cannot break
more stable Kubernetes configurations, something that happened already once
despite careful reviews.

The goal is to rotate the implementations wholesale instead of copying
individual code chunks, i.e. at some point "incubating" replaces "stable",
"experimental" replaces "incubating", and "experimental" becomes a copy of
"incubating" until new changes get added to it again.

Ideally changes should be limited to "experimental", but sometimes changes have
to be applied the same way across different variants, for example bug fixes
or changes to the package API.

Tests are shared between all implementations, with test cases applied depending
on what features they require. Further testing is covered by
test/integration/scheduler_perf. When promoting implementations, the selection
of implementations which support certain features there needs to be
updated. For example `[]string{"experimental"}` in `TestSchedulerPerf` of
`test/integration/scheduler_perf/dra/consumablecapacity/consumablecapacity_test.go`
will eventually become `[]string{"incubating", "experimental"}`. The explicit
selection of the implementation for benchmarking in
`EnableAllocators("experimental")` then becomes `EnableAllocators("incubating")`.
