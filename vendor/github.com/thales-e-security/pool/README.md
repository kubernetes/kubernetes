pool is a copy of a few packages from https://github.com/vitessio/vitess.

Vitess has some useful Go packages, however they are not versioned with Go modules,
which causes issues (e.g. https://github.com/ThalesIgnite/crypto11/issues/56). They
are also buried inside a large project, which forms a heavyweight dependency.

This package exposes the resource pool implementation and some of the atomic types.
