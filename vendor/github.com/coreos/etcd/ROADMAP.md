#  etcd roadmap

**work in progress**

This document defines a high level roadmap for etcd development.

The dates below should not be considered authoritative, but rather indicative of the projected timeline of the project. The [milestones defined in GitHub](https://github.com/coreos/etcd/milestones) represent the most up-to-date and issue-for-issue plans.

etcd 2.3 is our current stable branch. The roadmap below outlines new features that will be added to etcd, and while subject to change, define what future stable will look like.

### etcd 3.0 (April)
- v3 API ([see also the issue tag](https://github.com/coreos/etcd/issues?utf8=%E2%9C%93&q=label%3Aarea/v3api))
    - Leases
    - Binary protocol
    - Support a large number of watchers
    - Failure guarantees documented
-  Simple v3 client (golang)
- v3 API
    - Locking
- Better disk backend
    - Improved write throughput
    - Support larger datasets and histories
- Simpler disaster recovery UX
- Integrated with Kubernetes
- Mirroring

### etcd 3.1 (July)
- API bindings for other languages

### etcd 3.+ (future)
- Horizontally scalable proxy layer
