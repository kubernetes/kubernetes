# CERTIFICATE-TRANSPARENCY-GO Changelog

## v1.0.20 - Minimal Gossip / Go 1.11 Fix / Utility Improvements

Published 2018-07-05 09:21:34 +0000 UTC

Enhancements have been made to various utilities including `scanner`, `sctcheck`, `loglist` and `x509util`.

The `allow_verification_with_non_compliant_keys` flag has been removed from `signatures.go`.

An implementation of Gossip has been added. See the `gossip/minimal` package for more information.

An X.509 compatibility issue for Go 1.11 has been fixed. This should be backwards compatible with 1.10.

Commit [37a384cd035e722ea46e55029093e26687138edf](https://api.github.com/repos/google/certificate-transparency-go/commits/37a384cd035e722ea46e55029093e26687138edf) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.20)

## v1.0.19 - CTFE User Quota

Published 2018-06-01 13:51:52 +0000 UTC

CTFE now supports Trillian Log's explicit quota API; quota can be requested based on the remote user's IP, as well as per-issuing certificate in submitted chains.

Commit [8736a411b4ff214ea20687e46c2b67d66ebd83fc](https://api.github.com/repos/google/certificate-transparency-go/commits/8736a411b4ff214ea20687e46c2b67d66ebd83fc) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.19)

## v1.0.18 - Adding Migration Tool / Client Additions / K8 Config

Published 2018-06-01 14:28:20 +0000 UTC

Work on a log migration tool (Migrillian) is in progress. This is not yet ready for production use but will provide features for mirroring and migrating logs.

The `RequestLog` API allows for logging of SCTs when they are issued by CTFE.

The CT Go client now supports `GetEntryAndProof`. Utilities have been switched over to use the `glog` package.

Commit [77abf2dac5410a62c04ac1c662c6d0fa54afc2dc](https://api.github.com/repos/google/certificate-transparency-go/commits/77abf2dac5410a62c04ac1c662c6d0fa54afc2dc) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.18)

## v1.0.17 - Merkle verification / Tracing / Demo script / CORS

Published 2018-06-01 14:25:16 +0000 UTC

Now uses Merkle Tree verification from Trillian.

The CT server now supports CORS.

Request tracing added using OpenCensus. For GCE / K8 it just requires the flag to be enabled to export traces to Stackdriver. Other environments may differ.

A demo script was added that goes through setting up a simple deployment suitable for development / demo purposes. This may be useful for those new to the project.

Commit [3c3d22ce946447d047a03228ebb4a41e3e4eb15b](https://api.github.com/repos/google/certificate-transparency-go/commits/3c3d22ce946447d047a03228ebb4a41e3e4eb15b) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.17)

## v1.0.16 - Lifecycle test / Go 1.10.1

Published 2018-06-01 14:22:23 +0000 UTC

An integration test was added that goes through a create / drain queue / freeze lifecycle for a log.

Changes to `x509` were merged from Go 1.10.1.

Commit [a72423d09b410b80673fd1135ba1022d04bac6cd](https://api.github.com/repos/google/certificate-transparency-go/commits/a72423d09b410b80673fd1135ba1022d04bac6cd) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.16)

## v1.0.15 - More control of verification, grpclb, stackdriver metrics

Published 2018-06-01 14:20:32 +0000 UTC

Facilities were added to the `x509` package to control whether verification checks are applied.

Log server requests are now balanced using `gRPClb`.

For Kubernetes, metrics can be published to Stackdriver monitoring.

Commit [684d6eee6092774e54d301ccad0ed61bc8d010c1](https://api.github.com/repos/google/certificate-transparency-go/commits/684d6eee6092774e54d301ccad0ed61bc8d010c1) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.15)

## v1.0.14 - SQLite Removed, LeafHashForLeaf

Published 2018-06-01 14:15:37 +0000 UTC

Support for SQLlite was removed. This motivation was ongoing test flakiness caused by multi-user access. This database may work for an embedded scenario but is not suitable for use in a server environment.

A `LeafHashForLeaf` client API was added and is now used by the CT client and integration tests.

Commit [698cd6a661196db4b2e71437422178ffe8705006](https://api.github.com/repos/google/certificate-transparency-go/commits/698cd6a661196db4b2e71437422178ffe8705006) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.14)

## v1.0.13 - Crypto changes, util updates, sync with trillian repo, loglist verification

Published 2018-06-01 14:15:21 +0000 UTC

Some of our custom crypto package that were wrapping calls to the standard package have been removed and the base features used directly.

Updates were made to GCE ingress and health checks.

The log list utility can verify signatures.

Commit [480c3654a70c5383b9543ec784203030aedbd3a5](https://api.github.com/repos/google/certificate-transparency-go/commits/480c3654a70c5383b9543ec784203030aedbd3a5) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.13)

## v1.0.12 - Client / util updates & CTFE fixes

Published 2018-06-01 14:13:42 +0000 UTC

The CT client can now use a JSON loglist to find logs.

CTFE had a fix applied for preissued precerts.

A DNS client was added and CT client was extended to support DNS retrieval.

Commit [74c06c95e0b304a050a1c33764c8a01d653a16e3](https://api.github.com/repos/google/certificate-transparency-go/commits/74c06c95e0b304a050a1c33764c8a01d653a16e3) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.12)

## v1.0.11 - Kubernetes CI / Integration fixes

Published 2018-06-01 14:12:18 +0000 UTC

Updates to Kubernetes configs, mostly related to running a CI instance.

Commit [0856acca7e0ab7f082ae83a1fbb5d21160962efc](https://api.github.com/repos/google/certificate-transparency-go/commits/0856acca7e0ab7f082ae83a1fbb5d21160962efc) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.11)

## v1.0.10 - More scanner, x509, utility and client fixes. CTFE updates

Published 2018-06-01 14:09:47 +0000 UTC

The CT client was using the wrong protobuffer library package. To guard against this in future a check has been added to our lint config.

The `x509` and `asn1` packages have had upstream fixes applied from Go 1.10rc1.

Commit [1bec4527572c443752ad4f2830bef88be0533236](https://api.github.com/repos/google/certificate-transparency-go/commits/1bec4527572c443752ad4f2830bef88be0533236) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.10)

## v1.0.9 - Scanner, x509, utility and client fixes

Published 2018-06-01 14:11:13 +0000 UTC

The `scanner` utility now displays throughput stats.

Build instructions and README files were updated.

The `certcheck` utility can be told to ignore unknown critical X.509 extensions.

Commit [c06833528d04a94eed0c775104d1107bab9ae17c](https://api.github.com/repos/google/certificate-transparency-go/commits/c06833528d04a94eed0c775104d1107bab9ae17c) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.9)

## v1.0.8 - Client fixes, align with trillian repo

Published 2018-06-01 14:06:44 +0000 UTC



Commit [e8b02c60f294b503dbb67de0868143f5d4935e56](https://api.github.com/repos/google/certificate-transparency-go/commits/e8b02c60f294b503dbb67de0868143f5d4935e56) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.8)

## v1.0.7 - CTFE fixes

Published 2018-06-01 14:06:13 +0000 UTC

An issue was fixed with CTFE signature caching. In an unlikely set of circumstances this could lead to log mis-operation. While the chances of this are small, we recommend that versions prior to this one are not deployed.

Commit [52c0590bd3b4b80c5497005b0f47e10557425eeb](https://api.github.com/repos/google/certificate-transparency-go/commits/52c0590bd3b4b80c5497005b0f47e10557425eeb) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.7)

## v1.0.6 - crlcheck improvements / other fixes

Published 2018-06-01 14:04:22 +0000 UTC

The `crlcheck` utility has had several fixes and enhancements. Additionally the `hammer` now supports temporal logs.

Commit [3955e4a00c42e83ff17ce25003976159c5d0f0f9](https://api.github.com/repos/google/certificate-transparency-go/commits/3955e4a00c42e83ff17ce25003976159c5d0f0f9) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.6)

## v1.0.5 - X509 and asn1 fixes

Published 2018-06-01 14:02:58 +0000 UTC

This release is mostly fixes to the `x509` and `asn1` packages. Some command line utilties were also updated.

Commit [ae40d07cce12f1227c6e658e61c9dddb7646f97b](https://api.github.com/repos/google/certificate-transparency-go/commits/ae40d07cce12f1227c6e658e61c9dddb7646f97b) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.5)

## v1.0.4 - Multi log backend configs

Published 2018-06-01 14:02:07 +0000 UTC

Support was added to allow CTFE to use multiple backends, each serving a distinct set of logs. It allows for e.g. regional backend deployment with common frontend servers.

Commit [62023ed90b41fa40854957b5dec7d9d73594723f](https://api.github.com/repos/google/certificate-transparency-go/commits/62023ed90b41fa40854957b5dec7d9d73594723f) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.4)

## v1.0.3 - Hammer updates, use standard context

Published 2018-06-01 14:01:11 +0000 UTC

After the Go 1.9 migration references to anything other than the standard `context` package have been removed. This is the only one that should be used from now on.

Commit [b28beed8b9aceacc705e0ff4a11d435a310e3d97](https://api.github.com/repos/google/certificate-transparency-go/commits/b28beed8b9aceacc705e0ff4a11d435a310e3d97) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.3)

## v1.0.2 - Go 1.9

Published 2018-06-01 14:00:00 +0000 UTC

Go 1.9 is now required to build the code.

Commit [3aed33d672ee43f04b1e8a00b25ca3e2e2e74309](https://api.github.com/repos/google/certificate-transparency-go/commits/3aed33d672ee43f04b1e8a00b25ca3e2e2e74309) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.2)

## v1.0.1 - Hammer and client improvements

Published 2018-06-01 13:59:29 +0000 UTC



Commit [c28796cc21776667fb05d6300e32d9517be96515](https://api.github.com/repos/google/certificate-transparency-go/commits/c28796cc21776667fb05d6300e32d9517be96515) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0.1)

## v1.0 - First Trillian CT Release

Published 2018-06-01 13:59:00 +0000 UTC

This is the point that corresponds to the 1.0 release in the trillian repo.

Commit [abb79e468b6f3bbd48d1ab0c9e68febf80d52c4d](https://api.github.com/repos/google/certificate-transparency-go/commits/abb79e468b6f3bbd48d1ab0c9e68febf80d52c4d) Download [zip](https://api.github.com/repos/google/certificate-transparency-go/zipball/v1.0)

