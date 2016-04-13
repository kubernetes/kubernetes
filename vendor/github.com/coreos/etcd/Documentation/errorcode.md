# Error Code
======

This document describes the error code used in key space '/v2/keys'. Feel free to import 'github.com/coreos/etcd/error' to use.

It's categorized into four groups:

- Command Related Error

| name                 | code | strerror              |
|----------------------|------|-----------------------|
| EcodeKeyNotFound     | 100  | "Key not found"       |
| EcodeTestFailed      | 101  | "Compare failed"      |
| EcodeNotFile         | 102  | "Not a file"          |
| EcodeNotDir          | 104  | "Not a directory"     |
| EcodeNodeExist       | 105  | "Key already exists"  |
| EcodeRootROnly       | 107  | "Root is read only"   |
| EcodeDirNotEmpty     | 108  | "Directory not empty" |

- Post Form Related Error

| name                     | code | strerror |
|--------------------------|------|------------------------------------------------|
| EcodePrevValueRequired   | 201  | "PrevValue is Required in POST form"           |
| EcodeTTLNaN              | 202  | "The given TTL in POST form is not a number"   |
| EcodeIndexNaN            | 203  | "The given index in POST form is not a number" |
| EcodeInvalidField        | 209  | "Invalid field"                                |
| EcodeInvalidForm         | 210  | "Invalid POST form"                            |

- Raft Related Error

| name              | code | strerror                 |
|-------------------|------|--------------------------|
| EcodeRaftInternal | 300  | "Raft Internal Error"    |
| EcodeLeaderElect  | 301  | "During Leader Election" |

- Etcd Related Error

| name                    | code | strerror                                               |
|-------------------------|------|--------------------------------------------------------|
| EcodeWatcherCleared     | 400  | "watcher is cleared due to etcd recovery"              |
| EcodeEventIndexCleared  | 401  | "The event in requested index is outdated and cleared" |
