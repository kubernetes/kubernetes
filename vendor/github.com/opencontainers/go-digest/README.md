# go-digest

[![GoDoc](https://godoc.org/github.com/opencontainers/go-digest?status.svg)](https://godoc.org/github.com/opencontainers/go-digest) [![Go Report Card](https://goreportcard.com/badge/github.com/opencontainers/go-digest)](https://goreportcard.com/report/github.com/opencontainers/go-digest) [![Build Status](https://travis-ci.org/opencontainers/go-digest.svg?branch=master)](https://travis-ci.org/opencontainers/go-digest)

Common digest package used across the container ecosystem.

Please see the [godoc](https://godoc.org/github.com/opencontainers/go-digest) for more information.

# What is a digest?

A digest is just a [hash](https://en.wikipedia.org/wiki/Hash_function).

The most common use case for a digest is to create a content identifier for use in [Content Addressable Storage](https://en.wikipedia.org/wiki/Content-addressable_storage) systems:

```go
id := digest.FromBytes([]byte("my content"))
```

In the example above, the id can be used to uniquely identify the byte slice "my content".
This allows two disparate applications to agree on a verifiable identifier without having to trust one another.

An identifying digest can be verified, as follows:

```go
if id != digest.FromBytes([]byte("my content")) {
  return errors.New("the content has changed!")
}
```

A `Verifier` type can be used to handle cases where an `io.Reader` makes more sense:

```go
rd := getContent()
verifier := id.Verifier()
io.Copy(verifier, rd)

if !verifier.Verified() {
  return errors.New("the content has changed!")
}
```

Using [Merkle DAGs](https://en.wikipedia.org/wiki/Merkle_tree), this can power a rich, safe, content distribution system.

# Usage

While the [godoc](https://godoc.org/github.com/opencontainers/go-digest) is considered the best resource, a few important items need to be called out when using this package.

1. Make sure to import the hash implementations into your application or the package will panic.
    You should have something like the following in the main (or other entrypoint) of your application:
   
    ```go
    import (
        _ "crypto/sha256"
        _ "crypto/sha512"
    )
    ```
    This may seem inconvenient but it allows you replace the hash 
    implementations with others, such as https://github.com/stevvooe/resumable.
 
2. Even though `digest.Digest` may be assemblable as a string, _always_ verify your input with `digest.Parse` or use `Digest.Validate` when accepting untrusted input.
    While there are measures to avoid common problems, this will ensure you have valid digests in the rest of your application.

3. While alternative encodings of hash values (digests) are possible (for example, base64), this package deals exclusively with hex-encoded digests.

# Stability

The Go API, at this stage, is considered stable, unless otherwise noted.

As always, before using a package export, read the [godoc](https://godoc.org/github.com/opencontainers/go-digest).

# Contributing

This package is considered fairly complete.
It has been in production in thousands (millions?) of deployments and is fairly battle-hardened.
New additions will be met with skepticism.
If you think there is a missing feature, please file a bug clearly describing the problem and the alternatives you tried before submitting a PR.

## Code of Conduct

Participation in the OpenContainers community is governed by [OpenContainer's Code of Conduct][code-of-conduct].

## Security

If you find an issue, please follow the [security][security] protocol to report it.

# Copyright and license

Copyright © 2019, 2020 OCI Contributors
Copyright © 2016 Docker, Inc.
All rights reserved, except as follows.
Code is released under the [Apache 2.0 license](LICENSE).
This `README.md` file and the [`CONTRIBUTING.md`](CONTRIBUTING.md) file are licensed under the Creative Commons Attribution 4.0 International License under the terms and conditions set forth in the file [`LICENSE.docs`](LICENSE.docs).
You may obtain a duplicate copy of the same license, titled CC BY-SA 4.0, at http://creativecommons.org/licenses/by-sa/4.0/.

[security]: https://github.com/opencontainers/org/blob/master/security
[code-of-conduct]: https://github.com/opencontainers/org/blob/master/CODE_OF_CONDUCT.md
