<!--[metadata]>
+++
draft = true
+++
<![end-metadata]-->

# Distribution API Implementations

This is a list of known implementations of the Distribution API spec.

## [Docker Distribution Registry](https://github.com/docker/distribution)

Docker distribution is the reference implementation of the distribution API
specification. It aims to fully implement the entire specification.

### Releases
#### 2.0.1 (_in development_)
Implements API 2.0.1

_Known Issues_
 - No resumable push support
 - Content ranges ignored
 - Blob upload status will always return a starting range of 0

#### 2.0.0
Implements API 2.0.0

_Known Issues_
 - No resumable push support
 - No PATCH implementation for blob upload
 - Content ranges ignored

