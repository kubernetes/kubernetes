<!--[metadata]>
+++
title = "registry Compatibility"
description = "describes get by digest pitfall"
keywords = ["registry, manifest, images, tags, repository, distribution, digest"]
+++
<![end-metadata]-->

# Registry Compatibility

## Synopsis
*If a manifest is pulled by _digest_ from a registry 2.3 with Docker Engine 1.9
and older, and the manifest was pushed with Docker Engine 1.10, a security check
will cause the Engine to receive a manifest it cannot use and the pull will fail.*

## Registry Manifest Support

Historically, the registry has supported a [single manifest type](https://github.com/docker/distribution/blob/master/docs/spec/manifest-v2-1.md)
known as _Schema 1_.

With the move toward multiple architecture images the distribution project
introduced two new manifest types:  Schema 2 manifests and manifest lists.  The
registry 2.3 supports all three manifest types and in order to be compatible
with older Docker engines will, in certain cases, do an on-the-fly
transformation of a manifest before serving the JSON in the response.

This conversion has some implications for pulling manifests by digest and this
document enumerate these implications.


## Content Addressable Storage (CAS)

Manifests are stored and retrieved in the registry by keying off a digest
representing a hash of the contents.  One of the advantages provided by CAS is
security: if the contents are changed, then the digest will no longer match.
This prevents any modification of the manifest by a MITM attack or an untrusted
third party.

When a manifest is stored by the registry, this digest is returned in the HTTP
response headers and, if events are configured, delivered within the event.  The
manifest can either be retrieved by the tag, or this digest.

For registry versions 2.2.1 and below, the registry will always store and
serve _Schema 1_ manifests.  The Docker Engine 1.10 will first
attempt to send a _Schema 2_ manifest, falling back to sending a
Schema 1 type manifest when it detects that the registry does not
support the new version.


## Registry v2.3

### Manifest Push with Docker 1.9 and Older

The Docker Engine will construct a _Schema 1_ manifest which the
registry will persist to disk.

When the manifest is pulled by digest or tag with any docker version, a
_Schema 1_ manifest will be returned.

### Manifest Push with Docker 1.10

The docker engine will construct a _Schema 2_ manifest which the
registry will persist to disk.

When the manifest is pulled by digest or tag with Docker Engine 1.10, a
_Schema 2_ manifest will be returned.  The Docker Engine 1.10
understands the new manifest format.

When the manifest is pulled by *tag* with Docker Engine 1.9 and older, the
manifest is converted on-the-fly to _Schema 1_ and sent in the
response.  The Docker Engine 1.9 is compatible with this older format.

*When the manifest is pulled by _digest_ with Docker Engine 1.9 and older, the
same rewriting process will not happen in the registry.  If this were to happen
the digest would no longer match the hash of the manifest and would violate the
constraints of CAS.*

For this reason if a manifest is pulled by _digest_ from a registry 2.3 with Docker
Engine 1.9 and older, and the manifest was pushed with Docker Engine 1.10, a
security check will cause the Engine to receive a manifest it cannot use and the
pull will fail.