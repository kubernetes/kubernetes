# Changelog

## 2.6.0 (2017-01-18)

#### Storage
- S3: fixed bug in delete due to read-after-write inconsistency
- S3: allow EC2 IAM roles to be used when authorizing region endpoints
- S3: add Object ACL Support
- S3: fix delete method's notion of subpaths
- S3: use multipart upload API in `Move` method for performance
- S3: add v2 signature signing for legacy S3 clones
- Swift: add simple heuristic to detect incomplete DLOs during read ops
- Swift: support different user and tenant domains
- Swift: bulk deletes in chunks
- Aliyun OSS: fix delete method's notion of subpaths
- Aliyun OSS: optimize data copy after upload finishes
- Azure: close leaking response body
- Fix storage drivers dropping non-EOF errors when listing repositories
- Compare path properly when listing repositories in catalog
- Add a foreign layer URL host whitelist
- Improve catalog enumerate runtime

#### Registry
- Export `storage.CreateOptions` in top-level package
- Enable notifications to endpoints that use self-signed certificates
- Properly validate multi-URL foreign layers
- Add control over validation of URLs in pushed manifests
- Proxy mode: fix socket leak when pull is cancelled
- Tag service: properly handle error responses on HEAD request
- Support for custom authentication URL in proxying registry
- Add configuration option to disable access logging
- Add notification filtering by target media type
- Manifest: `References()` returns all children
- Honor `X-Forwarded-Port` and Forwarded headers
- Reference: Preserve tag and digest in With* functions
- Add policy configuration for enforcing repository classes

#### Client
- Changes the client Tags `All()` method to follow links
- Allow registry clients to connect via HTTP2
- Better handling of OAuth errors in client
  
#### Spec
- Manifest: clarify relationship between urls and foreign layers
- Authorization: add support for repository classes

#### Manifest
- Override media type returned from `Stat()` for existing manifests
- Add plugin mediatype to distribution manifest

#### Docs
- Document `TOOMANYREQUESTS` error code
- Document required Let's Encrypt port
- Improve documentation around implementation of OAuth2
- Improve documentation for configuration

#### Auth
- Add support for registry type in scope
- Add support for using v2 ping challenges for v1
- Add leeway to JWT `nbf` and `exp` checking
- htpasswd: dynamically parse htpasswd file
- Fix missing auth headers with PATCH HTTP request when pushing to default port

#### Dockerfile
- Update to go1.7
- Reorder Dockerfile steps for better layer caching

#### Notes

Documentation has moved to the documentation repository at
`github.com/docker/docker.github.io/tree/master/registry`

The registry is go 1.7 compliant, and passes newer, more restrictive `lint` and `vet` ing.


## 2.5.0 (2016-06-14)

#### Storage
- Ensure uploads directory is cleaned after upload is committed
- Add ability to cap concurrent operations in filesystem driver
- S3: Add 'us-gov-west-1' to the valid region list
- Swift: Handle ceph not returning Last-Modified header for HEAD requests
- Add redirect middleware

#### Registry
- Add support for blobAccessController middleware
- Add support for layers from foreign sources
- Remove signature store
- Add support for Let's Encrypt
- Correct yaml key names in configuration

#### Client
- Add option to get content digest from manifest get

#### Spec
- Update the auth spec scope grammar to reflect the fact that hostnames are optionally supported
- Clarify API documentation around catalog fetch behavior

#### API
- Support returning HTTP 429 (Too Many Requests)

#### Documentation
- Update auth documentation examples to show "expires in" as int

#### Docker Image
- Use Alpine Linux as base image


