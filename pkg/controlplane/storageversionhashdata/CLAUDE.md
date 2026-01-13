# Package: storageversionhashdata

## Purpose
This package provides test data containing storage version hashes for Kubernetes API resources. These hashes are used to verify that storage versions remain stable across releases unless intentionally changed.

## Key Variables

- **NoStorageVersionHash**: Set of resources that legitimately have no storage version hash (virtual resources like tokenreviews, subjectaccessreviews)
- **GVRToStorageVersionHash**: Map of GroupVersionResource strings to their expected storage version hashes

## Resources Without Storage Hash

Resources that are non-persisted (virtual) or create-only:
- v1/bindings
- v1/componentstatuses
- authentication.k8s.io/v1/tokenreviews
- authorization.k8s.io/v1/*subjectaccessreviews

## Design Notes

- Storage version hashes are base64-encoded and should remain stable
- Package is explicitly marked as test-only
- Hash changes indicate storage version changes requiring migration
- Covers core resources (pods, services, configmaps) and all API groups
- Used by integration tests to detect unintended storage version changes
