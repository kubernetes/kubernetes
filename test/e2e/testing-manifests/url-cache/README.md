The content of this directory is a copy of the files referenced via
URL in the E2E test suite with framework.testfiles.RegisterURLs.

Only ever edit files here for testing purposes! Such a change must
never get merged into Kubernetes.

Instead:
- update the files in the original location after testing the changes
- update the URLs
- run hack/update-url-cache.sh
