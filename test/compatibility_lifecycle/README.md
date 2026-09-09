This directory contains commands for [compatibility lifecycle verification](https://github.com/kubernetes/enhancements/blob/master/keps/sig-architecture/4330-compatibility-versions/README.md)

Currently, the following commands are implemented:
```
# Verify feature gate list is up to date
go run test/compatibility_lifecycle/main.go feature-gates verify

# Update feature gate list
go run test/compatibility_lifecycle/main.go feature-gates update
```
