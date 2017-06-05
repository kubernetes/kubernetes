#!/usr/bin/env bash

# A script for updating godep dependencies for the vendored directory /cmd/
# without pulling in etcd itself as a dependency.

rm -rf Godeps vendor
ln -s cmd/vendor vendor
godep save ./...
rm -rf cmd/Godeps
rm vendor
mv Godeps cmd/

