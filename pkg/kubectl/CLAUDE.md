# Package: kubectl

## Purpose
This package provides the functions used by the kubectl command line tool. The code is kept in this package (rather than in cmd/kubectl) to better support unit testing. The main() method for kubectl is only an entry point and should contain no functionality.

## Contents

- **cmd/convert/**: Implementation of the `kubectl convert` command

## Design Notes

- Package exists to separate kubectl logic from the entry point
- Enables better unit testing of kubectl functionality
- Most kubectl command implementations are in k8s.io/kubectl/pkg/cmd
- This package holds commands that require access to internal Kubernetes types
