# Package: generated

## Purpose
This package is the designated destination for all generated code in Kubernetes. It serves as a container for auto-generated files to keep them organized and separate from hand-written code.

## Contents

- **openapi/**: Generated OpenAPI definitions for all Kubernetes API types

## Design Notes

- Package exists primarily for organizational purposes
- Generated files are typically created by code generators (openapi-gen, etc.)
- Not all generated files in Kubernetes currently use this package, but the plan is to consolidate them here
- The doc.go file documents the package's purpose
