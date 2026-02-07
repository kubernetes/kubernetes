// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

// Package swag contains a bunch of helper functions for go-openapi and go-swagger projects.
//
// You may also use it standalone for your projects.
//
// NOTE: all features that used to be exposed as package-level members (constants, variables,
// functions and types) are now deprecated and are superseded by equivalent features in
// more specialized sub-packages.
// Moving forward, no additional feature will be added to the [swag] API directly at the root package level,
// which remains there for backward-compatibility purposes.
//
// Child modules will continue to evolve or some new ones may be added in the future.
//
// # Modules
//
//   - [cmdutils]      utilities to work with CLIs
//
//   - [conv]          type conversion utilities
//
//   - [fileutils]     file utilities
//
//   - [jsonname]      JSON utilities
//
//   - [jsonutils]     JSON utilities
//
//   - [loading]       file loading
//
//   - [mangling]      safe name generation
//
//   - [netutils]      networking utilities
//
//   - [stringutils]   `string` utilities
//
//   - [typeutils]     `go` types utilities
//
//   - [yamlutils]     YAML utilities
//
// # Dependencies
//
// This repo has a few dependencies outside of the standard library:
//
//   - YAML utilities depend on [go.yaml.in/yaml/v3]
package swag

//go:generate mockery
