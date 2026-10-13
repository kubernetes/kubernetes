// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package builtinconfig provides legacy methods for
// configuring builtin plugins from a common config file.
// As a user, its best to configure plugins individually
// with plugin config files specified in the `transformers:`
// or `generators:` field, than to use this legacy
// configuration technique.
package builtinconfig
