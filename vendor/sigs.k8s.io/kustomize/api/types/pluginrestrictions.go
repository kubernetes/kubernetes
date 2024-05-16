// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// Some plugin classes
// - builtin: plugins defined in the kustomize repo.
//   May be freely used and re-configured.
// - local: plugins that aren't builtin but are
//   locally defined (presumably by the user), meaning
//   the kustomization refers to them via a relative
//   file path, not a URL.
// - remote: require a build-time download to obtain.
//   Unadvised, unless one controls the
//   serving site.
//
//go:generate stringer -type=PluginRestrictions
type PluginRestrictions int

const (
	PluginRestrictionsUnknown PluginRestrictions = iota

	// Non-builtin plugins completely disabled.
	PluginRestrictionsBuiltinsOnly

	// No restrictions, do whatever you want.
	PluginRestrictionsNone
)

// BuiltinPluginLoadingOptions distinguish ways in which builtin plugins are used.
//go:generate stringer -type=BuiltinPluginLoadingOptions
type BuiltinPluginLoadingOptions int

const (
	BploUndefined BuiltinPluginLoadingOptions = iota

	// Desired in production use for performance.
	BploUseStaticallyLinked

	// Desired in testing and development cycles where it's undesirable
	// to generate static code.
	BploLoadFromFileSys
)

// FnPluginLoadingOptions set way functions-based plugins are restricted
type FnPluginLoadingOptions struct {
	// Allow to run executables
	EnableExec bool
	// Allow to run starlark
	EnableStar bool
	// Allow container access to network
	Network     bool
	NetworkName string
	// list of mounts
	Mounts []string
	// list of env variables to pass to fn
	Env []string
	// Run as uid and gid of the command executor
	AsCurrentUser bool
	// Run in this working directory
	WorkingDir string
}
