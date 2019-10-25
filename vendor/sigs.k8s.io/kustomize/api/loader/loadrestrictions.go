// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package loader

import (
	"fmt"

	"github.com/spf13/pflag"
	"sigs.k8s.io/kustomize/api/filesys"
)

//go:generate stringer -type=loadRestrictions
type loadRestrictions int

const (
	unknown loadRestrictions = iota
	rootOnly
	none
)

const (
	flagName = "load_restrictor"
)

var (
	flagValue = rootOnly.String()
	flagHelp  = "if set to '" + none.String() +
		"', local kustomizations may load files from outside their root. " +
		"This does, however, break the relocatability of the kustomization."
)

func AddFlagLoadRestrictor(set *pflag.FlagSet) {
	set.StringVar(
		&flagValue, flagName,
		rootOnly.String(), flagHelp)
}

func ValidateFlagLoadRestrictor() (LoadRestrictorFunc, error) {
	switch flagValue {
	case rootOnly.String():
		return RestrictionRootOnly, nil
	case none.String():
		return RestrictionNone, nil
	default:
		return nil, fmt.Errorf(
			"illegal flag value --%s %s; legal values: %v",
			flagName, flagValue,
			[]string{rootOnly.String(), none.String()})
	}
}

type LoadRestrictorFunc func(
	filesys.FileSystem, filesys.ConfirmedDir, string) (string, error)

func RestrictionRootOnly(
	fSys filesys.FileSystem, root filesys.ConfirmedDir, path string) (string, error) {
	d, f, err := fSys.CleanedAbs(path)
	if err != nil {
		return "", err
	}
	if f == "" {
		return "", fmt.Errorf("'%s' must be a file", path)
	}
	if !d.HasPrefix(root) {
		return "", fmt.Errorf(
			"security; file '%s' is not in or below '%s'",
			path, root)
	}
	return d.Join(f), nil
}

func RestrictionNone(
	_ filesys.FileSystem, _ filesys.ConfirmedDir, path string) (string, error) {
	return path, nil
}
