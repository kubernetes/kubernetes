// Package aaparser is a convenience package interacting with `apparmor_parser`.
package aaparser

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

const (
	binary = "apparmor_parser"
)

// GetVersion returns the major and minor version of apparmor_parser.
func GetVersion() (int, error) {
	output, err := cmd("", "--version")
	if err != nil {
		return -1, err
	}

	return parseVersion(output)
}

// LoadProfile runs `apparmor_parser -Kr` on a specified apparmor profile to
// replace the profile. The `-K` is necessary to make sure that apparmor_parser
// doesn't try to write to a read-only filesystem.
func LoadProfile(profilePath string) error {
	_, err := cmd("", "-Kr", profilePath)
	return err
}

// cmd runs `apparmor_parser` with the passed arguments.
func cmd(dir string, arg ...string) (string, error) {
	c := exec.Command(binary, arg...)
	c.Dir = dir

	output, err := c.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("running `%s %s` failed with output: %s\nerror: %v", c.Path, strings.Join(c.Args, " "), output, err)
	}

	return string(output), nil
}

// parseVersion takes the output from `apparmor_parser --version` and returns
// a representation of the {major, minor, patch} version as a single number of
// the form MMmmPPP {major, minor, patch}.
func parseVersion(output string) (int, error) {
	// output is in the form of the following:
	// AppArmor parser version 2.9.1
	// Copyright (C) 1999-2008 Novell Inc.
	// Copyright 2009-2012 Canonical Ltd.

	lines := strings.SplitN(output, "\n", 2)
	words := strings.Split(lines[0], " ")
	version := words[len(words)-1]

	// split by major minor version
	v := strings.Split(version, ".")
	if len(v) == 0 || len(v) > 3 {
		return -1, fmt.Errorf("parsing version failed for output: `%s`", output)
	}

	// Default the versions to 0.
	var majorVersion, minorVersion, patchLevel int

	majorVersion, err := strconv.Atoi(v[0])
	if err != nil {
		return -1, err
	}

	if len(v) > 1 {
		minorVersion, err = strconv.Atoi(v[1])
		if err != nil {
			return -1, err
		}
	}
	if len(v) > 2 {
		patchLevel, err = strconv.Atoi(v[2])
		if err != nil {
			return -1, err
		}
	}

	// major*10^5 + minor*10^3 + patch*10^0
	numericVersion := majorVersion*1e5 + minorVersion*1e3 + patchLevel
	return numericVersion, nil
}
