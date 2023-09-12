/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package verflag defines utility functions to handle command line flags
// related to version of Kubernetes.
package verflag

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	flag "github.com/spf13/pflag"

	"k8s.io/component-base/version"
)

type versionValue string

const (
	VersionFalse versionValue = "false"
	VersionTrue  versionValue = "true"
	VersionRaw   versionValue = "raw"
)

const strRawVersion string = "raw"

func (v *versionValue) IsBoolFlag() bool {
	return true
}

func (v *versionValue) Get() interface{} {
	return versionValue(*v)
}

func (v *versionValue) Set(s string) error {
	if s == strRawVersion {
		*v = VersionRaw
		return nil
	}

	if strings.HasPrefix(s, "v") {
		err := version.SetDynamicVersion(s)
		if err == nil {
			*v = versionValue(s)
		}
		return err
	}

	boolVal, err := strconv.ParseBool(s)
	if err == nil {
		if boolVal {
			*v = VersionTrue
		} else {
			*v = VersionFalse
		}
	}
	return err
}

func (v *versionValue) String() string {
	return string(*v)
}

// The type of the flag as required by the pflag.Value interface
func (v *versionValue) Type() string {
	return "version"
}

func VersionVar(p *versionValue, name string, value versionValue, usage string) {
	*p = value
	flag.Var(p, name, usage)
	// "--version" will be treated as "--version=true"
	flag.Lookup(name).NoOptDefVal = "true"
}

func Version(name string, value versionValue, usage string) *versionValue {
	p := new(versionValue)
	VersionVar(p, name, value, usage)
	return p
}

const versionFlagName = "version"

var (
	versionFlag = Version(versionFlagName, VersionFalse, "--version, --version=raw prints version information and quits; --version=vX.Y.Z... sets the reported version")
	programName = "Kubernetes"
)

// AddFlags registers this package's flags on arbitrary FlagSets, such that they point to the
// same value as the global flags.
func AddFlags(fs *flag.FlagSet) {
	fs.AddFlag(flag.Lookup(versionFlagName))
}

// variables for unit testing PrintAndExitIfRequested
var (
	output = io.Writer(os.Stdout)
	exit   = os.Exit
)

// PrintAndExitIfRequested will check if --version or --version=raw was passed
// and, if so, print the version and exit.
func PrintAndExitIfRequested() {
	if *versionFlag == VersionRaw {
		fmt.Fprintf(output, "%#v\n", version.Get())
		exit(0)
	} else if *versionFlag == VersionTrue {
		fmt.Fprintf(output, "%s %s\n", programName, version.Get())
		exit(0)
	}
}
