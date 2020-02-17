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
	"os"
	"strconv"

	flag "github.com/spf13/pflag"

	k8sversion "k8s.io/component-base/version"
)

type versionValue string

const (
	versionFalse versionValue = "false"
	versionTrue  versionValue = "true"
	versionRaw   versionValue = "raw"
)

func (v *versionValue) IsBoolFlag() bool {
	return true
}

func (v *versionValue) Get() interface{} {
	return versionValue(*v)
}

func (v *versionValue) Set(s string) error {
	if s == string(versionRaw) {
		*v = versionRaw
		return nil
	}
	boolVal, err := strconv.ParseBool(s)
	if boolVal {
		*v = versionTrue
	} else {
		*v = versionFalse
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

func versionVar(p *versionValue, name string, value versionValue, usage string) {
	*p = value
	flag.Var(p, name, usage)
	// "--version" will be treated as "--version=true"
	flag.Lookup(name).NoOptDefVal = "true"
}

func version(name string, value versionValue, usage string) *versionValue {
	p := new(versionValue)
	versionVar(p, name, value, usage)
	return p
}

const versionFlagName = "version"

var (
	versionFlag = version(versionFlagName, versionFalse, "Print version information and quit")
)

// AddFlags registers this package's flags on arbitrary FlagSets, such that they point to the
// same value as the global flags.
func AddFlags(fs *flag.FlagSet) {
	fs.AddFlag(flag.Lookup(versionFlagName))
}

// PrintAndExitIfRequested will check if the -version flag was passed
// and, if so, print the version and exit.
func PrintAndExitIfRequested() {
	switch *versionFlag {
	case versionRaw:
		fmt.Printf("%#v\n", k8sversion.Get())
		os.Exit(0)
	case versionTrue:
		fmt.Printf("Kubernetes %s\n", k8sversion.Get())
		os.Exit(0)
	}
}
