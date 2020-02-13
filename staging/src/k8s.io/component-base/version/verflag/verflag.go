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
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	flag "github.com/spf13/pflag"

	"k8s.io/component-base/version"
	"sigs.k8s.io/yaml"
)

type versionValue int

const (
	VersionFalse versionValue = iota
	VersionTrue
	VersionRaw
	VersionJson
	VersionYaml
)

const (
	strTrueVersion string = "true"
	strRawVersion  string = "raw"
	strJsonVersion string = "json"
	strYamlVersion string = "yaml"
)

func (v *versionValue) IsBoolFlag() bool {
	return true
}

func (v *versionValue) Get() interface{} {
	return versionValue(*v)
}

func (v *versionValue) Set(s string) error {
	switch s {
	case strRawVersion:
		*v = VersionRaw
	case strJsonVersion:
		*v = VersionJson
	case strYamlVersion:
		*v = VersionYaml
	default:
		boolVal, err := strconv.ParseBool(s)
		if boolVal {
			*v = VersionTrue
		} else {
			*v = VersionFalse
		}
		return err
	}
	return nil
}

func (v *versionValue) String() string {
	switch *v {
	case VersionTrue:
		return strTrueVersion
	case VersionRaw:
		return strRawVersion
	case VersionJson:
		return strJsonVersion
	case VersionYaml:
		return strYamlVersion
	}
	return "false"
}

// The type of the flag as required by the pflag.Value interface
func (v *versionValue) Type() string {
	return "version"
}

func VersionVar(p *versionValue, name string, value versionValue, usage string) {
	*p = value
	flag.Var(p, name, usage)
	// "--version" will be treated as "--version=true"
	flag.Lookup(name).NoOptDefVal = strTrueVersion
}

func Version(name string, value versionValue, usage string) *versionValue {
	p := new(versionValue)
	VersionVar(p, name, value, usage)
	return p
}

const versionFlagName = "version"

var (
	versionFlag = Version(versionFlagName, VersionFalse,
		fmt.Sprintf("Print version information and quit. Can be set to %q, %q, %q or %q", strTrueVersion, strRawVersion, strJsonVersion, strYamlVersion))
)

// AddFlags registers this package's flags on arbitrary FlagSets, such that they point to the
// same value as the global flags.
func AddFlags(fs *flag.FlagSet) {
	fs.AddFlag(flag.Lookup(versionFlagName))
}

// PrintAndExitIfRequested will check if the -version flag was passed
// and, if so, print the version and exit.
func PrintAndExitIfRequested() {
	ver := version.Get()
	verStr := ""

	switch *versionFlag {
	case VersionTrue:
		verStr = fmt.Sprintf("Kubernetes %s", ver)
	case VersionRaw:
		verStr = fmt.Sprintf("%#v", ver)
	case VersionJson:
		bytes, err := json.MarshalIndent(&ver, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to marshal version data: %v\n", err)
			os.Exit(1)
		}
		verStr = string(bytes)
	case VersionYaml:
		bytes, err := yaml.Marshal(&ver)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to marshal version data: %v\n", err)
			os.Exit(1)
		}
		verStr = string(bytes)
	}

	if verStr != "" {
		fmt.Println(verStr)
		os.Exit(0)
	}
}
