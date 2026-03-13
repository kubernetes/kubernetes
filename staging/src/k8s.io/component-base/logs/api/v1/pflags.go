/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/pflag"
)

// VModuleConfigurationPflag implements the pflag.Value interface for a
// VModuleConfiguration. The value pointer must not be nil.
func VModuleConfigurationPflag(value *VModuleConfiguration) pflag.Value {
	return vmoduleConfigurationPFlag{value}
}

type vmoduleConfigurationPFlag struct {
	value *VModuleConfiguration
}

// String returns the -vmodule parameter (comma-separated list of pattern=N).
func (wrapper vmoduleConfigurationPFlag) String() string {
	if wrapper.value == nil {
		return ""
	}
	var patterns []string
	for _, item := range *wrapper.value {
		patterns = append(patterns, fmt.Sprintf("%s=%d", item.FilePattern, item.Verbosity))
	}
	return strings.Join(patterns, ",")
}

// Set parses the -vmodule parameter (comma-separated list of pattern=N).
func (wrapper vmoduleConfigurationPFlag) Set(value string) error {
	// This code mirrors https://github.com/kubernetes/klog/blob/9ad246211af1ed84621ee94a26fcce0038b69cd1/klog.go#L287-L313

	for _, pat := range strings.Split(value, ",") {
		if len(pat) == 0 {
			// Empty strings such as from a trailing comma can be ignored.
			continue
		}
		patLev := strings.Split(pat, "=")
		if len(patLev) != 2 || len(patLev[0]) == 0 || len(patLev[1]) == 0 {
			return fmt.Errorf("%q does not have the pattern=N format", pat)
		}
		pattern := patLev[0]
		// 31 instead of 32 to ensure that it also fits into int32.
		v, err := strconv.ParseUint(patLev[1], 10, 31)
		if err != nil {
			return fmt.Errorf("parsing verbosity in %q: %v", pat, err)
		}
		*wrapper.value = append(*wrapper.value, VModuleItem{FilePattern: pattern, Verbosity: VerbosityLevel(v)})
	}
	return nil
}

func (wrapper vmoduleConfigurationPFlag) Type() string {
	return "pattern=N,..."
}

// VerbosityLevelPflag implements the pflag.Value interface for a verbosity
// level value.
func VerbosityLevelPflag(value *VerbosityLevel) pflag.Value {
	return verbosityLevelPflag{value}
}

type verbosityLevelPflag struct {
	value *VerbosityLevel
}

func (wrapper verbosityLevelPflag) String() string {
	if wrapper.value == nil {
		return "0"
	}
	return strconv.FormatInt(int64(*wrapper.value), 10)
}

func (wrapper verbosityLevelPflag) Get() interface{} {
	if wrapper.value == nil {
		return VerbosityLevel(0)
	}
	return *wrapper.value
}

func (wrapper verbosityLevelPflag) Set(value string) error {
	// Limited to int32 for compatibility with klog.
	v, err := strconv.ParseUint(value, 10, 31)
	if err != nil {
		return err
	}
	*wrapper.value = VerbosityLevel(v)
	return nil
}

func (wrapper verbosityLevelPflag) Type() string {
	return "Level"
}
