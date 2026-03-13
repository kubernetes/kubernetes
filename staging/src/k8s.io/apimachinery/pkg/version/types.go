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

package version

// Info contains versioning information.
// TODO: Add []string of api versions supported? It's still unclear
// how we'll want to distribute that information.
type Info struct {
	// Major is the major version of the binary version
	Major string `json:"major"`
	// Minor is the minor version of the binary version
	Minor string `json:"minor"`
	// EmulationMajor is the major version of the emulation version
	EmulationMajor string `json:"emulationMajor,omitempty"`
	// EmulationMinor is the minor version of the emulation version
	EmulationMinor string `json:"emulationMinor,omitempty"`
	// MinCompatibilityMajor is the major version of the minimum compatibility version
	MinCompatibilityMajor string `json:"minCompatibilityMajor,omitempty"`
	// MinCompatibilityMinor is the minor version of the minimum compatibility version
	MinCompatibilityMinor string `json:"minCompatibilityMinor,omitempty"`
	GitVersion            string `json:"gitVersion"`
	GitCommit             string `json:"gitCommit"`
	GitTreeState          string `json:"gitTreeState"`
	BuildDate             string `json:"buildDate"`
	GoVersion             string `json:"goVersion"`
	Compiler              string `json:"compiler"`
	Platform              string `json:"platform"`
}

// String returns info as a human-friendly version string.
func (info Info) String() string {
	return info.GitVersion
}
