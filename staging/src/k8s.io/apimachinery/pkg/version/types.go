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
// See https://github.com/kubernetes/community/blob/master/contributors/design-proposals/release/versioning.md
// TODO: Add []string of api versions supported? It's still unclear
// how we'll want to distribute that information.
type Info struct {
	Major        string `json:"major"`        // major version, always numeric
	Minor        string `json:"minor"`        // minor version, numeric possibly followed by "+"
	GitVersion   string `json:"gitVersion"`   // semantic version, vX.Y.Z possibly followed by "beta", etc.
	GitCommit    string `json:"gitCommit"`    // sha1 from git
	GitTreeState string `json:"gitTreeState"` // state of git tree, either "clean" or "dirty"
	BuildDate    string `json:"buildDate"`    // build date in ISO8601 format
	GoVersion    string `json:"goVersion"`
	Compiler     string `json:"compiler"`
	Platform     string `json:"platform"`
}

// String returns info as a human-friendly version string.
func (info Info) String() string {
	return info.GitVersion
}
