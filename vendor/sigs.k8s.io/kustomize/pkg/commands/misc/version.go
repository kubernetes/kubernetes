/*
Copyright 2018 The Kubernetes Authors.

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

package misc

import (
	"fmt"
	"io"
	"runtime"

	"github.com/spf13/cobra"
)

var (
	kustomizeVersion = "unknown"
	goos             = runtime.GOOS
	goarch           = runtime.GOARCH
	gitCommit        = "$Format:%H$" // sha1 from git, output of $(git rev-parse HEAD)

	buildDate = "1970-01-01T00:00:00Z" // build date in ISO8601 format, output of $(date -u +'%Y-%m-%dT%H:%M:%SZ')
)

// version returns the version of kustomize.
type version struct {
	// KustomizeVersion is a kustomize binary version.
	KustomizeVersion string `json:"kustomizeVersion"`
	// GitCommit is a git commit
	GitCommit string `json:"gitCommit"`
	// BuildDate is a build date of the binary.
	BuildDate string `json:"buildDate"`
	// GoOs holds OS name.
	GoOs string `json:"goOs"`
	// GoArch holds architecture name.
	GoArch string `json:"goArch"`
}

// getVersion returns version.
func getVersion() version {
	return version{
		kustomizeVersion,
		gitCommit,
		buildDate,
		goos,
		goarch,
	}
}

// Print prints version.
func (v version) Print(w io.Writer) {
	fmt.Fprintf(w, "Version: %+v\n", v)
}

// NewCmdVersion makes version command.
func NewCmdVersion(w io.Writer) *cobra.Command {
	return &cobra.Command{
		Use:     "version",
		Short:   "Prints the kustomize version",
		Example: `kustomize version`,
		Run: func(cmd *cobra.Command, args []string) {
			getVersion().Print(w)
		},
	}
}
