// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package provenance

import (
	"fmt"
	"runtime"
	"strings"
)

var (
	version = "unknown"
	// sha1 from git, output of $(git rev-parse HEAD)
	gitCommit = "$Format:%H$"
	// build date in ISO8601 format, output of $(date -u +'%Y-%m-%dT%H:%M:%SZ')
	buildDate = "1970-01-01T00:00:00Z"
	goos      = runtime.GOOS
	goarch    = runtime.GOARCH
)

// Provenance holds information about the build of an executable.
type Provenance struct {
	// Version of the kustomize binary.
	Version string `json:"version,omitempty"`
	// GitCommit is a git commit
	GitCommit string `json:"gitCommit,omitempty"`
	// BuildDate is date of the build.
	BuildDate string `json:"buildDate,omitempty"`
	// GoOs holds OS name.
	GoOs string `json:"goOs,omitempty"`
	// GoArch holds architecture name.
	GoArch string `json:"goArch,omitempty"`
}

// GetProvenance returns an instance of Provenance.
func GetProvenance() Provenance {
	return Provenance{
		version,
		gitCommit,
		buildDate,
		goos,
		goarch,
	}
}

// Full returns the full provenance stamp.
func (v Provenance) Full() string {
	return fmt.Sprintf("%+v", v)
}

// Short returns the shortened provenance stamp.
func (v Provenance) Short() string {
	return fmt.Sprintf(
		"%v",
		Provenance{
			Version:   v.Version,
			BuildDate: v.BuildDate,
		})
}

// Semver returns the semantic version of kustomize.
// kustomize version is set in format "kustomize/vX.X.X" in every release.
// X.X.X is a semver. If the version string is not in this format,
// return the original version string
func (v Provenance) Semver() string {
	return strings.TrimPrefix(v.Version, "kustomize/")
}
