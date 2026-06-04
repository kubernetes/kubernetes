// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package provenance

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/blang/semver/v4"
)

// These variables are set at build time using ldflags.
//
//nolint:gochecknoglobals
var (
	// During a release, this will be set to the release tag, e.g. "kustomize/v4.5.7"
	version = developmentVersion
	// build date in ISO8601 format, output of $(date -u +'%Y-%m-%dT%H:%M:%SZ')
	buildDate = unknown
)

const (
	// This default value, (devel), matches
	// the value debug.BuildInfo uses for an unset main module version.
	developmentVersion = "(devel)"

	// ModulePath is kustomize module path, defined in kustomize/go.mod
	ModulePath = "sigs.k8s.io/kustomize/kustomize/v5"

	// This is default value, unknown, substituted when
	// the value can't be determined from debug.BuildInfo.
	unknown = "unknown"
)

// Provenance holds information about the build of an executable.
type Provenance struct {
	// Version of the kustomize binary.
	Version string `json:"version,omitempty" yaml:"version,omitempty"`
	// GitCommit is a git commit
	GitCommit string `json:"gitCommit,omitempty" yaml:"gitCommit,omitempty"`
	// BuildDate is date of the build.
	BuildDate string `json:"buildDate,omitempty" yaml:"buildDate,omitempty"`
	// GoOs holds OS name.
	GoOs string `json:"goOs,omitempty" yaml:"goOs,omitempty"`
	// GoArch holds architecture name.
	GoArch string `json:"goArch,omitempty" yaml:"goArch,omitempty"`
	// GoVersion holds Go version.
	GoVersion string `json:"goVersion,omitempty" yaml:"goVersion,omitempty"`
}

// GetProvenance returns an instance of Provenance.
func GetProvenance() Provenance {
	p := Provenance{
		BuildDate: buildDate,
		Version:   version,
		GitCommit: unknown,
		GoOs:      runtime.GOOS,
		GoArch:    runtime.GOARCH,
		GoVersion: runtime.Version(),
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return p
	}

	for _, setting := range info.Settings {
		// For now, the git commit is the only information of interest.
		// We could consider adding other info such as the commit date in the future.
		if setting.Key == "vcs.revision" {
			p.GitCommit = setting.Value
			break
		}
	}
	p.Version = FindVersion(info, p.Version)

	return p
}

// FindVersion searches for a version in the depth of dependencies including replacements,
// otherwise, it tries to get version from debug.BuildInfo Main.
func FindVersion(info *debug.BuildInfo, version string) string {
	for _, dep := range info.Deps {
		if dep != nil && dep.Path == ModulePath {
			if dep.Version == developmentVersion {
				continue
			}
			v, err := GetMostRecentTag(*dep)
			if err != nil {
				fmt.Printf("failed to get most recent tag for %s: %v\n", dep.Path, err)
				continue
			}

			return v
		}
	}

	if version == developmentVersion && info.Main.Version != "" {
		return info.Main.Version
	}

	return version
}

func GetMostRecentTag(m debug.Module) (string, error) {
	for m.Replace != nil {
		m = *m.Replace
	}

	split := strings.Split(m.Version, "-")
	sv, err := semver.Parse(strings.TrimPrefix(split[0], "v"))

	if err != nil {
		return "", fmt.Errorf("failed to parse version %s: %w", m.Version, err)
	}

	if len(split) > 1 && sv.Patch > 0 {
		sv.Patch -= 1
	}
	return fmt.Sprintf("v%s", sv.FinalizeVersion()), nil
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
