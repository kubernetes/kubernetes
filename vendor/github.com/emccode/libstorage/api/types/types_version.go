package types

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"
)

// VersionInfo provides information about the libStorage version.
type VersionInfo struct {

	// SemVer is the semantic version string
	SemVer string

	// ShaLong is the commit hash from which this package was built
	ShaLong string

	// BuildTimestamp is the UTC timestamp for when this package was built.
	BuildTimestamp time.Time

	// Branch is the branch name from which this package was built
	Branch string

	// Arch is the OS-Arch string of the system on which this package is
	// supported.
	Arch string
}

// String returns the version information as a string.
func (v *VersionInfo) String() string {
	buf := &bytes.Buffer{}
	fmt.Fprintf(buf, "SemVer: %s\n", v.SemVer)
	fmt.Fprintf(buf, "OsArch: %s\n", v.Arch)
	fmt.Fprintf(buf, "Branch: %s\n", v.Branch)
	fmt.Fprintf(buf, "Commit: %s\n", v.ShaLong)
	fmt.Fprintf(buf, "Formed: %s\n", v.BuildTimestamp.Format(time.RFC1123))
	return buf.String()
}

// MarshalJSON returns the JSON representation of the version.
func (v *VersionInfo) MarshalJSON() ([]byte, error) {

	ver := &struct {
		SemVer         string `json:"semver"`
		ShaLong        string `json:"shaLong"`
		BuildTimestamp int64  `json:"buildTimestamp"`
		Branch         string `json:"branch"`
		Arch           string `json:"arch"`
	}{
		SemVer:         v.SemVer,
		ShaLong:        v.ShaLong,
		BuildTimestamp: v.BuildTimestamp.Unix(),
		Branch:         v.Branch,
		Arch:           v.Arch,
	}

	return json.Marshal(ver)
}

// MarshalYAML returns the object to marshal to the YAML representation of the
// version.
func (v *VersionInfo) MarshalYAML() (interface{}, error) {

	return &struct {
		SemVer         string `yaml:"semver"`
		ShaLong        string `yaml:"shaLong"`
		BuildTimestamp int64  `yaml:"buildTimestamp"`
		Branch         string `yaml:"branch"`
		Arch           string `yaml:"arch"`
	}{
		SemVer:         v.SemVer,
		ShaLong:        v.ShaLong,
		BuildTimestamp: v.BuildTimestamp.Unix(),
		Branch:         v.Branch,
		Arch:           v.Arch,
	}, nil
}
