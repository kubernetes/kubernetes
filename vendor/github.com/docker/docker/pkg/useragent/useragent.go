// Package useragent provides helper functions to pack
// version information into a single User-Agent header.
package useragent

import (
	"errors"
	"strings"
)

var (
	ErrNilRequest = errors.New("request cannot be nil")
)

// VersionInfo is used to model UserAgent versions.
type VersionInfo struct {
	Name    string
	Version string
}

func (vi *VersionInfo) isValid() bool {
	const stopChars = " \t\r\n/"
	name := vi.Name
	vers := vi.Version
	if len(name) == 0 || strings.ContainsAny(name, stopChars) {
		return false
	}
	if len(vers) == 0 || strings.ContainsAny(vers, stopChars) {
		return false
	}
	return true
}

// Convert versions to a string and append the string to the string base.
//
// Each VersionInfo will be converted to a string in the format of
// "product/version", where the "product" is get from the name field, while
// version is get from the version field. Several pieces of verson information
// will be concatinated and separated by space.
//
// Example:
// AppendVersions("base", VersionInfo{"foo", "1.0"}, VersionInfo{"bar", "2.0"})
// results in "base foo/1.0 bar/2.0".
func AppendVersions(base string, versions ...VersionInfo) string {
	if len(versions) == 0 {
		return base
	}

	verstrs := make([]string, 0, 1+len(versions))
	if len(base) > 0 {
		verstrs = append(verstrs, base)
	}

	for _, v := range versions {
		if !v.isValid() {
			continue
		}
		verstrs = append(verstrs, v.Name+"/"+v.Version)
	}
	return strings.Join(verstrs, " ")
}
