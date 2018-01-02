package auth

import (
	"net/http"
	"strings"
)

// APIVersion represents a version of an API including its
// type and version number.
type APIVersion struct {
	// Type refers to the name of a specific API specification
	// such as "registry"
	Type string

	// Version is the version of the API specification implemented,
	// This may omit the revision number and only include
	// the major and minor version, such as "2.0"
	Version string
}

// String returns the string formatted API Version
func (v APIVersion) String() string {
	return v.Type + "/" + v.Version
}

// APIVersions gets the API versions out of an HTTP response using the provided
// version header as the key for the HTTP header.
func APIVersions(resp *http.Response, versionHeader string) []APIVersion {
	versions := []APIVersion{}
	if versionHeader != "" {
		for _, supportedVersions := range resp.Header[http.CanonicalHeaderKey(versionHeader)] {
			for _, version := range strings.Fields(supportedVersions) {
				versions = append(versions, ParseAPIVersion(version))
			}
		}
	}
	return versions
}

// ParseAPIVersion parses an API version string into an APIVersion
// Format (Expected, not enforced):
// API version string = <API type> '/' <API version>
// API type = [a-z][a-z0-9]*
// API version = [0-9]+(\.[0-9]+)?
// TODO(dmcgowan): Enforce format, add error condition, remove unknown type
func ParseAPIVersion(versionStr string) APIVersion {
	idx := strings.IndexRune(versionStr, '/')
	if idx == -1 {
		return APIVersion{
			Type:    "unknown",
			Version: versionStr,
		}
	}
	return APIVersion{
		Type:    strings.ToLower(versionStr[:idx]),
		Version: versionStr[idx+1:],
	}
}
