// +build linux

package volume

import (
	"strings"
	"testing"
)

func TestParseMountRawPropagation(t *testing.T) {
	var (
		valid   []string
		invalid map[string]string
	)

	valid = []string{
		"/hostPath:/containerPath:shared",
		"/hostPath:/containerPath:rshared",
		"/hostPath:/containerPath:slave",
		"/hostPath:/containerPath:rslave",
		"/hostPath:/containerPath:private",
		"/hostPath:/containerPath:rprivate",
		"/hostPath:/containerPath:ro,shared",
		"/hostPath:/containerPath:ro,slave",
		"/hostPath:/containerPath:ro,private",
		"/hostPath:/containerPath:ro,z,shared",
		"/hostPath:/containerPath:ro,Z,slave",
		"/hostPath:/containerPath:Z,ro,slave",
		"/hostPath:/containerPath:slave,Z,ro",
		"/hostPath:/containerPath:Z,slave,ro",
		"/hostPath:/containerPath:slave,ro,Z",
		"/hostPath:/containerPath:rslave,ro,Z",
		"/hostPath:/containerPath:ro,rshared,Z",
		"/hostPath:/containerPath:ro,Z,rprivate",
	}
	invalid = map[string]string{
		"/path:/path:ro,rshared,rslave":   `invalid mode`,
		"/path:/path:ro,z,rshared,rslave": `invalid mode`,
		"/path:shared":                    "invalid volume specification",
		"/path:slave":                     "invalid volume specification",
		"/path:private":                   "invalid volume specification",
		"name:/absolute-path:shared":      "invalid volume specification",
		"name:/absolute-path:rshared":     "invalid volume specification",
		"name:/absolute-path:slave":       "invalid volume specification",
		"name:/absolute-path:rslave":      "invalid volume specification",
		"name:/absolute-path:private":     "invalid volume specification",
		"name:/absolute-path:rprivate":    "invalid volume specification",
	}

	for _, path := range valid {
		if _, err := ParseMountRaw(path, "local"); err != nil {
			t.Fatalf("ParseMountRaw(`%q`) should succeed: error %q", path, err)
		}
	}

	for path, expectedError := range invalid {
		if _, err := ParseMountRaw(path, "local"); err == nil {
			t.Fatalf("ParseMountRaw(`%q`) should have failed validation. Err %v", path, err)
		} else {
			if !strings.Contains(err.Error(), expectedError) {
				t.Fatalf("ParseMountRaw(`%q`) error should contain %q, got %v", path, expectedError, err.Error())
			}
		}
	}
}
