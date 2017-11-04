package autorest

import (
	"bytes"
	"fmt"
	"strings"
	"sync"
)

const (
	major = 8
	minor = 0
	patch = 0
	tag   = ""
)

var once sync.Once
var version string

// Version returns the semantic version (see http://semver.org).
func Version() string {
	once.Do(func() {
		semver := fmt.Sprintf("%d.%d.%d", major, minor, patch)
		verBuilder := bytes.NewBufferString(semver)
		if tag != "" && tag != "-" {
			updated := strings.TrimPrefix(tag, "-")
			_, err := verBuilder.WriteString("-" + updated)
			if err == nil {
				verBuilder = bytes.NewBufferString(semver)
			}
		}
		version = verBuilder.String()
	})
	return version
}
