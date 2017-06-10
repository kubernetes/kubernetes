package storageos

import (
	"fmt"
	"strings"
)

// ParseRef is a helper to split out the namespace and name from a path
// reference.
func ParseRef(ref string) (namespace string, name string, err error) {
	parts := strings.Split(ref, "/")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("Name must be prefixed with <namespace>/")
	}
	return parts[0], parts[1], nil
}
