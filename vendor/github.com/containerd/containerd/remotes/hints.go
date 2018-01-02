package remotes

import "strings"

// HintExists returns true if a hint of the provided kind and values exists in
// the set of provided hints.
func HintExists(kind, value string, hints ...string) bool {
	for _, hint := range hints {
		if strings.HasPrefix(hint, kind) && strings.HasSuffix(hint, value) {
			return true
		}
	}

	return false
}

// HintValues returns a slice of the values of the hints that match kind.
func HintValues(kind string, hints ...string) []string {
	var values []string
	for _, hint := range hints {
		if strings.HasPrefix(hint, kind) {
			parts := strings.SplitN(hint, ":", 2)
			if len(parts) < 2 {
				continue
			}
			values = append(values, parts[1])
		}
	}

	return values
}
