package pagination

import "fmt"

// LinkedPageBase may be embedded to implement a page that provides navigational "Next" and "Previous" links within its result.
type LinkedPageBase struct {
	PageResult

	// LinkPath lists the keys that should be traversed within a response to arrive at the "next" pointer.
	// If any link along the path is missing, an empty URL will be returned.
	// If any link results in an unexpected value type, an error will be returned.
	// When left as "nil", []string{"links", "next"} will be used as a default.
	LinkPath []string
}

// NextPageURL extracts the pagination structure from a JSON response and returns the "next" link, if one is present.
// It assumes that the links are available in a "links" element of the top-level response object.
// If this is not the case, override NextPageURL on your result type.
func (current LinkedPageBase) NextPageURL() (string, error) {
	var path []string
	var key string

	if current.LinkPath == nil {
		path = []string{"links", "next"}
	} else {
		path = current.LinkPath
	}

	submap, ok := current.Body.(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("Expected an object, but was %#v", current.Body)
	}

	for {
		key, path = path[0], path[1:len(path)]

		value, ok := submap[key]
		if !ok {
			return "", nil
		}

		if len(path) > 0 {
			submap, ok = value.(map[string]interface{})
			if !ok {
				return "", fmt.Errorf("Expected an object, but was %#v", value)
			}
		} else {
			if value == nil {
				// Actual null element.
				return "", nil
			}

			url, ok := value.(string)
			if !ok {
				return "", fmt.Errorf("Expected a string, but was %#v", value)
			}

			return url, nil
		}
	}
}

// GetBody returns the linked page's body. This method is needed to satisfy the
// Page interface.
func (current LinkedPageBase) GetBody() interface{} {
	return current.Body
}
