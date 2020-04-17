// untested sections: 5

package matchers

import (
	"fmt"
	"reflect"
	"strings"
)

func formattedMessage(comparisonMessage string, failurePath []interface{}) string {
	var diffMessage string
	if len(failurePath) == 0 {
		diffMessage = ""
	} else {
		diffMessage = fmt.Sprintf("\n\nfirst mismatched key: %s", formattedFailurePath(failurePath))
	}
	return fmt.Sprintf("%s%s", comparisonMessage, diffMessage)
}

func formattedFailurePath(failurePath []interface{}) string {
	formattedPaths := []string{}
	for i := len(failurePath) - 1; i >= 0; i-- {
		switch p := failurePath[i].(type) {
		case int:
			formattedPaths = append(formattedPaths, fmt.Sprintf(`[%d]`, p))
		default:
			if i != len(failurePath)-1 {
				formattedPaths = append(formattedPaths, ".")
			}
			formattedPaths = append(formattedPaths, fmt.Sprintf(`"%s"`, p))
		}
	}
	return strings.Join(formattedPaths, "")
}

func deepEqual(a interface{}, b interface{}) (bool, []interface{}) {
	var errorPath []interface{}
	if reflect.TypeOf(a) != reflect.TypeOf(b) {
		return false, errorPath
	}

	switch a.(type) {
	case []interface{}:
		if len(a.([]interface{})) != len(b.([]interface{})) {
			return false, errorPath
		}

		for i, v := range a.([]interface{}) {
			elementEqual, keyPath := deepEqual(v, b.([]interface{})[i])
			if !elementEqual {
				return false, append(keyPath, i)
			}
		}
		return true, errorPath

	case map[interface{}]interface{}:
		if len(a.(map[interface{}]interface{})) != len(b.(map[interface{}]interface{})) {
			return false, errorPath
		}

		for k, v1 := range a.(map[interface{}]interface{}) {
			v2, ok := b.(map[interface{}]interface{})[k]
			if !ok {
				return false, errorPath
			}
			elementEqual, keyPath := deepEqual(v1, v2)
			if !elementEqual {
				return false, append(keyPath, k)
			}
		}
		return true, errorPath

	case map[string]interface{}:
		if len(a.(map[string]interface{})) != len(b.(map[string]interface{})) {
			return false, errorPath
		}

		for k, v1 := range a.(map[string]interface{}) {
			v2, ok := b.(map[string]interface{})[k]
			if !ok {
				return false, errorPath
			}
			elementEqual, keyPath := deepEqual(v1, v2)
			if !elementEqual {
				return false, append(keyPath, k)
			}
		}
		return true, errorPath

	default:
		return a == b, errorPath
	}
}
