// untested sections: 5

package matchers

import (
	"fmt"
	"reflect"
	"strings"
)

func formattedMessage(comparisonMessage string, failurePath []any) string {
	var diffMessage string
	if len(failurePath) == 0 {
		diffMessage = ""
	} else {
		diffMessage = fmt.Sprintf("\n\nfirst mismatched key: %s", formattedFailurePath(failurePath))
	}
	return fmt.Sprintf("%s%s", comparisonMessage, diffMessage)
}

func formattedFailurePath(failurePath []any) string {
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

func deepEqual(a any, b any) (bool, []any) {
	var errorPath []any
	if reflect.TypeOf(a) != reflect.TypeOf(b) {
		return false, errorPath
	}

	switch a.(type) {
	case []any:
		if len(a.([]any)) != len(b.([]any)) {
			return false, errorPath
		}

		for i, v := range a.([]any) {
			elementEqual, keyPath := deepEqual(v, b.([]any)[i])
			if !elementEqual {
				return false, append(keyPath, i)
			}
		}
		return true, errorPath

	case map[any]any:
		if len(a.(map[any]any)) != len(b.(map[any]any)) {
			return false, errorPath
		}

		for k, v1 := range a.(map[any]any) {
			v2, ok := b.(map[any]any)[k]
			if !ok {
				return false, errorPath
			}
			elementEqual, keyPath := deepEqual(v1, v2)
			if !elementEqual {
				return false, append(keyPath, k)
			}
		}
		return true, errorPath

	case map[string]any:
		if len(a.(map[string]any)) != len(b.(map[string]any)) {
			return false, errorPath
		}

		for k, v1 := range a.(map[string]any) {
			v2, ok := b.(map[string]any)[k]
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
