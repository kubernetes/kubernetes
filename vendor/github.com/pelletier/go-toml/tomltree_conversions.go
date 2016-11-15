package toml

// Tools to convert a TomlTree to different representations

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// encodes a string to a TOML-compliant string value
func encodeTomlString(value string) string {
	result := ""
	for _, rr := range value {
		intRr := uint16(rr)
		switch rr {
		case '\b':
			result += "\\b"
		case '\t':
			result += "\\t"
		case '\n':
			result += "\\n"
		case '\f':
			result += "\\f"
		case '\r':
			result += "\\r"
		case '"':
			result += "\\\""
		case '\\':
			result += "\\\\"
		default:
			if intRr < 0x001F {
				result += fmt.Sprintf("\\u%0.4X", intRr)
			} else {
				result += string(rr)
			}
		}
	}
	return result
}

// Value print support function for ToString()
// Outputs the TOML compliant string representation of a value
func toTomlValue(item interface{}, indent int) string {
	tab := strings.Repeat(" ", indent)
	switch value := item.(type) {
	case int64:
		return tab + strconv.FormatInt(value, 10)
	case float64:
		return tab + strconv.FormatFloat(value, 'f', -1, 64)
	case string:
		return tab + "\"" + encodeTomlString(value) + "\""
	case bool:
		if value {
			return "true"
		}
		return "false"
	case time.Time:
		return tab + value.Format(time.RFC3339)
	case []interface{}:
		result := tab + "[\n"
		for _, item := range value {
			result += toTomlValue(item, indent+2) + ",\n"
		}
		return result + tab + "]"
	default:
		panic(fmt.Sprintf("unsupported value type: %v", value))
	}
}

// Recursive support function for ToString()
// Outputs a tree, using the provided keyspace to prefix group names
func (t *TomlTree) toToml(indent, keyspace string) string {
	result := ""
	for k, v := range t.values {
		// figure out the keyspace
		combinedKey := k
		if keyspace != "" {
			combinedKey = keyspace + "." + combinedKey
		}
		// output based on type
		switch node := v.(type) {
		case []*TomlTree:
			for _, item := range node {
				if len(item.Keys()) > 0 {
					result += fmt.Sprintf("\n%s[[%s]]\n", indent, combinedKey)
				}
				result += item.toToml(indent+"  ", combinedKey)
			}
		case *TomlTree:
			if len(node.Keys()) > 0 {
				result += fmt.Sprintf("\n%s[%s]\n", indent, combinedKey)
			}
			result += node.toToml(indent+"  ", combinedKey)
		case map[string]interface{}:
			sub := TreeFromMap(node)

			if len(sub.Keys()) > 0 {
				result += fmt.Sprintf("\n%s[%s]\n", indent, combinedKey)
			}
			result += sub.toToml(indent+"  ", combinedKey)
		case *tomlValue:
			result += fmt.Sprintf("%s%s = %s\n", indent, k, toTomlValue(node.value, 0))
		default:
			result += fmt.Sprintf("%s%s = %s\n", indent, k, toTomlValue(v, 0))
		}
	}
	return result
}

// ToString is an alias for String
func (t *TomlTree) ToString() string {
	return t.String()
}

// String generates a human-readable representation of the current tree.
// Output spans multiple lines, and is suitable for ingest by a TOML parser
func (t *TomlTree) String() string {
	return t.toToml("", "")
}

// ToMap recursively generates a representation of the current tree using map[string]interface{}.
func (t *TomlTree) ToMap() map[string]interface{} {
	result := map[string]interface{}{}

	for k, v := range t.values {
		switch node := v.(type) {
		case []*TomlTree:
			var array []interface{}
			for _, item := range node {
				array = append(array, item.ToMap())
			}
			result[k] = array
		case *TomlTree:
			result[k] = node.ToMap()
		case map[string]interface{}:
			sub := TreeFromMap(node)
			result[k] = sub.ToMap()
		case *tomlValue:
			result[k] = node.value
		}
	}

	return result
}
