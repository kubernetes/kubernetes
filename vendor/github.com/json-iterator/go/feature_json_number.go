package jsoniter

import "encoding/json"

type Number string

func CastJsonNumber(val interface{}) (string, bool) {
	switch typedVal := val.(type) {
	case json.Number:
		return string(typedVal), true
	case Number:
		return string(typedVal), true
	}
	return "", false
}
