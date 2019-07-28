package bson

// Current state of the JSON tag fallback option.
var useJSONTagFallback = false
var useRespectNilValues = false

// SetJSONTagFallback enables or disables the JSON-tag fallback for structure tagging. When this is enabled, structures
// without BSON tags on a field will fall-back to using the JSON tag (if present).
func SetJSONTagFallback(state bool) {
	useJSONTagFallback = state
}

// JSONTagFallbackState returns the current status of the JSON tag fallback compatability option. See SetJSONTagFallback
// for more information.
func JSONTagFallbackState() bool {
	return useJSONTagFallback
}

// SetRespectNilValues enables or disables serializing nil slices or maps to `null` values.
// In other words it enables `encoding/json` compatible behaviour.
func SetRespectNilValues(state bool) {
	useRespectNilValues = state
}

// RespectNilValuesState returns the current status of the JSON nil slices and maps fallback compatibility option.
// See SetRespectNilValues for more information.
func RespectNilValuesState() bool {
	return useRespectNilValues
}
