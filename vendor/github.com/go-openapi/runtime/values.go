package runtime

// Values typically represent parameters on a http request.
type Values map[string][]string

// GetOK returns the values collection for the given key.
// When the key is present in the map it will return true for hasKey.
// When the value is not empty it will return true for hasValue.
func (v Values) GetOK(key string) (value []string, hasKey bool, hasValue bool) {
	value, hasKey = v[key]
	if !hasKey {
		return
	}
	if len(value) == 0 {
		return
	}
	hasValue = true
	return
}
