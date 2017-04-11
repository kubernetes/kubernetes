package token

// StringSet is a useful type for looking up strings.
type stringSet map[string]struct{}

// NewStringSet creates a new StringSet with the given strings.
func newStringSet(keys ...string) stringSet {
	ss := make(stringSet, len(keys))
	ss.add(keys...)
	return ss
}

// Add inserts the given keys into this StringSet.
func (ss stringSet) add(keys ...string) {
	for _, key := range keys {
		ss[key] = struct{}{}
	}
}

// Contains returns whether the given key is in this StringSet.
func (ss stringSet) contains(key string) bool {
	_, ok := ss[key]
	return ok
}

// Keys returns a slice of all keys in this StringSet.
func (ss stringSet) keys() []string {
	keys := make([]string, 0, len(ss))

	for key := range ss {
		keys = append(keys, key)
	}

	return keys
}
