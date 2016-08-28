package types

// Store is a key/value store with case-insensitive keys.
type Store interface {

	// Map returns the contents of the store as a map[string]interface{}.
	Map() map[string]interface{}

	// Keys returns a list of all the keys in the store.
	Keys() []string

	// IsSet returns true if a key exists.
	IsSet(k string) bool

	// Get returns a value for the key; a nil value if the key does not exist.
	Get(k string) interface{}

	// GetString returns a string value for a key; an empty string if the key
	// does not exist.
	GetString(k string) string

	// GetStringPtr returns a pointer to a string value for a key; nil if
	// the key does not exist.
	GetStringPtr(k string) *string

	// GetBool returns a boolean value for the key; false if the key does not
	// exist.
	GetBool(k string) bool

	// GetBoolPtr returns a pointer to a boolean value for the key; nil if the
	// key does not exist.
	GetBoolPtr(k string) *bool

	// GetInt return an int value for the key; 0 if the key does not exist.
	GetInt(k string) int

	// GetInt return a pointer to an int value for the key; nil if the key does
	// not exist.
	GetIntPtr(k string) *int

	// GetInt64 return an int64 value for the key; 0 if the key does not exist.
	GetInt64(k string) int64

	// GetInt64Ptr return a pointer to an int64 value for the key; nil if the
	// key does not exist.
	GetInt64Ptr(k string) *int64

	// GetIntSlice returns an int slice value for a key; a nil value if
	// the key does not exist.
	GetIntSlice(k string) []int

	// GetStringSlice returns a string slice value for a key; a nil value if
	// the key does not exist.
	GetStringSlice(k string) []string

	// GetBoolSlice returns a bool slice value for a key; a nil value if
	// the key does not exist.
	GetBoolSlice(k string) []bool

	// GetInstanceID returns an instance ID for a key; a nil value if the key
	// does not exist.
	GetInstanceID(k string) *InstanceID

	// GetMap returns a map value for a key; a nil value if the key does not
	// exist.
	GetMap(k string) map[string]interface{}

	// GetStore returns a Store value for a key; a nil value if the key does
	// not exist.
	GetStore(k string) Store

	// Set sets a key/value in the store.
	Set(k string, v interface{})

	// Deletes a key/value from the store. If the value exists in the map it
	// is returned.
	Delete(k string) interface{}
}
