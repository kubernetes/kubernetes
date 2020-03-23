package toml

import "strings"

// MetaData allows access to meta information about TOML data that may not
// be inferrable via reflection. In particular, whether a key has been defined
// and the TOML type of a key.
type MetaData struct {
	mapping map[string]interface{}
	types   map[string]tomlType
	keys    []Key
	decoded map[string]bool
	context Key // Used only during decoding.
}

// IsDefined returns true if the key given exists in the TOML data. The key
// should be specified hierarchially. e.g.,
//
//	// access the TOML key 'a.b.c'
//	IsDefined("a", "b", "c")
//
// IsDefined will return false if an empty key given. Keys are case sensitive.
func (md *MetaData) IsDefined(key ...string) bool {
	if len(key) == 0 {
		return false
	}

	var hash map[string]interface{}
	var ok bool
	var hashOrVal interface{} = md.mapping
	for _, k := range key {
		if hash, ok = hashOrVal.(map[string]interface{}); !ok {
			return false
		}
		if hashOrVal, ok = hash[k]; !ok {
			return false
		}
	}
	return true
}

// Type returns a string representation of the type of the key specified.
//
// Type will return the empty string if given an empty key or a key that
// does not exist. Keys are case sensitive.
func (md *MetaData) Type(key ...string) string {
	fullkey := strings.Join(key, ".")
	if typ, ok := md.types[fullkey]; ok {
		return typ.typeString()
	}
	return ""
}

// Key is the type of any TOML key, including key groups. Use (MetaData).Keys
// to get values of this type.
type Key []string

func (k Key) String() string {
	return strings.Join(k, ".")
}

func (k Key) maybeQuotedAll() string {
	var ss []string
	for i := range k {
		ss = append(ss, k.maybeQuoted(i))
	}
	return strings.Join(ss, ".")
}

func (k Key) maybeQuoted(i int) string {
	quote := false
	for _, c := range k[i] {
		if !isBareKeyChar(c) {
			quote = true
			break
		}
	}
	if quote {
		return "\"" + strings.Replace(k[i], "\"", "\\\"", -1) + "\""
	}
	return k[i]
}

func (k Key) add(piece string) Key {
	newKey := make(Key, len(k)+1)
	copy(newKey, k)
	newKey[len(k)] = piece
	return newKey
}

// Keys returns a slice of every key in the TOML data, including key groups.
// Each key is itself a slice, where the first element is the top of the
// hierarchy and the last is the most specific.
//
// The list will have the same order as the keys appeared in the TOML data.
//
// All keys returned are non-empty.
func (md *MetaData) Keys() []Key {
	return md.keys
}

// Undecoded returns all keys that have not been decoded in the order in which
// they appear in the original TOML document.
//
// This includes keys that haven't been decoded because of a Primitive value.
// Once the Primitive value is decoded, the keys will be considered decoded.
//
// Also note that decoding into an empty interface will result in no decoding,
// and so no keys will be considered decoded.
//
// In this sense, the Undecoded keys correspond to keys in the TOML document
// that do not have a concrete type in your representation.
func (md *MetaData) Undecoded() []Key {
	undecoded := make([]Key, 0, len(md.keys))
	for _, key := range md.keys {
		if !md.decoded[key.String()] {
			undecoded = append(undecoded, key)
		}
	}
	return undecoded
}
