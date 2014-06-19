package objx

// Exclude returns a new Map with the keys in the specified []string
// excluded.
func (d Map) Exclude(exclude []string) Map {

	excluded := make(Map)
	for k, v := range d {
		var shouldInclude bool = true
		for _, toExclude := range exclude {
			if k == toExclude {
				shouldInclude = false
				break
			}
		}
		if shouldInclude {
			excluded[k] = v
		}
	}

	return excluded
}

// Copy creates a shallow copy of the Obj.
func (m Map) Copy() Map {
	copied := make(map[string]interface{})
	for k, v := range m {
		copied[k] = v
	}
	return New(copied)
}

// Merge blends the specified map with a copy of this map and returns the result.
//
// Keys that appear in both will be selected from the specified map.
// This method requires that the wrapped object be a map[string]interface{}
func (m Map) Merge(merge Map) Map {
	return m.Copy().MergeHere(merge)
}

// Merge blends the specified map with this map and returns the current map.
//
// Keys that appear in both will be selected from the specified map.  The original map
// will be modified. This method requires that
// the wrapped object be a map[string]interface{}
func (m Map) MergeHere(merge Map) Map {

	for k, v := range merge {
		m[k] = v
	}

	return m

}

// Transform builds a new Obj giving the transformer a chance
// to change the keys and values as it goes. This method requires that
// the wrapped object be a map[string]interface{}
func (m Map) Transform(transformer func(key string, value interface{}) (string, interface{})) Map {
	newMap := make(map[string]interface{})
	for k, v := range m {
		modifiedKey, modifiedVal := transformer(k, v)
		newMap[modifiedKey] = modifiedVal
	}
	return New(newMap)
}

// TransformKeys builds a new map using the specified key mapping.
//
// Unspecified keys will be unaltered.
// This method requires that the wrapped object be a map[string]interface{}
func (m Map) TransformKeys(mapping map[string]string) Map {
	return m.Transform(func(key string, value interface{}) (string, interface{}) {

		if newKey, ok := mapping[key]; ok {
			return newKey, value
		}

		return key, value
	})
}
