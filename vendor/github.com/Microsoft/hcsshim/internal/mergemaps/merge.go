package mergemaps

import "encoding/json"

// Merge recursively merges map `fromMap` into map `ToMap`. Any pre-existing values
// in ToMap are overwritten. Values in fromMap are added to ToMap.
// From http://stackoverflow.com/questions/40491438/merging-two-json-strings-in-golang
func Merge(fromMap, ToMap interface{}) interface{} {
	switch fromMap := fromMap.(type) {
	case map[string]interface{}:
		ToMap, ok := ToMap.(map[string]interface{})
		if !ok {
			return fromMap
		}
		for keyToMap, valueToMap := range ToMap {
			if valueFromMap, ok := fromMap[keyToMap]; ok {
				fromMap[keyToMap] = Merge(valueFromMap, valueToMap)
			} else {
				fromMap[keyToMap] = valueToMap
			}
		}
	case nil:
		// merge(nil, map[string]interface{...}) -> map[string]interface{...}
		ToMap, ok := ToMap.(map[string]interface{})
		if ok {
			return ToMap
		}
	}
	return fromMap
}

// MergeJSON merges the contents of a JSON string into an object representation,
// returning a new object suitable for translating to JSON.
func MergeJSON(object interface{}, additionalJSON []byte) (interface{}, error) {
	if len(additionalJSON) == 0 {
		return object, nil
	}
	objectJSON, err := json.Marshal(object)
	if err != nil {
		return nil, err
	}
	var objectMap, newMap map[string]interface{}
	err = json.Unmarshal(objectJSON, &objectMap)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(additionalJSON, &newMap)
	if err != nil {
		return nil, err
	}
	return Merge(newMap, objectMap), nil
}
