package log

import "fmt"

// InterleavedKVToFields converts keyValues a la Span.LogKV() to a Field slice
// a la Span.LogFields().
func InterleavedKVToFields(keyValues ...interface{}) ([]Field, error) {
	if len(keyValues)%2 != 0 {
		return nil, fmt.Errorf("non-even keyValues len: %d", len(keyValues))
	}
	fields := make([]Field, len(keyValues)/2)
	for i := 0; i*2 < len(keyValues); i++ {
		key, ok := keyValues[i*2].(string)
		if !ok {
			return nil, fmt.Errorf(
				"non-string key (pair #%d): %T",
				i, keyValues[i*2])
		}
		switch typedVal := keyValues[i*2+1].(type) {
		case bool:
			fields[i] = Bool(key, typedVal)
		case string:
			fields[i] = String(key, typedVal)
		case int:
			fields[i] = Int(key, typedVal)
		case int8:
			fields[i] = Int32(key, int32(typedVal))
		case int16:
			fields[i] = Int32(key, int32(typedVal))
		case int32:
			fields[i] = Int32(key, typedVal)
		case int64:
			fields[i] = Int64(key, typedVal)
		case uint:
			fields[i] = Uint64(key, uint64(typedVal))
		case uint64:
			fields[i] = Uint64(key, typedVal)
		case uint8:
			fields[i] = Uint32(key, uint32(typedVal))
		case uint16:
			fields[i] = Uint32(key, uint32(typedVal))
		case uint32:
			fields[i] = Uint32(key, typedVal)
		case float32:
			fields[i] = Float32(key, typedVal)
		case float64:
			fields[i] = Float64(key, typedVal)
		default:
			// When in doubt, coerce to a string
			fields[i] = String(key, fmt.Sprint(typedVal))
		}
	}
	return fields, nil
}
