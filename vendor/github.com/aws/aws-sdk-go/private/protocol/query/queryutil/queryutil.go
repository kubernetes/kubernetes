package queryutil

import (
	"encoding/base64"
	"fmt"
	"net/url"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/private/protocol"
)

// Parse parses an object i and fills a url.Values object. The isEC2 flag
// indicates if this is the EC2 Query sub-protocol.
func Parse(body url.Values, i interface{}, isEC2 bool) error {
	q := queryParser{isEC2: isEC2}
	return q.parseValue(body, reflect.ValueOf(i), "", "")
}

func elemOf(value reflect.Value) reflect.Value {
	for value.Kind() == reflect.Ptr {
		value = value.Elem()
	}
	return value
}

type queryParser struct {
	isEC2 bool
}

func (q *queryParser) parseValue(v url.Values, value reflect.Value, prefix string, tag reflect.StructTag) error {
	value = elemOf(value)

	// no need to handle zero values
	if !value.IsValid() {
		return nil
	}

	t := tag.Get("type")
	if t == "" {
		switch value.Kind() {
		case reflect.Struct:
			t = "structure"
		case reflect.Slice:
			t = "list"
		case reflect.Map:
			t = "map"
		}
	}

	switch t {
	case "structure":
		return q.parseStruct(v, value, prefix)
	case "list":
		return q.parseList(v, value, prefix, tag)
	case "map":
		return q.parseMap(v, value, prefix, tag)
	default:
		return q.parseScalar(v, value, prefix, tag)
	}
}

func (q *queryParser) parseStruct(v url.Values, value reflect.Value, prefix string) error {
	if !value.IsValid() {
		return nil
	}

	t := value.Type()
	for i := 0; i < value.NumField(); i++ {
		elemValue := elemOf(value.Field(i))
		field := t.Field(i)

		if field.PkgPath != "" {
			continue // ignore unexported fields
		}

		if protocol.CanSetIdempotencyToken(value.Field(i), field) {
			token := protocol.GetIdempotencyToken()
			elemValue = reflect.ValueOf(token)
		}

		var name string
		if q.isEC2 {
			name = field.Tag.Get("queryName")
		}
		if name == "" {
			if field.Tag.Get("flattened") != "" && field.Tag.Get("locationNameList") != "" {
				name = field.Tag.Get("locationNameList")
			} else if locName := field.Tag.Get("locationName"); locName != "" {
				name = locName
			}
			if name != "" && q.isEC2 {
				name = strings.ToUpper(name[0:1]) + name[1:]
			}
		}
		if name == "" {
			name = field.Name
		}

		if prefix != "" {
			name = prefix + "." + name
		}

		if err := q.parseValue(v, elemValue, name, field.Tag); err != nil {
			return err
		}
	}
	return nil
}

func (q *queryParser) parseList(v url.Values, value reflect.Value, prefix string, tag reflect.StructTag) error {
	// If it's empty, generate an empty value
	if !value.IsNil() && value.Len() == 0 {
		v.Set(prefix, "")
		return nil
	}

	// check for unflattened list member
	if !q.isEC2 && tag.Get("flattened") == "" {
		prefix += ".member"
	}

	for i := 0; i < value.Len(); i++ {
		slicePrefix := prefix
		if slicePrefix == "" {
			slicePrefix = strconv.Itoa(i + 1)
		} else {
			slicePrefix = slicePrefix + "." + strconv.Itoa(i+1)
		}
		if err := q.parseValue(v, value.Index(i), slicePrefix, ""); err != nil {
			return err
		}
	}
	return nil
}

func (q *queryParser) parseMap(v url.Values, value reflect.Value, prefix string, tag reflect.StructTag) error {
	// If it's empty, generate an empty value
	if !value.IsNil() && value.Len() == 0 {
		v.Set(prefix, "")
		return nil
	}

	// check for unflattened list member
	if !q.isEC2 && tag.Get("flattened") == "" {
		prefix += ".entry"
	}

	// sort keys for improved serialization consistency.
	// this is not strictly necessary for protocol support.
	mapKeyValues := value.MapKeys()
	mapKeys := map[string]reflect.Value{}
	mapKeyNames := make([]string, len(mapKeyValues))
	for i, mapKey := range mapKeyValues {
		name := mapKey.String()
		mapKeys[name] = mapKey
		mapKeyNames[i] = name
	}
	sort.Strings(mapKeyNames)

	for i, mapKeyName := range mapKeyNames {
		mapKey := mapKeys[mapKeyName]
		mapValue := value.MapIndex(mapKey)

		kname := tag.Get("locationNameKey")
		if kname == "" {
			kname = "key"
		}
		vname := tag.Get("locationNameValue")
		if vname == "" {
			vname = "value"
		}

		// serialize key
		var keyName string
		if prefix == "" {
			keyName = strconv.Itoa(i+1) + "." + kname
		} else {
			keyName = prefix + "." + strconv.Itoa(i+1) + "." + kname
		}

		if err := q.parseValue(v, mapKey, keyName, ""); err != nil {
			return err
		}

		// serialize value
		var valueName string
		if prefix == "" {
			valueName = strconv.Itoa(i+1) + "." + vname
		} else {
			valueName = prefix + "." + strconv.Itoa(i+1) + "." + vname
		}

		if err := q.parseValue(v, mapValue, valueName, ""); err != nil {
			return err
		}
	}

	return nil
}

func (q *queryParser) parseScalar(v url.Values, r reflect.Value, name string, tag reflect.StructTag) error {
	switch value := r.Interface().(type) {
	case string:
		v.Set(name, value)
	case []byte:
		if !r.IsNil() {
			v.Set(name, base64.StdEncoding.EncodeToString(value))
		}
	case bool:
		v.Set(name, strconv.FormatBool(value))
	case int64:
		v.Set(name, strconv.FormatInt(value, 10))
	case int:
		v.Set(name, strconv.Itoa(value))
	case float64:
		v.Set(name, strconv.FormatFloat(value, 'f', -1, 64))
	case float32:
		v.Set(name, strconv.FormatFloat(float64(value), 'f', -1, 32))
	case time.Time:
		const ISO8601UTC = "2006-01-02T15:04:05Z"
		v.Set(name, value.UTC().Format(ISO8601UTC))
	default:
		return fmt.Errorf("unsupported value for param %s: %v (%s)", name, r.Interface(), r.Type().Name())
	}
	return nil
}
