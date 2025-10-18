package int64support

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
)

var (
	_ = json.Marshaler(new(Object))
	_ = json.Unmarshaler(new(Object))
)

func (o *Object) MarshalJSON() ([]byte, error) {
	if o.OptionalNumber == nil {
		return ([]byte)("{}"), nil
	}
	return ([]byte)(fmt.Sprintf("{\"optional_number\": %d}", *o.OptionalNumber)), nil
}

func (o *Object) UnmarshalJSON(b []byte) error {
	var (
		trim  = func(v []byte) []byte { return bytes.Trim(v, " \n\r\t") }
		strip = func(v []byte, first, last byte) ([]byte, error) {
			x := len(v)
			if x < 2 || v[0] != first || v[x-1] != last {
				return nil, fmt.Errorf("failed to strip %q and %q from byte sequence", first, last)
			}
			return v[1 : x-1], nil
		}
	)
	b, err := strip(trim(b), '{', '}')
	if err != nil {
		return err
	}
	// poor man parser: assume the only commas appear between JSON key-value pairs,
	// and that object hierarchy is flat
	for xf, f := range bytes.Split(b, ([]byte)(",")) {
		parts := bytes.SplitN(f, ([]byte)(":"), 2)
		if x := len(parts); x != 2 {
			if xf == 0 && (x == 0 || (x == 1 && len(trim(parts[0])) == 0)) {
				return nil // empty object
			}
			return fmt.Errorf("failed to parse field-value seperator char ':'")
		}
		fieldName, err := strip(trim(parts[0]), '"', '"')
		if err != nil {
			return err
		}
		if string(fieldName) != "optional_number" {
			continue // ignore unknown field
		}
		fieldValue := trim(parts[1])
		v, err := strconv.ParseInt(string(fieldValue), 10, 64)
		if err != nil {
			return err
		}
		o.OptionalNumber = &v
		break
	}
	return nil
}
