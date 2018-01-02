package testing

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/libopenstorage/openstorage/pkg/jsonpb"
)

func init() {
	jsonpb.RegisterSimpleStringEnum("testing.Status", "status", Status_value)
}

// StatusSimpleValueOf returns the string format of Status
func StatusSimpleValueOf(s string) (Status, error) {
	obj, err := simpleValueOf("status", Status_value, s)
	return Status(obj), err
}

// SimpleString returns the string format of Status
func (x Status) SimpleString() string {
	return simpleString("status", Status_name, int32(x))
}

func simpleValueOf(typeString string, valueMap map[string]int32, s string) (int32, error) {
	obj, ok := valueMap[strings.ToUpper(fmt.Sprintf("%s_%s", typeString, s))]
	if !ok {
		return 0, fmt.Errorf("%s for %s", strings.ToUpper(typeString), s)
	}
	return obj, nil
}

func simpleString(typeString string, nameMap map[int32]string, v int32) string {
	s, ok := nameMap[v]
	if !ok {
		return strconv.Itoa(int(v))
	}
	return strings.TrimPrefix(strings.ToLower(s), fmt.Sprintf("%s_", strings.ToLower(typeString)))
}
