// +build codegen

package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/private/protocol"
)

// ShapeValueBuilder provides the logic to build the nested values for a shape.
// Base64BlobValues is true if the blob field in shapeRef.Shape.Type is base64
// encoded.
type ShapeValueBuilder struct {
	// Specifies if API shapes modeled as blob types, input values are base64
	// encoded or not, and strings values instead.
	Base64BlobValues bool

	// The helper that will provide the logic and formated code to convert a
	// timestamp input value into a Go time.Time.
	ParseTimeString func(ref *ShapeRef, memberName, v string) string
}

// NewShapeValueBuilder returns an initialized ShapeValueBuilder for generating
// API shape types initialized with values.
func NewShapeValueBuilder() ShapeValueBuilder {
	return ShapeValueBuilder{ParseTimeString: parseUnixTimeString}
}

// BuildShape will recursively build the referenced shape based on the json
// object provided.  isMap will dictate how the field name is specified. If
// isMap is true, we will expect the member name to be quotes like "Foo".
func (b ShapeValueBuilder) BuildShape(ref *ShapeRef, shapes map[string]interface{}, isMap bool) string {
	order := make([]string, len(shapes))
	for k := range shapes {
		order = append(order, k)
	}
	sort.Strings(order)

	ret := ""
	for _, name := range order {
		if name == "" {
			continue
		}
		shape := shapes[name]

		// If the shape isn't a map, we want to export the value, since every field
		// defined in our shapes are exported.
		if len(name) > 0 && !isMap && strings.ToLower(name[0:1]) == name[0:1] {
			name = strings.Title(name)
		}

		memName := name
		passRef := ref.Shape.MemberRefs[name]
		if isMap {
			memName = fmt.Sprintf("%q", memName)
			passRef = &ref.Shape.ValueRef
		}
		switch v := shape.(type) {
		case map[string]interface{}:
			ret += b.BuildComplex(name, memName, passRef, ref.Shape, v)
		case []interface{}:
			ret += b.BuildList(name, memName, passRef, v)
		default:

			ret += b.BuildScalar(name, memName, passRef, v, ref.Shape.Payload == name)
		}
	}
	return ret
}

// BuildList will construct a list shape based off the service's definition of
// that list.
func (b ShapeValueBuilder) BuildList(name, memName string, ref *ShapeRef, v []interface{}) string {
	ret := ""

	if len(v) == 0 || ref == nil {
		return ""
	}

	passRef := &ref.Shape.MemberRef
	ret += fmt.Sprintf("%s: %s {\n", memName, b.GoType(ref, false))
	ret += b.buildListElements(passRef, v)
	ret += "},\n"
	return ret
}

func (b ShapeValueBuilder) buildListElements(ref *ShapeRef, v []interface{}) string {
	if len(v) == 0 || ref == nil {
		return ""
	}

	ret := ""
	format := ""
	isComplex := false
	isList := false

	// get format for atomic type. If it is not an atomic type,
	// get the element.
	switch v[0].(type) {
	case string:
		format = "%s"
	case bool:
		format = "%t"
	case float64:
		switch ref.Shape.Type {
		case "integer", "int64", "long":
			format = "%d"
		default:
			format = "%f"
		}
	case []interface{}:
		isList = true
	case map[string]interface{}:
		isComplex = true
	}

	for _, elem := range v {
		if isComplex {
			ret += fmt.Sprintf("{\n%s\n},\n", b.BuildShape(ref, elem.(map[string]interface{}), ref.Shape.Type == "map"))
		} else if isList {
			ret += fmt.Sprintf("{\n%s\n},\n", b.buildListElements(&ref.Shape.MemberRef, elem.([]interface{})))
		} else {
			switch ref.Shape.Type {
			case "integer", "int64", "long":
				elem = int(elem.(float64))
			}
			ret += fmt.Sprintf("%s,\n", getValue(ref.Shape.Type, fmt.Sprintf(format, elem)))
		}
	}
	return ret
}

// BuildScalar will build atomic Go types.
func (b ShapeValueBuilder) BuildScalar(name, memName string, ref *ShapeRef, shape interface{}, isPayload bool) string {
	if ref == nil || ref.Shape == nil {
		return ""
	}

	switch v := shape.(type) {
	case bool:
		return convertToCorrectType(memName, ref.Shape.Type, fmt.Sprintf("%t", v))
	case int:
		if ref.Shape.Type == "timestamp" {
			return b.ParseTimeString(ref, memName, fmt.Sprintf("%d", v))
		}
		return convertToCorrectType(memName, ref.Shape.Type, fmt.Sprintf("%d", v))
	case float64:

		dataType := ref.Shape.Type

		if dataType == "timestamp" {
			return b.ParseTimeString(ref, memName, fmt.Sprintf("%f", v))
		}
		if dataType == "integer" || dataType == "int64" || dataType == "long" {
			return convertToCorrectType(memName, ref.Shape.Type, fmt.Sprintf("%d", int(shape.(float64))))
		}
		return convertToCorrectType(memName, ref.Shape.Type, fmt.Sprintf("%f", v))
	case string:
		t := ref.Shape.Type
		switch t {
		case "timestamp":
			return b.ParseTimeString(ref, memName, fmt.Sprintf("%s", v))

		case "jsonvalue":
			return fmt.Sprintf("%s: %#v,\n", memName, parseJSONString(v))

		case "blob":
			if (ref.Streaming || ref.Shape.Streaming) && isPayload {
				return fmt.Sprintf("%s: aws.ReadSeekCloser(strings.NewReader(%q)),\n", memName, v)
			}
			if b.Base64BlobValues {
				decodedBlob, err := base64.StdEncoding.DecodeString(v)
				if err != nil {
					panic(fmt.Errorf("Failed to decode string: %v", err))
				}
				return fmt.Sprintf("%s: []byte(%q),\n", memName, decodedBlob)
			}
			return fmt.Sprintf("%s: []byte(%q),\n", memName, v)
		default:
			return convertToCorrectType(memName, t, v)
		}
	default:
		panic(fmt.Errorf("Unsupported scalar type: %v", reflect.TypeOf(v)))
	}
}

// BuildComplex will build the shape's value for complex types such as structs,
// and maps.
func (b ShapeValueBuilder) BuildComplex(name, memName string, ref *ShapeRef, parent *Shape, v map[string]interface{}) string {
	switch parent.Type {
	case "structure":
		if ref.Shape.Type == "map" {
			return fmt.Sprintf(`%s: %s{
				%s
			},
			`, memName, b.GoType(ref, true), b.BuildShape(ref, v, true))
		} else {
			return fmt.Sprintf(`%s: &%s{
				%s
			},
			`, memName, b.GoType(ref, true), b.BuildShape(ref, v, false))
		}
	case "map":
		if ref.Shape.Type == "map" {
			return fmt.Sprintf(`%q: %s{
				%s
			},
			`, name, b.GoType(ref, false), b.BuildShape(ref, v, true))
		} else {
			return fmt.Sprintf(`%s: &%s{
				%s
			},
			`, memName, b.GoType(ref, true), b.BuildShape(ref, v, false))
		}
	default:
		panic(fmt.Sprintf("Expected complex type but received %q", ref.Shape.Type))
	}
}

// GoType returns the string of the shape's Go type identifier.
func (b ShapeValueBuilder) GoType(ref *ShapeRef, elem bool) string {

	if ref.Shape.Type != "structure" && ref.Shape.Type != "list" && ref.Shape.Type != "map" {
		// Scalars are always pointers.
		return ref.GoTypeWithPkgName()
	}

	prefix := ""
	if ref.Shape.Type == "list" {
		ref = &ref.Shape.MemberRef
		prefix = "[]"
	}

	if elem {
		return prefix + ref.Shape.GoTypeWithPkgNameElem()
	}
	return prefix + ref.GoTypeWithPkgName()
}

// parseJSONString a json string and returns aws.JSONValue.
func parseJSONString(input string) aws.JSONValue {
	var v aws.JSONValue
	if err := json.Unmarshal([]byte(input), &v); err != nil {
		panic(fmt.Sprintf("unable to unmarshal JSONValue, %v", err))
	}
	return v
}

// InlineParseModeledTime returns the string of an inline function which
// returns time.
func inlineParseModeledTime(format, v string) string {
	const formatTimeTmpl = `func() *time.Time{
        v, err := protocol.ParseTime("%s", "%s")
        if err != nil {
            panic(err)
        }
        return &v
    }()`

	return fmt.Sprintf(formatTimeTmpl, format, v)
}

// parseUnixTimeString returns a string which assigns the value of a time
// member using an inline function Defined inline function parses time in
// UnixTimeFormat.
func parseUnixTimeString(ref *ShapeRef, memName, v string) string {
	ref.API.AddSDKImport("private/protocol")
	return fmt.Sprintf("%s: %s,\n", memName, inlineParseModeledTime(protocol.UnixTimeFormatName, v))
}
