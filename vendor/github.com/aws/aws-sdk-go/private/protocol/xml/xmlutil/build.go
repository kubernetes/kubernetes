// Package xmlutil provides XML serialization of AWS requests and responses.
package xmlutil

import (
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/private/protocol"
)

const (
	floatNaN    = "NaN"
	floatInf    = "Infinity"
	floatNegInf = "-Infinity"
)

// BuildXML will serialize params into an xml.Encoder. Error will be returned
// if the serialization of any of the params or nested values fails.
func BuildXML(params interface{}, e *xml.Encoder) error {
	return buildXML(params, e, false)
}

func buildXML(params interface{}, e *xml.Encoder, sorted bool) error {
	b := xmlBuilder{encoder: e, namespaces: map[string]string{}}
	root := NewXMLElement(xml.Name{})
	if err := b.buildValue(reflect.ValueOf(params), root, ""); err != nil {
		return err
	}
	for _, c := range root.Children {
		for _, v := range c {
			return StructToXML(e, v, sorted)
		}
	}
	return nil
}

// Returns the reflection element of a value, if it is a pointer.
func elemOf(value reflect.Value) reflect.Value {
	for value.Kind() == reflect.Ptr {
		value = value.Elem()
	}
	return value
}

// A xmlBuilder serializes values from Go code to XML
type xmlBuilder struct {
	encoder    *xml.Encoder
	namespaces map[string]string
}

// buildValue generic XMLNode builder for any type. Will build value for their specific type
// struct, list, map, scalar.
//
// Also takes a "type" tag value to set what type a value should be converted to XMLNode as. If
// type is not provided reflect will be used to determine the value's type.
func (b *xmlBuilder) buildValue(value reflect.Value, current *XMLNode, tag reflect.StructTag) error {
	value = elemOf(value)
	if !value.IsValid() { // no need to handle zero values
		return nil
	} else if tag.Get("location") != "" { // don't handle non-body location values
		return nil
	}

	xml := tag.Get("xml")
	if len(xml) != 0 {
		name := strings.SplitAfterN(xml, ",", 2)[0]
		if name == "-" {
			return nil
		}
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
		if field, ok := value.Type().FieldByName("_"); ok {
			tag = tag + reflect.StructTag(" ") + field.Tag
		}
		return b.buildStruct(value, current, tag)
	case "list":
		return b.buildList(value, current, tag)
	case "map":
		return b.buildMap(value, current, tag)
	default:
		return b.buildScalar(value, current, tag)
	}
}

// buildStruct adds a struct and its fields to the current XMLNode. All fields and any nested
// types are converted to XMLNodes also.
func (b *xmlBuilder) buildStruct(value reflect.Value, current *XMLNode, tag reflect.StructTag) error {
	if !value.IsValid() {
		return nil
	}

	// unwrap payloads
	if payload := tag.Get("payload"); payload != "" {
		field, _ := value.Type().FieldByName(payload)
		tag = field.Tag
		value = elemOf(value.FieldByName(payload))

		if !value.IsValid() {
			return nil
		}
	}

	child := NewXMLElement(xml.Name{Local: tag.Get("locationName")})

	// there is an xmlNamespace associated with this struct
	if prefix, uri := tag.Get("xmlPrefix"), tag.Get("xmlURI"); uri != "" {
		ns := xml.Attr{
			Name:  xml.Name{Local: "xmlns"},
			Value: uri,
		}
		if prefix != "" {
			b.namespaces[prefix] = uri // register the namespace
			ns.Name.Local = "xmlns:" + prefix
		}

		child.Attr = append(child.Attr, ns)
	}

	var payloadFields, nonPayloadFields int

	t := value.Type()
	for i := 0; i < value.NumField(); i++ {
		member := elemOf(value.Field(i))
		field := t.Field(i)

		if field.PkgPath != "" {
			continue // ignore unexported fields
		}
		if field.Tag.Get("ignore") != "" {
			continue
		}

		mTag := field.Tag
		if mTag.Get("location") != "" { // skip non-body members
			nonPayloadFields++
			continue
		}
		payloadFields++

		if protocol.CanSetIdempotencyToken(value.Field(i), field) {
			token := protocol.GetIdempotencyToken()
			member = reflect.ValueOf(token)
		}

		memberName := mTag.Get("locationName")
		if memberName == "" {
			memberName = field.Name
			mTag = reflect.StructTag(string(mTag) + ` locationName:"` + memberName + `"`)
		}
		if err := b.buildValue(member, child, mTag); err != nil {
			return err
		}
	}

	// Only case where the child shape is not added is if the shape only contains
	// non-payload fields, e.g headers/query.
	if !(payloadFields == 0 && nonPayloadFields > 0) {
		current.AddChild(child)
	}

	return nil
}

// buildList adds the value's list items to the current XMLNode as children nodes. All
// nested values in the list are converted to XMLNodes also.
func (b *xmlBuilder) buildList(value reflect.Value, current *XMLNode, tag reflect.StructTag) error {
	if value.IsNil() { // don't build omitted lists
		return nil
	}

	// check for unflattened list member
	flattened := tag.Get("flattened") != ""

	xname := xml.Name{Local: tag.Get("locationName")}
	if flattened {
		for i := 0; i < value.Len(); i++ {
			child := NewXMLElement(xname)
			current.AddChild(child)
			if err := b.buildValue(value.Index(i), child, ""); err != nil {
				return err
			}
		}
	} else {
		list := NewXMLElement(xname)
		current.AddChild(list)

		for i := 0; i < value.Len(); i++ {
			iname := tag.Get("locationNameList")
			if iname == "" {
				iname = "member"
			}

			child := NewXMLElement(xml.Name{Local: iname})
			list.AddChild(child)
			if err := b.buildValue(value.Index(i), child, ""); err != nil {
				return err
			}
		}
	}

	return nil
}

// buildMap adds the value's key/value pairs to the current XMLNode as children nodes. All
// nested values in the map are converted to XMLNodes also.
//
// Error will be returned if it is unable to build the map's values into XMLNodes
func (b *xmlBuilder) buildMap(value reflect.Value, current *XMLNode, tag reflect.StructTag) error {
	if value.IsNil() { // don't build omitted maps
		return nil
	}

	maproot := NewXMLElement(xml.Name{Local: tag.Get("locationName")})
	current.AddChild(maproot)
	current = maproot

	kname, vname := "key", "value"
	if n := tag.Get("locationNameKey"); n != "" {
		kname = n
	}
	if n := tag.Get("locationNameValue"); n != "" {
		vname = n
	}

	// sorting is not required for compliance, but it makes testing easier
	keys := make([]string, value.Len())
	for i, k := range value.MapKeys() {
		keys[i] = k.String()
	}
	sort.Strings(keys)

	for _, k := range keys {
		v := value.MapIndex(reflect.ValueOf(k))

		mapcur := current
		if tag.Get("flattened") == "" { // add "entry" tag to non-flat maps
			child := NewXMLElement(xml.Name{Local: "entry"})
			mapcur.AddChild(child)
			mapcur = child
		}

		kchild := NewXMLElement(xml.Name{Local: kname})
		kchild.Text = k
		vchild := NewXMLElement(xml.Name{Local: vname})
		mapcur.AddChild(kchild)
		mapcur.AddChild(vchild)

		if err := b.buildValue(v, vchild, ""); err != nil {
			return err
		}
	}

	return nil
}

// buildScalar will convert the value into a string and append it as a attribute or child
// of the current XMLNode.
//
// The value will be added as an attribute if tag contains a "xmlAttribute" attribute value.
//
// Error will be returned if the value type is unsupported.
func (b *xmlBuilder) buildScalar(value reflect.Value, current *XMLNode, tag reflect.StructTag) error {
	var str string

	switch converted := value.Interface().(type) {
	case string:
		str = converted
	case []byte:
		if !value.IsNil() {
			str = base64.StdEncoding.EncodeToString(converted)
		}
	case bool:
		str = strconv.FormatBool(converted)
	case int64:
		str = strconv.FormatInt(converted, 10)
	case int:
		str = strconv.Itoa(converted)
	case float64:
		switch {
		case math.IsNaN(converted):
			str = floatNaN
		case math.IsInf(converted, 1):
			str = floatInf
		case math.IsInf(converted, -1):
			str = floatNegInf
		default:
			str = strconv.FormatFloat(converted, 'f', -1, 64)
		}
	case float32:
		// The SDK doesn't render float32 values in types, only float64. This case would never be hit currently.
		asFloat64 := float64(converted)
		switch {
		case math.IsNaN(asFloat64):
			str = floatNaN
		case math.IsInf(asFloat64, 1):
			str = floatInf
		case math.IsInf(asFloat64, -1):
			str = floatNegInf
		default:
			str = strconv.FormatFloat(asFloat64, 'f', -1, 32)
		}
	case time.Time:
		format := tag.Get("timestampFormat")
		if len(format) == 0 {
			format = protocol.ISO8601TimeFormatName
		}

		str = protocol.FormatTime(format, converted)
	default:
		return fmt.Errorf("unsupported value for param %s: %v (%s)",
			tag.Get("locationName"), value.Interface(), value.Type().Name())
	}

	xname := xml.Name{Local: tag.Get("locationName")}
	if tag.Get("xmlAttribute") != "" { // put into current node's attribute list
		attr := xml.Attr{Name: xname, Value: str}
		current.Attr = append(current.Attr, attr)
	} else if len(xname.Local) == 0 {
		current.Text = str
	} else { // regular text node
		current.AddChild(&XMLNode{Name: xname, Text: str})
	}
	return nil
}
