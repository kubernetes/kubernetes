package xmlutil

import (
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// UnmarshalXML deserializes an xml.Decoder into the container v. V
// needs to match the shape of the XML expected to be decoded.
// If the shape doesn't match unmarshaling will fail.
func UnmarshalXML(v interface{}, d *xml.Decoder, wrapper string) error {
	n, _ := XMLToStruct(d, nil)
	if n.Children != nil {
		for _, root := range n.Children {
			for _, c := range root {
				if wrappedChild, ok := c.Children[wrapper]; ok {
					c = wrappedChild[0] // pull out wrapped element
				}

				err := parse(reflect.ValueOf(v), c, "")
				if err != nil {
					if err == io.EOF {
						return nil
					}
					return err
				}
			}
		}
		return nil
	}
	return nil
}

// parse deserializes any value from the XMLNode. The type tag is used to infer the type, or reflect
// will be used to determine the type from r.
func parse(r reflect.Value, node *XMLNode, tag reflect.StructTag) error {
	rtype := r.Type()
	if rtype.Kind() == reflect.Ptr {
		rtype = rtype.Elem() // check kind of actual element type
	}

	t := tag.Get("type")
	if t == "" {
		switch rtype.Kind() {
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
		if field, ok := rtype.FieldByName("SDKShapeTraits"); ok {
			tag = field.Tag
		}
		return parseStruct(r, node, tag)
	case "list":
		return parseList(r, node, tag)
	case "map":
		return parseMap(r, node, tag)
	default:
		return parseScalar(r, node, tag)
	}
}

// parseStruct deserializes a structure and its fields from an XMLNode. Any nested
// types in the structure will also be deserialized.
func parseStruct(r reflect.Value, node *XMLNode, tag reflect.StructTag) error {
	t := r.Type()
	if r.Kind() == reflect.Ptr {
		if r.IsNil() { // create the structure if it's nil
			s := reflect.New(r.Type().Elem())
			r.Set(s)
			r = s
		}

		r = r.Elem()
		t = t.Elem()
	}

	// unwrap any payloads
	if payload := tag.Get("payload"); payload != "" {
		field, _ := t.FieldByName(payload)
		return parseStruct(r.FieldByName(payload), node, field.Tag)
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if c := field.Name[0:1]; strings.ToLower(c) == c {
			continue // ignore unexported fields
		}

		// figure out what this field is called
		name := field.Name
		if field.Tag.Get("flattened") != "" && field.Tag.Get("locationNameList") != "" {
			name = field.Tag.Get("locationNameList")
		} else if locName := field.Tag.Get("locationName"); locName != "" {
			name = locName
		}

		// try to find the field by name in elements
		elems := node.Children[name]

		if elems == nil { // try to find the field in attributes
			for _, a := range node.Attr {
				if name == a.Name.Local {
					// turn this into a text node for de-serializing
					elems = []*XMLNode{&XMLNode{Text: a.Value}}
				}
			}
		}

		member := r.FieldByName(field.Name)
		for _, elem := range elems {
			err := parse(member, elem, field.Tag)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// parseList deserializes a list of values from an XML node. Each list entry
// will also be deserialized.
func parseList(r reflect.Value, node *XMLNode, tag reflect.StructTag) error {
	t := r.Type()

	if tag.Get("flattened") == "" { // look at all item entries
		mname := "member"
		if name := tag.Get("locationNameList"); name != "" {
			mname = name
		}

		if Children, ok := node.Children[mname]; ok {
			if r.IsNil() {
				r.Set(reflect.MakeSlice(t, len(Children), len(Children)))
			}

			for i, c := range Children {
				err := parse(r.Index(i), c, "")
				if err != nil {
					return err
				}
			}
		}
	} else { // flattened list means this is a single element
		if r.IsNil() {
			r.Set(reflect.MakeSlice(t, 0, 0))
		}

		childR := reflect.Zero(t.Elem())
		r.Set(reflect.Append(r, childR))
		err := parse(r.Index(r.Len()-1), node, "")
		if err != nil {
			return err
		}
	}

	return nil
}

// parseMap deserializes a map from an XMLNode. The direct children of the XMLNode
// will also be deserialized as map entries.
func parseMap(r reflect.Value, node *XMLNode, tag reflect.StructTag) error {
	t := r.Type()
	if r.Kind() == reflect.Ptr {
		t = t.Elem()
		if r.IsNil() {
			r.Set(reflect.New(t))
			r.Elem().Set(reflect.MakeMap(t))
		}

		r = r.Elem()
	}

	if tag.Get("flattened") == "" { // look at all child entries
		for _, entry := range node.Children["entry"] {
			parseMapEntry(r, entry, tag)
		}
	} else { // this element is itself an entry
		parseMapEntry(r, node, tag)
	}

	return nil
}

// parseMapEntry deserializes a map entry from a XML node.
func parseMapEntry(r reflect.Value, node *XMLNode, tag reflect.StructTag) error {
	kname, vname := "key", "value"
	if n := tag.Get("locationNameKey"); n != "" {
		kname = n
	}
	if n := tag.Get("locationNameValue"); n != "" {
		vname = n
	}

	keys, ok := node.Children[kname]
	values := node.Children[vname]
	if ok {
		for i, key := range keys {
			keyR := reflect.ValueOf(key.Text)
			value := values[i]
			valueR := reflect.New(r.Type().Elem()).Elem()

			parse(valueR, value, "")
			r.SetMapIndex(keyR, valueR)
		}
	}
	return nil
}

// parseScaller deserializes an XMLNode value into a concrete type based on the
// interface type of r.
//
// Error is returned if the deserialization fails due to invalid type conversion,
// or unsupported interface type.
func parseScalar(r reflect.Value, node *XMLNode, tag reflect.StructTag) error {
	switch r.Interface().(type) {
	case *string:
		r.Set(reflect.ValueOf(&node.Text))
		return nil
	case []byte:
		b, err := base64.StdEncoding.DecodeString(node.Text)
		if err != nil {
			return err
		}
		r.Set(reflect.ValueOf(b))
	case *bool:
		v, err := strconv.ParseBool(node.Text)
		if err != nil {
			return err
		}
		r.Set(reflect.ValueOf(&v))
	case *int64:
		v, err := strconv.ParseInt(node.Text, 10, 64)
		if err != nil {
			return err
		}
		r.Set(reflect.ValueOf(&v))
	case *float64:
		v, err := strconv.ParseFloat(node.Text, 64)
		if err != nil {
			return err
		}
		r.Set(reflect.ValueOf(&v))
	case *time.Time:
		const ISO8601UTC = "2006-01-02T15:04:05Z"
		t, err := time.Parse(ISO8601UTC, node.Text)
		if err != nil {
			return err
		}
		r.Set(reflect.ValueOf(&t))
	default:
		return fmt.Errorf("unsupported value: %v (%s)", r.Interface(), r.Type())
	}
	return nil
}
