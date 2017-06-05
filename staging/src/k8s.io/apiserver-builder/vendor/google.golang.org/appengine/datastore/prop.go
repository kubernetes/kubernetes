// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"unicode"
)

// Entities with more than this many indexed properties will not be saved.
const maxIndexedProperties = 20000

// []byte fields more than 1 megabyte long will not be loaded or saved.
const maxBlobLen = 1 << 20

// Property is a name/value pair plus some metadata. A datastore entity's
// contents are loaded and saved as a sequence of Properties. An entity can
// have multiple Properties with the same name, provided that p.Multiple is
// true on all of that entity's Properties with that name.
type Property struct {
	// Name is the property name.
	Name string
	// Value is the property value. The valid types are:
	//	- int64
	//	- bool
	//	- string
	//	- float64
	//	- ByteString
	//	- *Key
	//	- time.Time
	//	- appengine.BlobKey
	//	- appengine.GeoPoint
	//	- []byte (up to 1 megabyte in length)
	// This set is smaller than the set of valid struct field types that the
	// datastore can load and save. A Property Value cannot be a slice (apart
	// from []byte); use multiple Properties instead. Also, a Value's type
	// must be explicitly on the list above; it is not sufficient for the
	// underlying type to be on that list. For example, a Value of "type
	// myInt64 int64" is invalid. Smaller-width integers and floats are also
	// invalid. Again, this is more restrictive than the set of valid struct
	// field types.
	//
	// A Value will have an opaque type when loading entities from an index,
	// such as via a projection query. Load entities into a struct instead
	// of a PropertyLoadSaver when using a projection query.
	//
	// A Value may also be the nil interface value; this is equivalent to
	// Python's None but not directly representable by a Go struct. Loading
	// a nil-valued property into a struct will set that field to the zero
	// value.
	Value interface{}
	// NoIndex is whether the datastore cannot index this property.
	NoIndex bool
	// Multiple is whether the entity can have multiple properties with
	// the same name. Even if a particular instance only has one property with
	// a certain name, Multiple should be true if a struct would best represent
	// it as a field of type []T instead of type T.
	Multiple bool
}

// ByteString is a short byte slice (up to 1500 bytes) that can be indexed.
type ByteString []byte

// PropertyLoadSaver can be converted from and to a slice of Properties.
type PropertyLoadSaver interface {
	Load([]Property) error
	Save() ([]Property, error)
}

// PropertyList converts a []Property to implement PropertyLoadSaver.
type PropertyList []Property

var (
	typeOfPropertyLoadSaver = reflect.TypeOf((*PropertyLoadSaver)(nil)).Elem()
	typeOfPropertyList      = reflect.TypeOf(PropertyList(nil))
)

// Load loads all of the provided properties into l.
// It does not first reset *l to an empty slice.
func (l *PropertyList) Load(p []Property) error {
	*l = append(*l, p...)
	return nil
}

// Save saves all of l's properties as a slice or Properties.
func (l *PropertyList) Save() ([]Property, error) {
	return *l, nil
}

// validPropertyName returns whether name consists of one or more valid Go
// identifiers joined by ".".
func validPropertyName(name string) bool {
	if name == "" {
		return false
	}
	for _, s := range strings.Split(name, ".") {
		if s == "" {
			return false
		}
		first := true
		for _, c := range s {
			if first {
				first = false
				if c != '_' && !unicode.IsLetter(c) {
					return false
				}
			} else {
				if c != '_' && !unicode.IsLetter(c) && !unicode.IsDigit(c) {
					return false
				}
			}
		}
	}
	return true
}

// structTag is the parsed `datastore:"name,options"` tag of a struct field.
// If a field has no tag, or the tag has an empty name, then the structTag's
// name is just the field name. A "-" name means that the datastore ignores
// that field.
type structTag struct {
	name    string
	noIndex bool
}

// structCodec describes how to convert a struct to and from a sequence of
// properties.
type structCodec struct {
	// byIndex gives the structTag for the i'th field.
	byIndex []structTag
	// byName gives the field codec for the structTag with the given name.
	byName map[string]fieldCodec
	// hasSlice is whether a struct or any of its nested or embedded structs
	// has a slice-typed field (other than []byte).
	hasSlice bool
	// complete is whether the structCodec is complete. An incomplete
	// structCodec may be encountered when walking a recursive struct.
	complete bool
}

// fieldCodec is a struct field's index and, if that struct field's type is
// itself a struct, that substruct's structCodec.
type fieldCodec struct {
	index          int
	substructCodec *structCodec
}

// structCodecs collects the structCodecs that have already been calculated.
var (
	structCodecsMutex sync.Mutex
	structCodecs      = make(map[reflect.Type]*structCodec)
)

// getStructCodec returns the structCodec for the given struct type.
func getStructCodec(t reflect.Type) (*structCodec, error) {
	structCodecsMutex.Lock()
	defer structCodecsMutex.Unlock()
	return getStructCodecLocked(t)
}

// getStructCodecLocked implements getStructCodec. The structCodecsMutex must
// be held when calling this function.
func getStructCodecLocked(t reflect.Type) (ret *structCodec, retErr error) {
	c, ok := structCodecs[t]
	if ok {
		return c, nil
	}
	c = &structCodec{
		byIndex: make([]structTag, t.NumField()),
		byName:  make(map[string]fieldCodec),
	}

	// Add c to the structCodecs map before we are sure it is good. If t is
	// a recursive type, it needs to find the incomplete entry for itself in
	// the map.
	structCodecs[t] = c
	defer func() {
		if retErr != nil {
			delete(structCodecs, t)
		}
	}()

	for i := range c.byIndex {
		f := t.Field(i)
		name, opts := f.Tag.Get("datastore"), ""
		if i := strings.Index(name, ","); i != -1 {
			name, opts = name[:i], name[i+1:]
		}
		if name == "" {
			if !f.Anonymous {
				name = f.Name
			}
		} else if name == "-" {
			c.byIndex[i] = structTag{name: name}
			continue
		} else if !validPropertyName(name) {
			return nil, fmt.Errorf("datastore: struct tag has invalid property name: %q", name)
		}

		substructType, fIsSlice := reflect.Type(nil), false
		switch f.Type.Kind() {
		case reflect.Struct:
			substructType = f.Type
		case reflect.Slice:
			if f.Type.Elem().Kind() == reflect.Struct {
				substructType = f.Type.Elem()
			}
			fIsSlice = f.Type != typeOfByteSlice
			c.hasSlice = c.hasSlice || fIsSlice
		}

		if substructType != nil && substructType != typeOfTime && substructType != typeOfGeoPoint {
			if name != "" {
				name = name + "."
			}
			sub, err := getStructCodecLocked(substructType)
			if err != nil {
				return nil, err
			}
			if !sub.complete {
				return nil, fmt.Errorf("datastore: recursive struct: field %q", f.Name)
			}
			if fIsSlice && sub.hasSlice {
				return nil, fmt.Errorf(
					"datastore: flattening nested structs leads to a slice of slices: field %q", f.Name)
			}
			c.hasSlice = c.hasSlice || sub.hasSlice
			for relName := range sub.byName {
				absName := name + relName
				if _, ok := c.byName[absName]; ok {
					return nil, fmt.Errorf("datastore: struct tag has repeated property name: %q", absName)
				}
				c.byName[absName] = fieldCodec{index: i, substructCodec: sub}
			}
		} else {
			if _, ok := c.byName[name]; ok {
				return nil, fmt.Errorf("datastore: struct tag has repeated property name: %q", name)
			}
			c.byName[name] = fieldCodec{index: i}
		}

		c.byIndex[i] = structTag{
			name:    name,
			noIndex: opts == "noindex",
		}
	}
	c.complete = true
	return c, nil
}

// structPLS adapts a struct to be a PropertyLoadSaver.
type structPLS struct {
	v     reflect.Value
	codec *structCodec
}

// newStructPLS returns a PropertyLoadSaver for the struct pointer p.
func newStructPLS(p interface{}) (PropertyLoadSaver, error) {
	v := reflect.ValueOf(p)
	if v.Kind() != reflect.Ptr || v.Elem().Kind() != reflect.Struct {
		return nil, ErrInvalidEntityType
	}
	v = v.Elem()
	codec, err := getStructCodec(v.Type())
	if err != nil {
		return nil, err
	}
	return structPLS{v, codec}, nil
}

// LoadStruct loads the properties from p to dst.
// dst must be a struct pointer.
func LoadStruct(dst interface{}, p []Property) error {
	x, err := newStructPLS(dst)
	if err != nil {
		return err
	}
	return x.Load(p)
}

// SaveStruct returns the properties from src as a slice of Properties.
// src must be a struct pointer.
func SaveStruct(src interface{}) ([]Property, error) {
	x, err := newStructPLS(src)
	if err != nil {
		return nil, err
	}
	return x.Save()
}
