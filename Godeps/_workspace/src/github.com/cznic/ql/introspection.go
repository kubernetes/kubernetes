// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"bytes"
	"fmt"
	"go/ast"
	"reflect"
	"strings"
	"sync"
)

var (
	schemaCache = map[reflect.Type]*StructInfo{}
	schemaMu    sync.RWMutex
)

// StructInfo describes a struct type. An instance of StructInfo obtained from
// StructSchema is shared and must not be mutated.  That includes the values
// pointed to by the elements of Fields and Indices.
type StructInfo struct {
	Fields  []*StructField // Fields describe the considered fields of a struct type.
	HasID   bool           // Whether the struct has a considered field named ID of type int64.
	Indices []*StructIndex // Indices describe indices defined by the index or uindex ql tags.
	IsPtr   bool           // Whether the StructInfo was derived from a pointer to a struct.
}

// StructIndex describes an index defined by the ql tag index or uindex.
type StructIndex struct {
	ColumnName string // Name of the column the index is on.
	Name       string // Name of the index.
	Unique     bool   // Whether the index is unique.
}

// StructField describes a considered field of a struct type.
type StructField struct {
	Index         int               // Index is the index of the field for reflect.Value.Field.
	IsID          bool              // Whether the field corresponds to record id().
	IsPtr         bool              // Whether the field is a pointer type.
	MarshalType   reflect.Type      // The reflect.Type a field must be converted to when marshaling or nil when it is assignable directly. (Field->value)
	Name          string            // Field name or value of the name tag (like in `ql:"name foo"`).
	ReflectType   reflect.Type      // The reflect.Type of the field.
	Tags          map[string]string // QL tags of this field. (`ql:"a, b c, d"` -> {"a": "", "b": "c", "d": ""})
	Type          Type              // QL type of the field.
	UnmarshalType reflect.Type      // The reflect.Type a value must be converted to when unmarshaling or nil when it is assignable directly. (Field<-value)
	ZeroPtr       reflect.Value     // The reflect.Zero value of the field if it's a pointer type.
}

func (s *StructField) check(v interface{}) error {
	t := reflect.TypeOf(v)
	if !s.ReflectType.AssignableTo(t) {
		if !s.ReflectType.ConvertibleTo(t) {
			return fmt.Errorf("type %s (%v) cannot be converted to %T", s.ReflectType.Name(), s.ReflectType.Kind(), t.Name())
		}

		s.MarshalType = t
	}

	if !t.AssignableTo(s.ReflectType) {
		if !t.ConvertibleTo(s.ReflectType) {
			return fmt.Errorf("type %s (%v) cannot be converted to %T", t.Name(), t.Kind(), s.ReflectType.Name())
		}

		s.UnmarshalType = s.ReflectType
	}
	return nil
}

func parseTag(s string) map[string]string {
	m := map[string]string{}
	for _, v := range strings.Split(s, ",") {
		v = strings.TrimSpace(v)
		switch n := strings.IndexRune(v, ' '); {
		case n < 0:
			m[v] = ""
		default:
			m[v[:n]] = v[n+1:]
		}
	}
	return m
}

// StructSchema returns StructInfo for v which must be a struct instance or a
// pointer to a struct.  The info is computed only once for every type.
// Subsequent calls to StructSchema for the same type return a cached
// StructInfo.
//
// Note: The returned StructSchema is shared and must be not mutated, including
// any other data structures it may point to.
func StructSchema(v interface{}) (*StructInfo, error) {
	if v == nil {
		return nil, fmt.Errorf("cannot derive schema for %T(%v)", v, v)
	}

	typ := reflect.TypeOf(v)
	schemaMu.RLock()
	if r, ok := schemaCache[typ]; ok {
		schemaMu.RUnlock()
		return r, nil
	}

	schemaMu.RUnlock()
	var schemaPtr bool
	t := typ
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
		schemaPtr = true
	}
	if k := t.Kind(); k != reflect.Struct {
		return nil, fmt.Errorf("cannot derive schema for type %T (%v)", v, k)
	}

	r := &StructInfo{IsPtr: schemaPtr}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		fn := f.Name
		if !ast.IsExported(fn) {
			continue
		}

		tags := parseTag(f.Tag.Get("ql"))
		if _, ok := tags["-"]; ok {
			continue
		}

		if s := tags["name"]; s != "" {
			fn = s
		}

		if fn == "ID" && f.Type.Kind() == reflect.Int64 {
			r.HasID = true
		}
		var ix, unique bool
		var xn string
		xfn := fn
		if s := tags["index"]; s != "" {
			if _, ok := tags["uindex"]; ok {
				return nil, fmt.Errorf("both index and uindex in QL struct tag")
			}

			ix, xn = true, s
		} else if s := tags["uindex"]; s != "" {
			if _, ok := tags["index"]; ok {
				return nil, fmt.Errorf("both index and uindex in QL struct tag")
			}

			ix, unique, xn = true, true, s
		}
		if ix {
			if fn == "ID" && r.HasID {
				xfn = "id()"
			}
			r.Indices = append(r.Indices, &StructIndex{Name: xn, ColumnName: xfn, Unique: unique})
		}

		sf := &StructField{Index: i, Name: fn, Tags: tags, Type: Type(-1), ReflectType: f.Type}
		fk := sf.ReflectType.Kind()
		if fk == reflect.Ptr {
			sf.IsPtr = true
			sf.ZeroPtr = reflect.Zero(sf.ReflectType)
			sf.ReflectType = sf.ReflectType.Elem()
			fk = sf.ReflectType.Kind()
		}

		switch fk {
		case reflect.Bool:
			sf.Type = Bool
			if err := sf.check(false); err != nil {
				return nil, err
			}
		case reflect.Int, reflect.Uint:
			return nil, fmt.Errorf("only integers of fixed size can be used to derive a schema: %v", fk)
		case reflect.Int8:
			sf.Type = Int8
			if err := sf.check(int8(0)); err != nil {
				return nil, err
			}
		case reflect.Int16:
			if err := sf.check(int16(0)); err != nil {
				return nil, err
			}
			sf.Type = Int16
		case reflect.Int32:
			if err := sf.check(int32(0)); err != nil {
				return nil, err
			}
			sf.Type = Int32
		case reflect.Int64:
			if sf.ReflectType.Name() == "Duration" && sf.ReflectType.PkgPath() == "time" {
				sf.Type = Duration
				break
			}

			sf.Type = Int64
			if err := sf.check(int64(0)); err != nil {
				return nil, err
			}
		case reflect.Uint8:
			sf.Type = Uint8
			if err := sf.check(uint8(0)); err != nil {
				return nil, err
			}
		case reflect.Uint16:
			sf.Type = Uint16
			if err := sf.check(uint16(0)); err != nil {
				return nil, err
			}
		case reflect.Uint32:
			sf.Type = Uint32
			if err := sf.check(uint32(0)); err != nil {
				return nil, err
			}
		case reflect.Uint64:
			sf.Type = Uint64
			if err := sf.check(uint64(0)); err != nil {
				return nil, err
			}
		case reflect.Float32:
			sf.Type = Float32
			if err := sf.check(float32(0)); err != nil {
				return nil, err
			}
		case reflect.Float64:
			sf.Type = Float64
			if err := sf.check(float64(0)); err != nil {
				return nil, err
			}
		case reflect.Complex64:
			sf.Type = Complex64
			if err := sf.check(complex64(0)); err != nil {
				return nil, err
			}
		case reflect.Complex128:
			sf.Type = Complex128
			if err := sf.check(complex128(0)); err != nil {
				return nil, err
			}
		case reflect.Slice:
			sf.Type = Blob
			if err := sf.check([]byte(nil)); err != nil {
				return nil, err
			}
		case reflect.Struct:
			switch sf.ReflectType.PkgPath() {
			case "math/big":
				switch sf.ReflectType.Name() {
				case "Int":
					sf.Type = BigInt
				case "Rat":
					sf.Type = BigRat
				}
			case "time":
				switch sf.ReflectType.Name() {
				case "Time":
					sf.Type = Time
				}
			}
		case reflect.String:
			sf.Type = String
			if err := sf.check(""); err != nil {
				return nil, err
			}
		}

		if sf.Type < 0 {
			return nil, fmt.Errorf("cannot derive schema for type %s (%v)", sf.ReflectType.Name(), fk)
		}

		sf.IsID = fn == "ID" && r.HasID
		r.Fields = append(r.Fields, sf)
	}

	schemaMu.Lock()
	schemaCache[typ] = r
	if t != typ {
		r2 := *r
		r2.IsPtr = false
		schemaCache[t] = &r2
	}
	schemaMu.Unlock()
	return r, nil
}

// MustStructSchema is like StructSchema but panics on error. It simplifies
// safe initialization of global variables holding StructInfo.
//
// MustStructSchema is safe for concurrent use by multiple goroutines.
func MustStructSchema(v interface{}) *StructInfo {
	s, err := StructSchema(v)
	if err != nil {
		panic(err)
	}

	return s
}

// SchemaOptions amend the result of Schema.
type SchemaOptions struct {
	// Don't wrap the CREATE statement(s) in a transaction.
	NoTransaction bool

	// Don't insert the IF NOT EXISTS clause in the CREATE statement(s).
	NoIfNotExists bool

	// Do not strip the "pkg." part from type name "pkg.Type", produce
	// "pkg_Type" table name instead. Applies only when no name is passed
	// to Schema().
	KeepPrefix bool
}

var zeroSchemaOptions SchemaOptions

// Schema returns a CREATE TABLE/INDEX statement(s) for a table derived from a
// struct or an error, if any.  The table is named using the name parameter. If
// name is an empty string then the type name of the struct is used while non
// conforming characters are replaced by underscores. Value v can be also a
// pointer to a struct.
//
// Every considered struct field type must be one of the QL types or a type
// convertible to string, bool, int*, uint*, float* or complex* type or pointer
// to such type. Integers with a width dependent on the architecture can not be
// used. Only exported fields are considered. If an exported field QL tag
// contains "-" (`ql:"-"`) then such field is not considered. A field with name
// ID, having type int64, corresponds to id() - and is thus not a part of the
// CREATE statement. A field QL tag containing "index name" or "uindex name"
// triggers additionally creating an index or unique index on the respective
// field.  Fields can be renamed using a QL tag "name newName".  Fields are
// considered in the order of appearance. A QL tag is a struct tag part
// prefixed by "ql:". Tags can be combined, for example:
//
//	type T struct {
//		Foo	string	`ql:"index xFoo, name Bar"`
//	}
//
// If opts.NoTransaction == true then the statement(s) are not wrapped in a
// transaction. If opt.NoIfNotExists == true then the CREATE statement(s) omits
// the IF NOT EXISTS clause. Passing nil opts is equal to passing
// &SchemaOptions{}
//
// Schema is safe for concurrent use by multiple goroutines.
func Schema(v interface{}, name string, opt *SchemaOptions) (List, error) {
	if opt == nil {
		opt = &zeroSchemaOptions
	}
	s, err := StructSchema(v)
	if err != nil {
		return List{}, err
	}

	var buf bytes.Buffer
	if !opt.NoTransaction {
		buf.WriteString("BEGIN TRANSACTION; ")
	}
	buf.WriteString("CREATE TABLE ")
	if !opt.NoIfNotExists {
		buf.WriteString("IF NOT EXISTS ")
	}
	if name == "" {
		name = fmt.Sprintf("%T", v)
		if !opt.KeepPrefix {
			a := strings.Split(name, ".")
			if l := len(a); l > 1 {
				name = a[l-1]
			}
		}
		nm := []rune{}
		for _, v := range name {
			switch {
			case v >= '0' && v <= '9' || v == '_' || v >= 'a' && v <= 'z' || v >= 'A' && v <= 'Z':
				// ok
			default:
				v = '_'
			}
			nm = append(nm, v)
		}
		name = string(nm)
	}
	buf.WriteString(name + " (")
	for _, v := range s.Fields {
		if v.IsID {
			continue
		}

		buf.WriteString(fmt.Sprintf("%s %s, ", v.Name, v.Type))
	}
	buf.WriteString("); ")
	for _, v := range s.Indices {
		buf.WriteString("CREATE ")
		if v.Unique {
			buf.WriteString("UNIQUE ")
		}
		buf.WriteString("INDEX ")
		if !opt.NoIfNotExists {
			buf.WriteString("IF NOT EXISTS ")
		}
		buf.WriteString(fmt.Sprintf("%s ON %s (%s); ", v.Name, name, v.ColumnName))
	}
	if !opt.NoTransaction {
		buf.WriteString("COMMIT; ")
	}
	l, err := Compile(buf.String())
	if err != nil {
		return List{}, fmt.Errorf("%s: %v", buf.String(), err)
	}

	return l, nil
}

// MustSchema is like Schema but panics on error. It simplifies safe
// initialization of global variables holding compiled schemas.
//
// MustSchema is safe for concurrent use by multiple goroutines.
func MustSchema(v interface{}, name string, opt *SchemaOptions) List {
	l, err := Schema(v, name, opt)
	if err != nil {
		panic(err)
	}

	return l
}

// Marshal converts, in the order of appearance, fields of a struct instance v
// to []interface{} or an error, if any. Value v can be also a pointer to a
// struct.
//
// Every considered struct field type must be one of the QL types or a type
// convertible to string, bool, int*, uint*, float* or complex* type or pointer
// to such type. Integers with a width dependent on the architecture can not be
// used. Only exported fields are considered. If an exported field QL tag
// contains "-" then such field is not considered. A QL tag is a struct tag
// part prefixed by "ql:".  Field with name ID, having type int64, corresponds
// to id() - and is thus not part of the result.
//
// Marshal is safe for concurrent use by multiple goroutines.
func Marshal(v interface{}) ([]interface{}, error) {
	s, err := StructSchema(v)
	if err != nil {
		return nil, err
	}

	val := reflect.ValueOf(v)
	if s.IsPtr {
		val = val.Elem()
	}
	n := len(s.Fields)
	if s.HasID {
		n--
	}
	r := make([]interface{}, n)
	j := 0
	for _, v := range s.Fields {
		if v.IsID {
			continue
		}

		f := val.Field(v.Index)
		if v.IsPtr {
			if f.IsNil() {
				r[j] = nil
				j++
				continue
			}

			f = f.Elem()
		}
		if m := v.MarshalType; m != nil {
			f = f.Convert(m)
		}
		r[j] = f.Interface()
		j++
	}
	return r, nil
}

// MustMarshal is like Marshal but panics on error. It simplifies marshaling of
// "safe" types, like eg. those which were already verified by Schema or
// MustSchema.  When the underlying Marshal returns an error, MustMarshal
// panics.
//
// MustMarshal is safe for concurrent use by multiple goroutines.
func MustMarshal(v interface{}) []interface{} {
	r, err := Marshal(v)
	if err != nil {
		panic(err)
	}

	return r
}

// Unmarshal stores data from []interface{} in the struct value pointed to by
// v.
//
// Every considered struct field type must be one of the QL types or a type
// convertible to string, bool, int*, uint*, float* or complex* type or pointer
// to such type. Integers with a width dependent on the architecture can not be
// used. Only exported fields are considered. If an exported field QL tag
// contains "-" then such field is not considered. A QL tag is a struct tag
// part prefixed by "ql:".  Fields are considered in the order of appearance.
// Types of values in data must be compatible with the corresponding considered
// field of v.
//
// If the struct has no ID field then the number of values in data must be equal
// to the number of considered fields of v.
//
//	type T struct {
//		A bool
//		B string
//	}
//
// Assuming the schema is
//
//	CREATE TABLE T (A bool, B string);
//
// Data might be a result of queries like
//
//	SELECT * FROM T;
//	SELECT A, B FROM T;
//
// If the struct has a considered ID field then the number of values in data
// must be equal to the number of considered fields in v - or one less. In the
// later case the ID field is not set.
//
//	type U struct {
//		ID int64
//		A  bool
//		B  string
//	}
//
// Assuming the schema is
//
//	CREATE TABLE T (A bool, B string);
//
// Data might be a result of queries like
//
//	SELECT * FROM T;             // ID not set
//	SELECT A, B FROM T;          // ID not set
//	SELECT id(), A, B FROM T;    // ID is set
//
// To unmarshal a value from data into a pointer field of v, Unmarshal first
// handles the case of the value being nil. In that case, Unmarshal sets the
// pointer to nil. Otherwise, Unmarshal unmarshals the data value into value
// pointed at by the pointer. If the pointer is nil, Unmarshal allocates a new
// value for it to point to.
//
// Unmarshal is safe for concurrent use by multiple goroutines.
func Unmarshal(v interface{}, data []interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			var ok bool
			if err, ok = r.(error); !ok {
				err = fmt.Errorf("%v", r)
			}
			err = fmt.Errorf("unmarshal: %v", err)
		}
	}()

	s, err := StructSchema(v)
	if err != nil {
		return err
	}

	if !s.IsPtr {
		return fmt.Errorf("unmarshal: need a pointer to a struct")
	}

	id := false
	nv, nf := len(data), len(s.Fields)
	switch s.HasID {
	case true:
		switch {
		case nv == nf:
			id = true
		case nv == nf-1:
			// ok
		default:
			return fmt.Errorf("unmarshal: got %d values, need %d or %d", nv, nf-1, nf)
		}
	default:
		switch {
		case nv == nf:
			// ok
		default:
			return fmt.Errorf("unmarshal: got %d values, need %d", nv, nf)
		}
	}

	j := 0
	vVal := reflect.ValueOf(v)
	if s.IsPtr {
		vVal = vVal.Elem()
	}
	for _, sf := range s.Fields {
		if sf.IsID && !id {
			continue
		}

		d := data[j]
		val := reflect.ValueOf(d)
		j++

		fVal := vVal.Field(sf.Index)
		if u := sf.UnmarshalType; u != nil {
			val = val.Convert(u)
		}
		if !sf.IsPtr {
			fVal.Set(val)
			continue
		}

		if d == nil {
			fVal.Set(sf.ZeroPtr)
			continue
		}

		if fVal.IsNil() {
			fVal.Set(reflect.New(sf.ReflectType))
		}

		fVal.Elem().Set(val)
	}
	return nil
}
