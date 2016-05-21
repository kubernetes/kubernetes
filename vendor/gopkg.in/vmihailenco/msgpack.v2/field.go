package msgpack

import "reflect"

type field struct {
	name      string
	index     []int
	omitEmpty bool

	encoder encoderFunc
	decoder decoderFunc
}

func (f *field) value(strct reflect.Value) reflect.Value {
	return strct.FieldByIndex(f.index)
}

func (f *field) Omit(strct reflect.Value) bool {
	return f.omitEmpty && isEmptyValue(f.value(strct))
}

func (f *field) EncodeValue(e *Encoder, strct reflect.Value) error {
	return f.encoder(e, f.value(strct))
}

func (f *field) DecodeValue(d *Decoder, strct reflect.Value) error {
	return f.decoder(d, f.value(strct))
}

//------------------------------------------------------------------------------

type fields struct {
	List  []*field
	Table map[string]*field

	omitEmpty bool
}

func newFields(numField int) *fields {
	return &fields{
		List:  make([]*field, 0, numField),
		Table: make(map[string]*field, numField),
	}
}

func (fs *fields) Len() int {
	return len(fs.List)
}

func (fs *fields) Add(field *field) {
	fs.List = append(fs.List, field)
	fs.Table[field.name] = field
	if field.omitEmpty {
		fs.omitEmpty = field.omitEmpty
	}
}

func (fs *fields) OmitEmpty(strct reflect.Value) []*field {
	if !fs.omitEmpty {
		return fs.List
	}

	fields := make([]*field, 0, fs.Len())
	for _, f := range fs.List {
		if !f.Omit(strct) {
			fields = append(fields, f)
		}
	}
	return fields
}

func getFields(typ reflect.Type) *fields {
	numField := typ.NumField()
	fs := newFields(numField)

	for i := 0; i < numField; i++ {
		f := typ.Field(i)
		if f.PkgPath != "" && !f.Anonymous {
			continue
		}

		name, opts := parseTag(f.Tag.Get("msgpack"))
		if name == "-" {
			continue
		}

		if opts.Contains("inline") {
			inlineFields(fs, f)
			continue
		}

		if name == "" {
			name = f.Name
		}
		field := field{
			name:      name,
			index:     f.Index,
			omitEmpty: opts.Contains("omitempty"),
			encoder:   getEncoder(f.Type),
			decoder:   getDecoder(f.Type),
		}
		fs.Add(&field)
	}
	return fs
}

func inlineFields(fs *fields, f reflect.StructField) {
	typ := f.Type
	if typ.Kind() == reflect.Ptr {
		typ = typ.Elem()
	}
	inlinedFields := getFields(typ).List
	for _, field := range inlinedFields {
		if _, ok := fs.Table[field.name]; ok {
			// Don't overwrite shadowed fields.
			continue
		}
		field.index = append(f.Index, field.index...)
		fs.Add(field)
	}
}

func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}
