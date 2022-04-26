package toml

import (
	"encoding"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"reflect"
	"strings"
)

// Unmarshaler is the interface implemented by objects that can unmarshal a
// TOML description of themselves.
type Unmarshaler interface {
	UnmarshalTOML(interface{}) error
}

// Unmarshal decodes the contents of `p` in TOML format into a pointer `v`.
func Unmarshal(p []byte, v interface{}) error {
	_, err := Decode(string(p), v)
	return err
}

// Primitive is a TOML value that hasn't been decoded into a Go value.
//
// This type can be used for any value, which will cause decoding to be delayed.
// You can use the PrimitiveDecode() function to "manually" decode these values.
//
// NOTE: The underlying representation of a `Primitive` value is subject to
// change. Do not rely on it.
//
// NOTE: Primitive values are still parsed, so using them will only avoid the
// overhead of reflection. They can be useful when you don't know the exact type
// of TOML data until runtime.
type Primitive struct {
	undecoded interface{}
	context   Key
}

// The significand precision for float32 and float64 is 24 and 53 bits; this is
// the range a natural number can be stored in a float without loss of data.
const (
	maxSafeFloat32Int = 16777215         // 2^24-1
	maxSafeFloat64Int = 9007199254740991 // 2^53-1
)

// PrimitiveDecode is just like the other `Decode*` functions, except it
// decodes a TOML value that has already been parsed. Valid primitive values
// can *only* be obtained from values filled by the decoder functions,
// including this method. (i.e., `v` may contain more `Primitive`
// values.)
//
// Meta data for primitive values is included in the meta data returned by
// the `Decode*` functions with one exception: keys returned by the Undecoded
// method will only reflect keys that were decoded. Namely, any keys hidden
// behind a Primitive will be considered undecoded. Executing this method will
// update the undecoded keys in the meta data. (See the example.)
func (md *MetaData) PrimitiveDecode(primValue Primitive, v interface{}) error {
	md.context = primValue.context
	defer func() { md.context = nil }()
	return md.unify(primValue.undecoded, rvalue(v))
}

// Decoder decodes TOML data.
//
// TOML tables correspond to Go structs or maps (dealer's choice – they can be
// used interchangeably).
//
// TOML table arrays correspond to either a slice of structs or a slice of maps.
//
// TOML datetimes correspond to Go time.Time values. Local datetimes are parsed
// in the local timezone.
//
// All other TOML types (float, string, int, bool and array) correspond to the
// obvious Go types.
//
// An exception to the above rules is if a type implements the TextUnmarshaler
// interface, in which case any primitive TOML value (floats, strings, integers,
// booleans, datetimes) will be converted to a []byte and given to the value's
// UnmarshalText method. See the Unmarshaler example for a demonstration with
// time duration strings.
//
// Key mapping
//
// TOML keys can map to either keys in a Go map or field names in a Go struct.
// The special `toml` struct tag can be used to map TOML keys to struct fields
// that don't match the key name exactly (see the example). A case insensitive
// match to struct names will be tried if an exact match can't be found.
//
// The mapping between TOML values and Go values is loose. That is, there may
// exist TOML values that cannot be placed into your representation, and there
// may be parts of your representation that do not correspond to TOML values.
// This loose mapping can be made stricter by using the IsDefined and/or
// Undecoded methods on the MetaData returned.
//
// This decoder does not handle cyclic types. Decode will not terminate if a
// cyclic type is passed.
type Decoder struct {
	r io.Reader
}

// NewDecoder creates a new Decoder.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r: r}
}

var (
	unmarshalToml = reflect.TypeOf((*Unmarshaler)(nil)).Elem()
	unmarshalText = reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()
)

// Decode TOML data in to the pointer `v`.
func (dec *Decoder) Decode(v interface{}) (MetaData, error) {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Ptr {
		s := "%q"
		if reflect.TypeOf(v) == nil {
			s = "%v"
		}

		return MetaData{}, e("cannot decode to non-pointer "+s, reflect.TypeOf(v))
	}
	if rv.IsNil() {
		return MetaData{}, e("cannot decode to nil value of %q", reflect.TypeOf(v))
	}

	// Check if this is a supported type: struct, map, interface{}, or something
	// that implements UnmarshalTOML or UnmarshalText.
	rv = indirect(rv)
	rt := rv.Type()
	if rv.Kind() != reflect.Struct && rv.Kind() != reflect.Map &&
		!(rv.Kind() == reflect.Interface && rv.NumMethod() == 0) &&
		!rt.Implements(unmarshalToml) && !rt.Implements(unmarshalText) {
		return MetaData{}, e("cannot decode to type %s", rt)
	}

	// TODO: parser should read from io.Reader? Or at the very least, make it
	// read from []byte rather than string
	data, err := ioutil.ReadAll(dec.r)
	if err != nil {
		return MetaData{}, err
	}

	p, err := parse(string(data))
	if err != nil {
		return MetaData{}, err
	}

	md := MetaData{
		mapping: p.mapping,
		types:   p.types,
		keys:    p.ordered,
		decoded: make(map[string]struct{}, len(p.ordered)),
		context: nil,
	}
	return md, md.unify(p.mapping, rv)
}

// Decode the TOML data in to the pointer v.
//
// See the documentation on Decoder for a description of the decoding process.
func Decode(data string, v interface{}) (MetaData, error) {
	return NewDecoder(strings.NewReader(data)).Decode(v)
}

// DecodeFile is just like Decode, except it will automatically read the
// contents of the file at path and decode it for you.
func DecodeFile(path string, v interface{}) (MetaData, error) {
	fp, err := os.Open(path)
	if err != nil {
		return MetaData{}, err
	}
	defer fp.Close()
	return NewDecoder(fp).Decode(v)
}

// unify performs a sort of type unification based on the structure of `rv`,
// which is the client representation.
//
// Any type mismatch produces an error. Finding a type that we don't know
// how to handle produces an unsupported type error.
func (md *MetaData) unify(data interface{}, rv reflect.Value) error {
	// Special case. Look for a `Primitive` value.
	// TODO: #76 would make this superfluous after implemented.
	if rv.Type() == reflect.TypeOf((*Primitive)(nil)).Elem() {
		// Save the undecoded data and the key context into the primitive
		// value.
		context := make(Key, len(md.context))
		copy(context, md.context)
		rv.Set(reflect.ValueOf(Primitive{
			undecoded: data,
			context:   context,
		}))
		return nil
	}

	// Special case. Unmarshaler Interface support.
	if rv.CanAddr() {
		if v, ok := rv.Addr().Interface().(Unmarshaler); ok {
			return v.UnmarshalTOML(data)
		}
	}

	// Special case. Look for a value satisfying the TextUnmarshaler interface.
	if v, ok := rv.Interface().(encoding.TextUnmarshaler); ok {
		return md.unifyText(data, v)
	}
	// TODO:
	// The behavior here is incorrect whenever a Go type satisfies the
	// encoding.TextUnmarshaler interface but also corresponds to a TOML hash or
	// array. In particular, the unmarshaler should only be applied to primitive
	// TOML values. But at this point, it will be applied to all kinds of values
	// and produce an incorrect error whenever those values are hashes or arrays
	// (including arrays of tables).

	k := rv.Kind()

	// laziness
	if k >= reflect.Int && k <= reflect.Uint64 {
		return md.unifyInt(data, rv)
	}
	switch k {
	case reflect.Ptr:
		elem := reflect.New(rv.Type().Elem())
		err := md.unify(data, reflect.Indirect(elem))
		if err != nil {
			return err
		}
		rv.Set(elem)
		return nil
	case reflect.Struct:
		return md.unifyStruct(data, rv)
	case reflect.Map:
		return md.unifyMap(data, rv)
	case reflect.Array:
		return md.unifyArray(data, rv)
	case reflect.Slice:
		return md.unifySlice(data, rv)
	case reflect.String:
		return md.unifyString(data, rv)
	case reflect.Bool:
		return md.unifyBool(data, rv)
	case reflect.Interface:
		// we only support empty interfaces.
		if rv.NumMethod() > 0 {
			return e("unsupported type %s", rv.Type())
		}
		return md.unifyAnything(data, rv)
	case reflect.Float32, reflect.Float64:
		return md.unifyFloat64(data, rv)
	}
	return e("unsupported type %s", rv.Kind())
}

func (md *MetaData) unifyStruct(mapping interface{}, rv reflect.Value) error {
	tmap, ok := mapping.(map[string]interface{})
	if !ok {
		if mapping == nil {
			return nil
		}
		return e("type mismatch for %s: expected table but found %T",
			rv.Type().String(), mapping)
	}

	for key, datum := range tmap {
		var f *field
		fields := cachedTypeFields(rv.Type())
		for i := range fields {
			ff := &fields[i]
			if ff.name == key {
				f = ff
				break
			}
			if f == nil && strings.EqualFold(ff.name, key) {
				f = ff
			}
		}
		if f != nil {
			subv := rv
			for _, i := range f.index {
				subv = indirect(subv.Field(i))
			}

			if isUnifiable(subv) {
				md.decoded[md.context.add(key).String()] = struct{}{}
				md.context = append(md.context, key)
				err := md.unify(datum, subv)
				if err != nil {
					return err
				}
				md.context = md.context[0 : len(md.context)-1]
			} else if f.name != "" {
				return e("cannot write unexported field %s.%s", rv.Type().String(), f.name)
			}
		}
	}
	return nil
}

func (md *MetaData) unifyMap(mapping interface{}, rv reflect.Value) error {
	if k := rv.Type().Key().Kind(); k != reflect.String {
		return fmt.Errorf(
			"toml: cannot decode to a map with non-string key type (%s in %q)",
			k, rv.Type())
	}

	tmap, ok := mapping.(map[string]interface{})
	if !ok {
		if tmap == nil {
			return nil
		}
		return md.badtype("map", mapping)
	}
	if rv.IsNil() {
		rv.Set(reflect.MakeMap(rv.Type()))
	}
	for k, v := range tmap {
		md.decoded[md.context.add(k).String()] = struct{}{}
		md.context = append(md.context, k)

		rvval := reflect.Indirect(reflect.New(rv.Type().Elem()))
		if err := md.unify(v, rvval); err != nil {
			return err
		}
		md.context = md.context[0 : len(md.context)-1]

		rvkey := indirect(reflect.New(rv.Type().Key()))
		rvkey.SetString(k)
		rv.SetMapIndex(rvkey, rvval)
	}
	return nil
}

func (md *MetaData) unifyArray(data interface{}, rv reflect.Value) error {
	datav := reflect.ValueOf(data)
	if datav.Kind() != reflect.Slice {
		if !datav.IsValid() {
			return nil
		}
		return md.badtype("slice", data)
	}
	if l := datav.Len(); l != rv.Len() {
		return e("expected array length %d; got TOML array of length %d", rv.Len(), l)
	}
	return md.unifySliceArray(datav, rv)
}

func (md *MetaData) unifySlice(data interface{}, rv reflect.Value) error {
	datav := reflect.ValueOf(data)
	if datav.Kind() != reflect.Slice {
		if !datav.IsValid() {
			return nil
		}
		return md.badtype("slice", data)
	}
	n := datav.Len()
	if rv.IsNil() || rv.Cap() < n {
		rv.Set(reflect.MakeSlice(rv.Type(), n, n))
	}
	rv.SetLen(n)
	return md.unifySliceArray(datav, rv)
}

func (md *MetaData) unifySliceArray(data, rv reflect.Value) error {
	l := data.Len()
	for i := 0; i < l; i++ {
		err := md.unify(data.Index(i).Interface(), indirect(rv.Index(i)))
		if err != nil {
			return err
		}
	}
	return nil
}

func (md *MetaData) unifyString(data interface{}, rv reflect.Value) error {
	if s, ok := data.(string); ok {
		rv.SetString(s)
		return nil
	}
	return md.badtype("string", data)
}

func (md *MetaData) unifyFloat64(data interface{}, rv reflect.Value) error {
	if num, ok := data.(float64); ok {
		switch rv.Kind() {
		case reflect.Float32:
			if num < -math.MaxFloat32 || num > math.MaxFloat32 {
				return e("value %f is out of range for float32", num)
			}
			fallthrough
		case reflect.Float64:
			rv.SetFloat(num)
		default:
			panic("bug")
		}
		return nil
	}

	if num, ok := data.(int64); ok {
		switch rv.Kind() {
		case reflect.Float32:
			if num < -maxSafeFloat32Int || num > maxSafeFloat32Int {
				return e("value %d is out of range for float32", num)
			}
			fallthrough
		case reflect.Float64:
			if num < -maxSafeFloat64Int || num > maxSafeFloat64Int {
				return e("value %d is out of range for float64", num)
			}
			rv.SetFloat(float64(num))
		default:
			panic("bug")
		}
		return nil
	}

	return md.badtype("float", data)
}

func (md *MetaData) unifyInt(data interface{}, rv reflect.Value) error {
	if num, ok := data.(int64); ok {
		if rv.Kind() >= reflect.Int && rv.Kind() <= reflect.Int64 {
			switch rv.Kind() {
			case reflect.Int, reflect.Int64:
				// No bounds checking necessary.
			case reflect.Int8:
				if num < math.MinInt8 || num > math.MaxInt8 {
					return e("value %d is out of range for int8", num)
				}
			case reflect.Int16:
				if num < math.MinInt16 || num > math.MaxInt16 {
					return e("value %d is out of range for int16", num)
				}
			case reflect.Int32:
				if num < math.MinInt32 || num > math.MaxInt32 {
					return e("value %d is out of range for int32", num)
				}
			}
			rv.SetInt(num)
		} else if rv.Kind() >= reflect.Uint && rv.Kind() <= reflect.Uint64 {
			unum := uint64(num)
			switch rv.Kind() {
			case reflect.Uint, reflect.Uint64:
				// No bounds checking necessary.
			case reflect.Uint8:
				if num < 0 || unum > math.MaxUint8 {
					return e("value %d is out of range for uint8", num)
				}
			case reflect.Uint16:
				if num < 0 || unum > math.MaxUint16 {
					return e("value %d is out of range for uint16", num)
				}
			case reflect.Uint32:
				if num < 0 || unum > math.MaxUint32 {
					return e("value %d is out of range for uint32", num)
				}
			}
			rv.SetUint(unum)
		} else {
			panic("unreachable")
		}
		return nil
	}
	return md.badtype("integer", data)
}

func (md *MetaData) unifyBool(data interface{}, rv reflect.Value) error {
	if b, ok := data.(bool); ok {
		rv.SetBool(b)
		return nil
	}
	return md.badtype("boolean", data)
}

func (md *MetaData) unifyAnything(data interface{}, rv reflect.Value) error {
	rv.Set(reflect.ValueOf(data))
	return nil
}

func (md *MetaData) unifyText(data interface{}, v encoding.TextUnmarshaler) error {
	var s string
	switch sdata := data.(type) {
	case Marshaler:
		text, err := sdata.MarshalTOML()
		if err != nil {
			return err
		}
		s = string(text)
	case TextMarshaler:
		text, err := sdata.MarshalText()
		if err != nil {
			return err
		}
		s = string(text)
	case fmt.Stringer:
		s = sdata.String()
	case string:
		s = sdata
	case bool:
		s = fmt.Sprintf("%v", sdata)
	case int64:
		s = fmt.Sprintf("%d", sdata)
	case float64:
		s = fmt.Sprintf("%f", sdata)
	default:
		return md.badtype("primitive (string-like)", data)
	}
	if err := v.UnmarshalText([]byte(s)); err != nil {
		return err
	}
	return nil
}

func (md *MetaData) badtype(dst string, data interface{}) error {
	return e("incompatible types: TOML key %q has type %T; destination has type %s", md.context, data, dst)
}

// rvalue returns a reflect.Value of `v`. All pointers are resolved.
func rvalue(v interface{}) reflect.Value {
	return indirect(reflect.ValueOf(v))
}

// indirect returns the value pointed to by a pointer.
//
// Pointers are followed until the value is not a pointer. New values are
// allocated for each nil pointer.
//
// An exception to this rule is if the value satisfies an interface of interest
// to us (like encoding.TextUnmarshaler).
func indirect(v reflect.Value) reflect.Value {
	if v.Kind() != reflect.Ptr {
		if v.CanSet() {
			pv := v.Addr()
			if _, ok := pv.Interface().(encoding.TextUnmarshaler); ok {
				return pv
			}
		}
		return v
	}
	if v.IsNil() {
		v.Set(reflect.New(v.Type().Elem()))
	}
	return indirect(reflect.Indirect(v))
}

func isUnifiable(rv reflect.Value) bool {
	if rv.CanSet() {
		return true
	}
	if _, ok := rv.Interface().(encoding.TextUnmarshaler); ok {
		return true
	}
	return false
}

func e(format string, args ...interface{}) error {
	return fmt.Errorf("toml: "+format, args...)
}
