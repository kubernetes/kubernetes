package gcfg

import (
	"fmt"
	"math/big"
	"reflect"
	"strings"
	"unicode"
	"unicode/utf8"

	"code.google.com/p/gcfg/types"
)

type tag struct {
	ident   string
	intMode string
}

func newTag(ts string) tag {
	t := tag{}
	s := strings.Split(ts, ",")
	t.ident = s[0]
	for _, tse := range s[1:] {
		if strings.HasPrefix(tse, "int=") {
			t.intMode = tse[len("int="):]
		}
	}
	return t
}

func fieldFold(v reflect.Value, name string) (reflect.Value, tag) {
	var n string
	r0, _ := utf8.DecodeRuneInString(name)
	if unicode.IsLetter(r0) && !unicode.IsLower(r0) && !unicode.IsUpper(r0) {
		n = "X"
	}
	n += strings.Replace(name, "-", "_", -1)
	f, ok := v.Type().FieldByNameFunc(func(fieldName string) bool {
		if !v.FieldByName(fieldName).CanSet() {
			return false
		}
		f, _ := v.Type().FieldByName(fieldName)
		t := newTag(f.Tag.Get("gcfg"))
		if t.ident != "" {
			return strings.EqualFold(t.ident, name)
		}
		return strings.EqualFold(n, fieldName)
	})
	if !ok {
		return reflect.Value{}, tag{}
	}
	return v.FieldByName(f.Name), newTag(f.Tag.Get("gcfg"))
}

type setter func(destp interface{}, blank bool, val string, t tag) error

var errUnsupportedType = fmt.Errorf("unsupported type")
var errBlankUnsupported = fmt.Errorf("blank value not supported for type")

var setters = []setter{
	typeSetter, textUnmarshalerSetter, kindSetter, scanSetter,
}

func textUnmarshalerSetter(d interface{}, blank bool, val string, t tag) error {
	dtu, ok := d.(textUnmarshaler)
	if !ok {
		return errUnsupportedType
	}
	if blank {
		return errBlankUnsupported
	}
	return dtu.UnmarshalText([]byte(val))
}

func boolSetter(d interface{}, blank bool, val string, t tag) error {
	if blank {
		reflect.ValueOf(d).Elem().Set(reflect.ValueOf(true))
		return nil
	}
	b, err := types.ParseBool(val)
	if err == nil {
		reflect.ValueOf(d).Elem().Set(reflect.ValueOf(b))
	}
	return err
}

func intMode(mode string) types.IntMode {
	var m types.IntMode
	if strings.ContainsAny(mode, "dD") {
		m |= types.Dec
	}
	if strings.ContainsAny(mode, "hH") {
		m |= types.Hex
	}
	if strings.ContainsAny(mode, "oO") {
		m |= types.Oct
	}
	return m
}

var typeModes = map[reflect.Type]types.IntMode{
	reflect.TypeOf(int(0)):    types.Dec | types.Hex,
	reflect.TypeOf(int8(0)):   types.Dec | types.Hex,
	reflect.TypeOf(int16(0)):  types.Dec | types.Hex,
	reflect.TypeOf(int32(0)):  types.Dec | types.Hex,
	reflect.TypeOf(int64(0)):  types.Dec | types.Hex,
	reflect.TypeOf(uint(0)):   types.Dec | types.Hex,
	reflect.TypeOf(uint8(0)):  types.Dec | types.Hex,
	reflect.TypeOf(uint16(0)): types.Dec | types.Hex,
	reflect.TypeOf(uint32(0)): types.Dec | types.Hex,
	reflect.TypeOf(uint64(0)): types.Dec | types.Hex,
	// use default mode (allow dec/hex/oct) for uintptr type
	reflect.TypeOf(big.Int{}): types.Dec | types.Hex,
}

func intModeDefault(t reflect.Type) types.IntMode {
	m, ok := typeModes[t]
	if !ok {
		m = types.Dec | types.Hex | types.Oct
	}
	return m
}

func intSetter(d interface{}, blank bool, val string, t tag) error {
	if blank {
		return errBlankUnsupported
	}
	mode := intMode(t.intMode)
	if mode == 0 {
		mode = intModeDefault(reflect.TypeOf(d).Elem())
	}
	return types.ParseInt(d, val, mode)
}

func stringSetter(d interface{}, blank bool, val string, t tag) error {
	if blank {
		return errBlankUnsupported
	}
	dsp, ok := d.(*string)
	if !ok {
		return errUnsupportedType
	}
	*dsp = val
	return nil
}

var kindSetters = map[reflect.Kind]setter{
	reflect.String:  stringSetter,
	reflect.Bool:    boolSetter,
	reflect.Int:     intSetter,
	reflect.Int8:    intSetter,
	reflect.Int16:   intSetter,
	reflect.Int32:   intSetter,
	reflect.Int64:   intSetter,
	reflect.Uint:    intSetter,
	reflect.Uint8:   intSetter,
	reflect.Uint16:  intSetter,
	reflect.Uint32:  intSetter,
	reflect.Uint64:  intSetter,
	reflect.Uintptr: intSetter,
}

var typeSetters = map[reflect.Type]setter{
	reflect.TypeOf(big.Int{}): intSetter,
}

func typeSetter(d interface{}, blank bool, val string, tt tag) error {
	t := reflect.ValueOf(d).Type().Elem()
	setter, ok := typeSetters[t]
	if !ok {
		return errUnsupportedType
	}
	return setter(d, blank, val, tt)
}

func kindSetter(d interface{}, blank bool, val string, tt tag) error {
	k := reflect.ValueOf(d).Type().Elem().Kind()
	setter, ok := kindSetters[k]
	if !ok {
		return errUnsupportedType
	}
	return setter(d, blank, val, tt)
}

func scanSetter(d interface{}, blank bool, val string, tt tag) error {
	if blank {
		return errBlankUnsupported
	}
	return types.ScanFully(d, val, 'v')
}

func set(cfg interface{}, sect, sub, name string, blank bool, value string) error {
	vPCfg := reflect.ValueOf(cfg)
	if vPCfg.Kind() != reflect.Ptr || vPCfg.Elem().Kind() != reflect.Struct {
		panic(fmt.Errorf("config must be a pointer to a struct"))
	}
	vCfg := vPCfg.Elem()
	vSect, _ := fieldFold(vCfg, sect)
	if !vSect.IsValid() {
		return fmt.Errorf("invalid section: section %q", sect)
	}
	if vSect.Kind() == reflect.Map {
		vst := vSect.Type()
		if vst.Key().Kind() != reflect.String ||
			vst.Elem().Kind() != reflect.Ptr ||
			vst.Elem().Elem().Kind() != reflect.Struct {
			panic(fmt.Errorf("map field for section must have string keys and "+
				" pointer-to-struct values: section %q", sect))
		}
		if vSect.IsNil() {
			vSect.Set(reflect.MakeMap(vst))
		}
		k := reflect.ValueOf(sub)
		pv := vSect.MapIndex(k)
		if !pv.IsValid() {
			vType := vSect.Type().Elem().Elem()
			pv = reflect.New(vType)
			vSect.SetMapIndex(k, pv)
		}
		vSect = pv.Elem()
	} else if vSect.Kind() != reflect.Struct {
		panic(fmt.Errorf("field for section must be a map or a struct: "+
			"section %q", sect))
	} else if sub != "" {
		return fmt.Errorf("invalid subsection: "+
			"section %q subsection %q", sect, sub)
	}
	vVar, t := fieldFold(vSect, name)
	if !vVar.IsValid() {
		return fmt.Errorf("invalid variable: "+
			"section %q subsection %q variable %q", sect, sub, name)
	}
	// vVal is either single-valued var, or newly allocated value within multi-valued var
	var vVal reflect.Value
	// multi-value if unnamed slice type
	isMulti := vVar.Type().Name() == "" && vVar.Kind() == reflect.Slice
	if isMulti && blank {
		vVar.Set(reflect.Zero(vVar.Type()))
		return nil
	}
	if isMulti {
		vVal = reflect.New(vVar.Type().Elem()).Elem()
	} else {
		vVal = vVar
	}
	isDeref := vVal.Type().Name() == "" && vVal.Type().Kind() == reflect.Ptr
	isNew := isDeref && vVal.IsNil()
	// vAddr is address of value to set (dereferenced & allocated as needed)
	var vAddr reflect.Value
	switch {
	case isNew:
		vAddr = reflect.New(vVal.Type().Elem())
	case isDeref && !isNew:
		vAddr = vVal
	default:
		vAddr = vVal.Addr()
	}
	vAddrI := vAddr.Interface()
	err, ok := error(nil), false
	for _, s := range setters {
		err = s(vAddrI, blank, value, t)
		if err == nil {
			ok = true
			break
		}
		if err != errUnsupportedType {
			return err
		}
	}
	if !ok {
		// in case all setters returned errUnsupportedType
		return err
	}
	if isNew { // set reference if it was dereferenced and newly allocated
		vVal.Set(vAddr)
	}
	if isMulti { // append if multi-valued
		vVar.Set(reflect.Append(vVar, vVal))
	}
	return nil
}
