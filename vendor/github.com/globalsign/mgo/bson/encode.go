// BSON library for Go
//
// Copyright (c) 2010-2012 - Gustavo Niemeyer <gustavo@niemeyer.net>
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// gobson - BSON library for Go.

package bson

import (
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"time"
)

// --------------------------------------------------------------------------
// Some internal infrastructure.

var (
	typeBinary         = reflect.TypeOf(Binary{})
	typeObjectId       = reflect.TypeOf(ObjectId(""))
	typeDBPointer      = reflect.TypeOf(DBPointer{"", ObjectId("")})
	typeSymbol         = reflect.TypeOf(Symbol(""))
	typeMongoTimestamp = reflect.TypeOf(MongoTimestamp(0))
	typeOrderKey       = reflect.TypeOf(MinKey)
	typeDocElem        = reflect.TypeOf(DocElem{})
	typeRawDocElem     = reflect.TypeOf(RawDocElem{})
	typeRaw            = reflect.TypeOf(Raw{})
	typeRawPtr         = reflect.PtrTo(reflect.TypeOf(Raw{}))
	typeURL            = reflect.TypeOf(url.URL{})
	typeTime           = reflect.TypeOf(time.Time{})
	typeString         = reflect.TypeOf("")
	typeJSONNumber     = reflect.TypeOf(json.Number(""))
	typeTimeDuration   = reflect.TypeOf(time.Duration(0))
)

var (
	// spec for []uint8 or []byte encoding
	arrayOps = map[string]bool{
		"$in":  true,
		"$nin": true,
		"$all": true,
	}
)

const itoaCacheSize = 32

const (
	getterUnknown = iota
	getterNone
	getterTypeVal
	getterTypePtr
	getterAddr
)

var itoaCache []string

var getterStyles map[reflect.Type]int
var getterIface reflect.Type
var getterMutex sync.RWMutex

func init() {
	itoaCache = make([]string, itoaCacheSize)
	for i := 0; i != itoaCacheSize; i++ {
		itoaCache[i] = strconv.Itoa(i)
	}
	var iface Getter
	getterIface = reflect.TypeOf(&iface).Elem()
	getterStyles = make(map[reflect.Type]int)
}

func itoa(i int) string {
	if i < itoaCacheSize {
		return itoaCache[i]
	}
	return strconv.Itoa(i)
}

func getterStyle(outt reflect.Type) int {
	getterMutex.RLock()
	style := getterStyles[outt]
	getterMutex.RUnlock()
	if style != getterUnknown {
		return style
	}

	getterMutex.Lock()
	defer getterMutex.Unlock()
	if outt.Implements(getterIface) {
		vt := outt
		for vt.Kind() == reflect.Ptr {
			vt = vt.Elem()
		}
		if vt.Implements(getterIface) {
			style = getterTypeVal
		} else {
			style = getterTypePtr
		}
	} else if reflect.PtrTo(outt).Implements(getterIface) {
		style = getterAddr
	} else {
		style = getterNone
	}
	getterStyles[outt] = style
	return style
}

func getGetter(outt reflect.Type, out reflect.Value) Getter {
	style := getterStyle(outt)
	if style == getterNone {
		return nil
	}
	if style == getterAddr {
		if !out.CanAddr() {
			return nil
		}
		return out.Addr().Interface().(Getter)
	}
	if style == getterTypeVal && out.Kind() == reflect.Ptr && out.IsNil() {
		return nil
	}
	return out.Interface().(Getter)
}

// --------------------------------------------------------------------------
// Marshaling of the document value itself.

type encoder struct {
	out []byte
}

func (e *encoder) addDoc(v reflect.Value) {
	for {
		if vi, ok := v.Interface().(Getter); ok {
			getv, err := vi.GetBSON()
			if err != nil {
				panic(err)
			}
			v = reflect.ValueOf(getv)
			continue
		}
		if v.Kind() == reflect.Ptr {
			v = v.Elem()
			continue
		}
		break
	}

	if v.Type() == typeRaw {
		raw := v.Interface().(Raw)
		if raw.Kind != 0x03 && raw.Kind != 0x00 {
			panic("Attempted to marshal Raw kind " + strconv.Itoa(int(raw.Kind)) + " as a document")
		}
		if len(raw.Data) == 0 {
			panic("Attempted to marshal empty Raw document")
		}
		e.addBytes(raw.Data...)
		return
	}

	start := e.reserveInt32()

	switch v.Kind() {
	case reflect.Map:
		e.addMap(v)
	case reflect.Struct:
		e.addStruct(v)
	case reflect.Array, reflect.Slice:
		e.addSlice(v)
	default:
		panic("Can't marshal " + v.Type().String() + " as a BSON document")
	}

	e.addBytes(0)
	e.setInt32(start, int32(len(e.out)-start))
}

func (e *encoder) addMap(v reflect.Value) {
	for _, k := range v.MapKeys() {
		e.addElem(fmt.Sprint(k), v.MapIndex(k), false)
	}
}

func (e *encoder) addStruct(v reflect.Value) {
	sinfo, err := getStructInfo(v.Type())
	if err != nil {
		panic(err)
	}
	var value reflect.Value
	if sinfo.InlineMap >= 0 {
		m := v.Field(sinfo.InlineMap)
		if m.Len() > 0 {
			for _, k := range m.MapKeys() {
				ks := k.String()
				if _, found := sinfo.FieldsMap[ks]; found {
					panic(fmt.Sprintf("Can't have key %q in inlined map; conflicts with struct field", ks))
				}
				e.addElem(ks, m.MapIndex(k), false)
			}
		}
	}
	for _, info := range sinfo.FieldsList {
		if info.Inline == nil {
			value = v.Field(info.Num)
		} else {
			// as pointers to struct are allowed here,
			// there is no guarantee that pointer won't be nil.
			//
			// It is expected allowed behaviour
			// so info.Inline MAY consist index to a nil pointer
			// and that is why we safely call v.FieldByIndex and just continue on panic
			field, errField := safeFieldByIndex(v, info.Inline)
			if errField != nil {
				continue
			}

			value = field
		}
		if info.OmitEmpty && isZero(value) {
			continue
		}
		if useRespectNilValues &&
			(value.Kind() == reflect.Slice || value.Kind() == reflect.Map) &&
			value.IsNil() {
			e.addElem(info.Key, reflect.ValueOf(nil), info.MinSize)
			continue
		}
		e.addElem(info.Key, value, info.MinSize)
	}
}

func safeFieldByIndex(v reflect.Value, index []int) (result reflect.Value, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			switch r := recovered.(type) {
			case string:
				err = fmt.Errorf("%s", r)
			case error:
				err = r
			}
		}
	}()

	result = v.FieldByIndex(index)
	return
}

func isZero(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.String:
		return len(v.String()) == 0
	case reflect.Ptr, reflect.Interface:
		return v.IsNil()
	case reflect.Slice:
		return v.Len() == 0
	case reflect.Map:
		return v.Len() == 0
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Struct:
		vt := v.Type()
		if vt == typeTime {
			return v.Interface().(time.Time).IsZero()
		}
		for i := 0; i < v.NumField(); i++ {
			if vt.Field(i).PkgPath != "" && !vt.Field(i).Anonymous {
				continue // Private field
			}
			if !isZero(v.Field(i)) {
				return false
			}
		}
		return true
	}
	return false
}

func (e *encoder) addSlice(v reflect.Value) {
	vi := v.Interface()
	if d, ok := vi.(D); ok {
		for _, elem := range d {
			e.addElem(elem.Name, reflect.ValueOf(elem.Value), false)
		}
		return
	}
	if d, ok := vi.(RawD); ok {
		for _, elem := range d {
			e.addElem(elem.Name, reflect.ValueOf(elem.Value), false)
		}
		return
	}
	l := v.Len()
	et := v.Type().Elem()
	if et == typeDocElem {
		for i := 0; i < l; i++ {
			elem := v.Index(i).Interface().(DocElem)
			e.addElem(elem.Name, reflect.ValueOf(elem.Value), false)
		}
		return
	}
	if et == typeRawDocElem {
		for i := 0; i < l; i++ {
			elem := v.Index(i).Interface().(RawDocElem)
			e.addElem(elem.Name, reflect.ValueOf(elem.Value), false)
		}
		return
	}
	for i := 0; i < l; i++ {
		e.addElem(itoa(i), v.Index(i), false)
	}
}

// --------------------------------------------------------------------------
// Marshaling of elements in a document.

func (e *encoder) addElemName(kind byte, name string) {
	e.addBytes(kind)
	e.addBytes([]byte(name)...)
	e.addBytes(0)
}

func (e *encoder) addElem(name string, v reflect.Value, minSize bool) {

	if !v.IsValid() {
		e.addElemName(0x0A, name)
		return
	}

	if getter := getGetter(v.Type(), v); getter != nil {
		getv, err := getter.GetBSON()
		if err != nil {
			panic(err)
		}
		e.addElem(name, reflect.ValueOf(getv), minSize)
		return
	}

	switch v.Kind() {

	case reflect.Interface:
		e.addElem(name, v.Elem(), minSize)

	case reflect.Ptr:
		e.addElem(name, v.Elem(), minSize)

	case reflect.String:
		s := v.String()
		switch v.Type() {
		case typeObjectId:
			if len(s) != 12 {
				panic("ObjectIDs must be exactly 12 bytes long (got " +
					strconv.Itoa(len(s)) + ")")
			}
			e.addElemName(0x07, name)
			e.addBytes([]byte(s)...)
		case typeSymbol:
			e.addElemName(0x0E, name)
			e.addStr(s)
		case typeJSONNumber:
			n := v.Interface().(json.Number)
			if i, err := n.Int64(); err == nil {
				e.addElemName(0x12, name)
				e.addInt64(i)
			} else if f, err := n.Float64(); err == nil {
				e.addElemName(0x01, name)
				e.addFloat64(f)
			} else {
				panic("failed to convert json.Number to a number: " + s)
			}
		default:
			e.addElemName(0x02, name)
			e.addStr(s)
		}

	case reflect.Float32, reflect.Float64:
		e.addElemName(0x01, name)
		e.addFloat64(v.Float())

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		u := v.Uint()
		if int64(u) < 0 {
			panic("BSON has no uint64 type, and value is too large to fit correctly in an int64")
		} else if u <= math.MaxInt32 && (minSize || v.Kind() <= reflect.Uint32) {
			e.addElemName(0x10, name)
			e.addInt32(int32(u))
		} else {
			e.addElemName(0x12, name)
			e.addInt64(int64(u))
		}

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch v.Type() {
		case typeMongoTimestamp:
			e.addElemName(0x11, name)
			e.addInt64(v.Int())

		case typeOrderKey:
			if v.Int() == int64(MaxKey) {
				e.addElemName(0x7F, name)
			} else {
				e.addElemName(0xFF, name)
			}
		case typeTimeDuration:
			// Stored as int64
			e.addElemName(0x12, name)

			e.addInt64(int64(v.Int() / 1e6))
		default:
			i := v.Int()
			if (minSize || v.Type().Kind() != reflect.Int64) && i >= math.MinInt32 && i <= math.MaxInt32 {
				// It fits into an int32, encode as such.
				e.addElemName(0x10, name)
				e.addInt32(int32(i))
			} else {
				e.addElemName(0x12, name)
				e.addInt64(i)
			}
		}

	case reflect.Bool:
		e.addElemName(0x08, name)
		if v.Bool() {
			e.addBytes(1)
		} else {
			e.addBytes(0)
		}

	case reflect.Map:
		e.addElemName(0x03, name)
		e.addDoc(v)

	case reflect.Slice:
		vt := v.Type()
		et := vt.Elem()
		if et.Kind() == reflect.Uint8 {
			if arrayOps[name] {
				e.addElemName(0x04, name)
				e.addDoc(v)
			} else {
				e.addElemName(0x05, name)
				e.addBinary(0x00, v.Bytes())
			}
		} else if et == typeDocElem || et == typeRawDocElem {
			e.addElemName(0x03, name)
			e.addDoc(v)
		} else {
			e.addElemName(0x04, name)
			e.addDoc(v)
		}

	case reflect.Array:
		et := v.Type().Elem()
		if et.Kind() == reflect.Uint8 {
			if arrayOps[name] {
				e.addElemName(0x04, name)
				e.addDoc(v)
			} else {
				e.addElemName(0x05, name)
				if v.CanAddr() {
					e.addBinary(0x00, v.Slice(0, v.Len()).Interface().([]byte))
				} else {
					n := v.Len()
					e.addInt32(int32(n))
					e.addBytes(0x00)
					for i := 0; i < n; i++ {
						el := v.Index(i)
						e.addBytes(byte(el.Uint()))
					}
				}
			}
		} else {
			e.addElemName(0x04, name)
			e.addDoc(v)
		}

	case reflect.Struct:
		switch s := v.Interface().(type) {

		case Raw:
			kind := s.Kind
			if kind == 0x00 {
				kind = 0x03
			}
			if len(s.Data) == 0 && kind != 0x06 && kind != 0x0A && kind != 0xFF && kind != 0x7F {
				panic("Attempted to marshal empty Raw document")
			}
			e.addElemName(kind, name)
			e.addBytes(s.Data...)

		case Binary:
			e.addElemName(0x05, name)
			e.addBinary(s.Kind, s.Data)

		case Decimal128:
			e.addElemName(0x13, name)
			e.addInt64(int64(s.l))
			e.addInt64(int64(s.h))

		case DBPointer:
			e.addElemName(0x0C, name)
			e.addStr(s.Namespace)
			if len(s.Id) != 12 {
				panic("ObjectIDs must be exactly 12 bytes long (got " +
					strconv.Itoa(len(s.Id)) + ")")
			}
			e.addBytes([]byte(s.Id)...)

		case RegEx:
			e.addElemName(0x0B, name)
			e.addCStr(s.Pattern)
			options := runes(s.Options)
			sort.Sort(options)
			e.addCStr(string(options))

		case JavaScript:
			if s.Scope == nil {
				e.addElemName(0x0D, name)
				e.addStr(s.Code)
			} else {
				e.addElemName(0x0F, name)
				start := e.reserveInt32()
				e.addStr(s.Code)
				e.addDoc(reflect.ValueOf(s.Scope))
				e.setInt32(start, int32(len(e.out)-start))
			}

		case time.Time:
			// MongoDB handles timestamps as milliseconds.
			e.addElemName(0x09, name)
			e.addInt64(s.Unix()*1000 + int64(s.Nanosecond()/1e6))

		case url.URL:
			e.addElemName(0x02, name)
			e.addStr(s.String())

		case undefined:
			e.addElemName(0x06, name)

		default:
			e.addElemName(0x03, name)
			e.addDoc(v)
		}

	default:
		panic("Can't marshal " + v.Type().String() + " in a BSON document")
	}
}

// -------------
// Helper method for sorting regex options
type runes []rune

func (a runes) Len() int           { return len(a) }
func (a runes) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a runes) Less(i, j int) bool { return a[i] < a[j] }

// --------------------------------------------------------------------------
// Marshaling of base types.

func (e *encoder) addBinary(subtype byte, v []byte) {
	if subtype == 0x02 {
		// Wonder how that brilliant idea came to life. Obsolete, luckily.
		e.addInt32(int32(len(v) + 4))
		e.addBytes(subtype)
		e.addInt32(int32(len(v)))
	} else {
		e.addInt32(int32(len(v)))
		e.addBytes(subtype)
	}
	e.addBytes(v...)
}

func (e *encoder) addStr(v string) {
	e.addInt32(int32(len(v) + 1))
	e.addCStr(v)
}

func (e *encoder) addCStr(v string) {
	e.addBytes([]byte(v)...)
	e.addBytes(0)
}

func (e *encoder) reserveInt32() (pos int) {
	pos = len(e.out)
	e.addBytes(0, 0, 0, 0)
	return pos
}

func (e *encoder) setInt32(pos int, v int32) {
	e.out[pos+0] = byte(v)
	e.out[pos+1] = byte(v >> 8)
	e.out[pos+2] = byte(v >> 16)
	e.out[pos+3] = byte(v >> 24)
}

func (e *encoder) addInt32(v int32) {
	u := uint32(v)
	e.addBytes(byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
}

func (e *encoder) addInt64(v int64) {
	u := uint64(v)
	e.addBytes(byte(u), byte(u>>8), byte(u>>16), byte(u>>24),
		byte(u>>32), byte(u>>40), byte(u>>48), byte(u>>56))
}

func (e *encoder) addFloat64(v float64) {
	e.addInt64(int64(math.Float64bits(v)))
}

func (e *encoder) addBytes(v ...byte) {
	e.out = append(e.out, v...)
}
