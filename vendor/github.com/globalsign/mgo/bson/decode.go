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
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"reflect"
	"strconv"
	"sync"
	"time"
)

type decoder struct {
	in      []byte
	i       int
	docType reflect.Type
}

var typeM = reflect.TypeOf(M{})

func newDecoder(in []byte) *decoder {
	return &decoder{in, 0, typeM}
}

// --------------------------------------------------------------------------
// Some helper functions.

func corrupted() {
	panic("Document is corrupted")
}

// --------------------------------------------------------------------------
// Unmarshaling of documents.

const (
	setterUnknown = iota
	setterNone
	setterType
	setterAddr
)

var setterStyles map[reflect.Type]int
var setterIface reflect.Type
var setterMutex sync.RWMutex

func init() {
	var iface Setter
	setterIface = reflect.TypeOf(&iface).Elem()
	setterStyles = make(map[reflect.Type]int)
}

func setterStyle(outt reflect.Type) int {
	setterMutex.RLock()
	style := setterStyles[outt]
	setterMutex.RUnlock()
	if style != setterUnknown {
		return style
	}

	setterMutex.Lock()
	defer setterMutex.Unlock()
	if outt.Implements(setterIface) {
		style = setterType
	} else if reflect.PtrTo(outt).Implements(setterIface) {
		style = setterAddr
	} else {
		style = setterNone
	}
	setterStyles[outt] = style
	return style
}

func getSetter(outt reflect.Type, out reflect.Value) Setter {
	style := setterStyle(outt)
	if style == setterNone {
		return nil
	}
	if style == setterAddr {
		if !out.CanAddr() {
			return nil
		}
		out = out.Addr()
	} else if outt.Kind() == reflect.Ptr && out.IsNil() {
		out.Set(reflect.New(outt.Elem()))
	}
	return out.Interface().(Setter)
}

func clearMap(m reflect.Value) {
	var none reflect.Value
	for _, k := range m.MapKeys() {
		m.SetMapIndex(k, none)
	}
}

func (d *decoder) readDocTo(out reflect.Value) {
	var elemType reflect.Type
	outt := out.Type()
	outk := outt.Kind()

	for {
		if outk == reflect.Ptr && out.IsNil() {
			out.Set(reflect.New(outt.Elem()))
		}
		if setter := getSetter(outt, out); setter != nil {
			raw := d.readRaw(ElementDocument)
			err := setter.SetBSON(raw)
			if _, ok := err.(*TypeError); err != nil && !ok {
				panic(err)
			}
			return
		}
		if outk == reflect.Ptr {
			out = out.Elem()
			outt = out.Type()
			outk = out.Kind()
			continue
		}
		break
	}

	var fieldsMap map[string]fieldInfo
	var inlineMap reflect.Value
	if outt == typeRaw {
		out.Set(reflect.ValueOf(d.readRaw(ElementDocument)))
		return
	}

	origout := out
	if outk == reflect.Interface {
		if d.docType.Kind() == reflect.Map {
			mv := reflect.MakeMap(d.docType)
			out.Set(mv)
			out = mv
		} else {
			dv := reflect.New(d.docType).Elem()
			out.Set(dv)
			out = dv
		}
		outt = out.Type()
		outk = outt.Kind()
	}

	docType := d.docType
	keyType := typeString
	convertKey := false
	switch outk {
	case reflect.Map:
		keyType = outt.Key()
		if keyType != typeString {
			convertKey = true
		}
		elemType = outt.Elem()
		if elemType == typeIface {
			d.docType = outt
		}
		if out.IsNil() {
			out.Set(reflect.MakeMap(out.Type()))
		} else if out.Len() > 0 {
			clearMap(out)
		}
	case reflect.Struct:
		sinfo, err := getStructInfo(out.Type())
		if err != nil {
			panic(err)
		}
		fieldsMap = sinfo.FieldsMap
		out.Set(sinfo.Zero)
		if sinfo.InlineMap != -1 {
			inlineMap = out.Field(sinfo.InlineMap)
			if !inlineMap.IsNil() && inlineMap.Len() > 0 {
				clearMap(inlineMap)
			}
			elemType = inlineMap.Type().Elem()
			if elemType == typeIface {
				d.docType = inlineMap.Type()
			}
		}
	case reflect.Slice:
		switch outt.Elem() {
		case typeDocElem:
			origout.Set(d.readDocElems(outt))
			return
		case typeRawDocElem:
			origout.Set(d.readRawDocElems(outt))
			return
		}
		fallthrough
	default:
		panic("Unsupported document type for unmarshalling: " + out.Type().String())
	}

	end := int(d.readInt32())
	end += d.i - 4
	if end <= d.i || end > len(d.in) || d.in[end-1] != '\x00' {
		corrupted()
	}
	for d.in[d.i] != '\x00' {
		kind := d.readByte()
		name := d.readCStr()
		if d.i >= end {
			corrupted()
		}

		switch outk {
		case reflect.Map:
			e := reflect.New(elemType).Elem()
			if d.readElemTo(e, kind) {
				k := reflect.ValueOf(name)
				if convertKey {
					mapKeyType := out.Type().Key()
					mapKeyKind := mapKeyType.Kind()

					switch mapKeyKind {
					case reflect.Int:
						fallthrough
					case reflect.Int8:
						fallthrough
					case reflect.Int16:
						fallthrough
					case reflect.Int32:
						fallthrough
					case reflect.Int64:
						fallthrough
					case reflect.Uint:
						fallthrough
					case reflect.Uint8:
						fallthrough
					case reflect.Uint16:
						fallthrough
					case reflect.Uint32:
						fallthrough
					case reflect.Uint64:
						fallthrough
					case reflect.Float32:
						fallthrough
					case reflect.Float64:
						parsed := d.parseMapKeyAsFloat(k, mapKeyKind)
						k = reflect.ValueOf(parsed)
					case reflect.String:
						mapKeyType = keyType
					default:
						panic("BSON map must have string or decimal keys. Got: " + outt.String())
					}

					k = k.Convert(mapKeyType)
				}
				out.SetMapIndex(k, e)
			}
		case reflect.Struct:
			if info, ok := fieldsMap[name]; ok {
				if info.Inline == nil {
					d.readElemTo(out.Field(info.Num), kind)
				} else {
					d.readElemTo(out.FieldByIndex(info.Inline), kind)
				}
			} else if inlineMap.IsValid() {
				if inlineMap.IsNil() {
					inlineMap.Set(reflect.MakeMap(inlineMap.Type()))
				}
				e := reflect.New(elemType).Elem()
				if d.readElemTo(e, kind) {
					inlineMap.SetMapIndex(reflect.ValueOf(name), e)
				}
			} else {
				d.dropElem(kind)
			}
		case reflect.Slice:
		}

		if d.i >= end {
			corrupted()
		}
	}
	d.i++ // '\x00'
	if d.i != end {
		corrupted()
	}
	d.docType = docType
}

func (decoder) parseMapKeyAsFloat(k reflect.Value, mapKeyKind reflect.Kind) float64 {
	parsed, err := strconv.ParseFloat(k.String(), 64)
	if err != nil {
		panic("Map key is defined to be a decimal type (" + mapKeyKind.String() + ") but got error " +
			err.Error())
	}

	return parsed
}

func (d *decoder) readArrayDocTo(out reflect.Value) {
	end := int(d.readInt32())
	end += d.i - 4
	if end <= d.i || end > len(d.in) || d.in[end-1] != '\x00' {
		corrupted()
	}
	i := 0
	l := out.Len()
	for d.in[d.i] != '\x00' {
		if i >= l {
			panic("Length mismatch on array field")
		}
		kind := d.readByte()
		for d.i < end && d.in[d.i] != '\x00' {
			d.i++
		}
		if d.i >= end {
			corrupted()
		}
		d.i++
		d.readElemTo(out.Index(i), kind)
		if d.i >= end {
			corrupted()
		}
		i++
	}
	if i != l {
		panic("Length mismatch on array field")
	}
	d.i++ // '\x00'
	if d.i != end {
		corrupted()
	}
}

func (d *decoder) readSliceDoc(t reflect.Type) interface{} {
	tmp := make([]reflect.Value, 0, 8)
	elemType := t.Elem()
	if elemType == typeRawDocElem {
		d.dropElem(ElementArray)
		return reflect.Zero(t).Interface()
	}
	if elemType == typeRaw {
		return d.readSliceOfRaw()
	}

	end := int(d.readInt32())
	end += d.i - 4
	if end <= d.i || end > len(d.in) || d.in[end-1] != '\x00' {
		corrupted()
	}
	for d.in[d.i] != '\x00' {
		kind := d.readByte()
		for d.i < end && d.in[d.i] != '\x00' {
			d.i++
		}
		if d.i >= end {
			corrupted()
		}
		d.i++
		e := reflect.New(elemType).Elem()
		if d.readElemTo(e, kind) {
			tmp = append(tmp, e)
		}
		if d.i >= end {
			corrupted()
		}
	}
	d.i++ // '\x00'
	if d.i != end {
		corrupted()
	}

	n := len(tmp)
	slice := reflect.MakeSlice(t, n, n)
	for i := 0; i != n; i++ {
		slice.Index(i).Set(tmp[i])
	}
	return slice.Interface()
}

func BSONElementSize(kind byte, offset int, buffer []byte) (int, error) {
	switch kind {
	case ElementFloat64: // Float64
		return 8, nil
	case ElementJavaScriptWithoutScope: // JavaScript without scope
		fallthrough
	case ElementSymbol: // Symbol
		fallthrough
	case ElementString: // UTF-8 string
		size, err := getSize(offset, buffer)
		if err != nil {
			return 0, err
		}
		if size < 1 {
			return 0, errors.New("String size can't be less then one byte")
		}
		size += 4
		if offset+size > len(buffer) {
			return 0, io.ErrUnexpectedEOF
		}
		if buffer[offset+size-1] != 0 {
			return 0, errors.New("Invalid string: non zero-terminated")
		}
		return size, nil
	case ElementArray: // Array
		fallthrough
	case ElementDocument: // Document
		size, err := getSize(offset, buffer)
		if err != nil {
			return 0, err
		}
		if size < 5 {
			return 0, errors.New("Declared document size is too small")
		}
		return size, nil
	case ElementBinary: // Binary
		size, err := getSize(offset, buffer)
		if err != nil {
			return 0, err
		}
		if size < 0 {
			return 0, errors.New("Binary data size can't be negative")
		}
		return size + 5, nil
	case Element06: // Undefined (obsolete, but still seen in the wild)
		return 0, nil
	case ElementObjectId: // ObjectId
		return 12, nil
	case ElementBool: // Bool
		return 1, nil
	case ElementDatetime: // Timestamp
		return 8, nil
	case ElementNil: // Nil
		return 0, nil
	case ElementRegEx: // RegEx
		end := offset
		for i := 0; i < 2; i++ {
			for end < len(buffer) && buffer[end] != '\x00' {
				end++
			}
			end++
		}
		if end > len(buffer) {
			return 0, io.ErrUnexpectedEOF
		}
		return end - offset, nil
	case ElementDBPointer: // DBPointer
		size, err := getSize(offset, buffer)
		if err != nil {
			return 0, err
		}
		if size < 1 {
			return 0, errors.New("String size can't be less then one byte")
		}
		return size + 12 + 4, nil
	case ElementJavaScriptWithScope: // JavaScript with scope
		size, err := getSize(offset, buffer)
		if err != nil {
			return 0, err
		}
		if size < 4+5+5 {
			return 0, errors.New("Declared document element is too small")
		}
		return size, nil
	case ElementInt32: // Int32
		return 4, nil
	case ElementTimestamp: // Mongo-specific timestamp
		return 8, nil
	case ElementInt64: // Int64
		return 8, nil
	case ElementDecimal128: // Decimal128
		return 16, nil
	case ElementMaxKey: // Max key
		return 0, nil
	case ElementMinKey: // Min key
		return 0, nil
	default:
		return 0, errors.New(fmt.Sprintf("Unknown element kind (0x%02X)", kind))
	}
}

func (d *decoder) readRaw(kind byte) Raw {
	size, err := BSONElementSize(kind, d.i, d.in)
	if err != nil {
		corrupted()
	}
	if d.i+size > len(d.in) {
		corrupted()
	}
	d.i += size
	return Raw{
		Kind: kind,
		Data: d.in[d.i-size : d.i],
	}
}

func (d *decoder) readSliceOfRaw() interface{} {
	tmp := make([]Raw, 0, 8)
	end := int(d.readInt32())
	end += d.i - 4
	if end <= d.i || end > len(d.in) || d.in[end-1] != '\x00' {
		corrupted()
	}
	for d.in[d.i] != '\x00' {
		kind := d.readByte()
		for d.i < end && d.in[d.i] != '\x00' {
			d.i++
		}
		if d.i >= end {
			corrupted()
		}
		d.i++
		e := d.readRaw(kind)
		tmp = append(tmp, e)
		if d.i >= end {
			corrupted()
		}
	}
	d.i++ // '\x00'
	if d.i != end {
		corrupted()
	}
	return tmp
}

var typeSlice = reflect.TypeOf([]interface{}{})
var typeIface = typeSlice.Elem()

func (d *decoder) readDocElems(typ reflect.Type) reflect.Value {
	docType := d.docType
	d.docType = typ
	slice := make([]DocElem, 0, 8)
	d.readDocWith(func(kind byte, name string) {
		e := DocElem{Name: name}
		v := reflect.ValueOf(&e.Value)
		if d.readElemTo(v.Elem(), kind) {
			slice = append(slice, e)
		}
	})
	slicev := reflect.New(typ).Elem()
	slicev.Set(reflect.ValueOf(slice))
	d.docType = docType
	return slicev
}

func (d *decoder) readRawDocElems(typ reflect.Type) reflect.Value {
	docType := d.docType
	d.docType = typ
	slice := make([]RawDocElem, 0, 8)
	d.readDocWith(func(kind byte, name string) {
		e := RawDocElem{Name: name, Value: d.readRaw(kind)}
		slice = append(slice, e)
	})
	slicev := reflect.New(typ).Elem()
	slicev.Set(reflect.ValueOf(slice))
	d.docType = docType
	return slicev
}

func (d *decoder) readDocWith(f func(kind byte, name string)) {
	end := int(d.readInt32())
	end += d.i - 4
	if end <= d.i || end > len(d.in) || d.in[end-1] != '\x00' {
		corrupted()
	}
	for d.in[d.i] != '\x00' {
		kind := d.readByte()
		name := d.readCStr()
		if d.i >= end {
			corrupted()
		}
		f(kind, name)
		if d.i >= end {
			corrupted()
		}
	}
	d.i++ // '\x00'
	if d.i != end {
		corrupted()
	}
}

// --------------------------------------------------------------------------
// Unmarshaling of individual elements within a document.
func (d *decoder) dropElem(kind byte) {
	size, err := BSONElementSize(kind, d.i, d.in)
	if err != nil {
		corrupted()
	}
	if d.i+size > len(d.in) {
		corrupted()
	}
	d.i += size
}

// Attempt to decode an element from the document and put it into out.
// If the types are not compatible, the returned ok value will be
// false and out will be unchanged.
func (d *decoder) readElemTo(out reflect.Value, kind byte) (good bool) {
	outt := out.Type()

	if outt == typeRaw {
		out.Set(reflect.ValueOf(d.readRaw(kind)))
		return true
	}

	if outt == typeRawPtr {
		raw := d.readRaw(kind)
		out.Set(reflect.ValueOf(&raw))
		return true
	}

	if kind == ElementDocument {
		// Delegate unmarshaling of documents.
		outt := out.Type()
		outk := out.Kind()
		switch outk {
		case reflect.Interface, reflect.Ptr, reflect.Struct, reflect.Map:
			d.readDocTo(out)
			return true
		}
		if setterStyle(outt) != setterNone {
			d.readDocTo(out)
			return true
		}
		if outk == reflect.Slice {
			switch outt.Elem() {
			case typeDocElem:
				out.Set(d.readDocElems(outt))
			case typeRawDocElem:
				out.Set(d.readRawDocElems(outt))
			default:
				d.dropElem(kind)
			}
			return true
		}
		d.dropElem(kind)
		return true
	}

	if setter := getSetter(outt, out); setter != nil {
		err := setter.SetBSON(d.readRaw(kind))
		if err == ErrSetZero {
			out.Set(reflect.Zero(outt))
			return true
		}
		if err == nil {
			return true
		}
		if _, ok := err.(*TypeError); !ok {
			panic(err)
		}
		return false
	}

	var in interface{}

	switch kind {
	case ElementFloat64:
		in = d.readFloat64()
	case ElementString:
		in = d.readStr()
	case ElementDocument:
		panic("Can't happen. Handled above.")
	case ElementArray:
		outt := out.Type()
		if setterStyle(outt) != setterNone {
			// Skip the value so its data is handed to the setter below.
			d.dropElem(kind)
			break
		}
		for outt.Kind() == reflect.Ptr {
			outt = outt.Elem()
		}
		switch outt.Kind() {
		case reflect.Array:
			d.readArrayDocTo(out)
			return true
		case reflect.Slice:
			in = d.readSliceDoc(outt)
		default:
			in = d.readSliceDoc(typeSlice)
		}
	case ElementBinary:
		b := d.readBinary()
		if b.Kind == BinaryGeneric || b.Kind == BinaryBinaryOld {
			in = b.Data
		} else {
			in = b
		}
	case Element06: // Undefined (obsolete, but still seen in the wild)
		in = Undefined
	case ElementObjectId:
		in = ObjectId(d.readBytes(12))
	case ElementBool:
		in = d.readBool()
	case ElementDatetime: // Timestamp
		// MongoDB handles timestamps as milliseconds.
		i := d.readInt64()
		if i == -62135596800000 {
			in = time.Time{} // In UTC for convenience.
		} else {
			in = time.Unix(i/1e3, i%1e3*1e6).UTC()
		}
	case ElementNil:
		in = nil
	case ElementRegEx:
		in = d.readRegEx()
	case ElementDBPointer:
		in = DBPointer{Namespace: d.readStr(), Id: ObjectId(d.readBytes(12))}
	case ElementJavaScriptWithoutScope:
		in = JavaScript{Code: d.readStr()}
	case ElementSymbol:
		in = Symbol(d.readStr())
	case ElementJavaScriptWithScope:
		start := d.i
		l := int(d.readInt32())
		js := JavaScript{d.readStr(), make(M)}
		d.readDocTo(reflect.ValueOf(js.Scope))
		if d.i != start+l {
			corrupted()
		}
		in = js
	case ElementInt32:
		in = int(d.readInt32())
	case ElementTimestamp: // Mongo-specific timestamp
		in = MongoTimestamp(d.readInt64())
	case ElementInt64:
		switch out.Type() {
		case typeTimeDuration:
			in = time.Duration(time.Duration(d.readInt64()) * time.Millisecond)
		default:
			in = d.readInt64()
		}
	case ElementDecimal128:
		in = Decimal128{
			l: uint64(d.readInt64()),
			h: uint64(d.readInt64()),
		}
	case ElementMaxKey:
		in = MaxKey
	case ElementMinKey:
		in = MinKey
	default:
		panic(fmt.Sprintf("Unknown element kind (0x%02X)", kind))
	}

	if in == nil {
		out.Set(reflect.Zero(outt))
		return true
	}

	outk := outt.Kind()

	// Dereference and initialize pointer if necessary.
	first := true
	for outk == reflect.Ptr {
		if !out.IsNil() {
			out = out.Elem()
		} else {
			elem := reflect.New(outt.Elem())
			if first {
				// Only set if value is compatible.
				first = false
				defer func(out, elem reflect.Value) {
					if good {
						out.Set(elem)
					}
				}(out, elem)
			} else {
				out.Set(elem)
			}
			out = elem
		}
		outt = out.Type()
		outk = outt.Kind()
	}

	inv := reflect.ValueOf(in)
	if outt == inv.Type() {
		out.Set(inv)
		return true
	}

	switch outk {
	case reflect.Interface:
		out.Set(inv)
		return true
	case reflect.String:
		switch inv.Kind() {
		case reflect.String:
			out.SetString(inv.String())
			return true
		case reflect.Slice:
			if b, ok := in.([]byte); ok {
				out.SetString(string(b))
				return true
			}
		case reflect.Int, reflect.Int64:
			if outt == typeJSONNumber {
				out.SetString(strconv.FormatInt(inv.Int(), 10))
				return true
			}
		case reflect.Float64:
			if outt == typeJSONNumber {
				out.SetString(strconv.FormatFloat(inv.Float(), 'f', -1, 64))
				return true
			}
		}
	case reflect.Slice, reflect.Array:
		// Remember, array (0x04) slices are built with the correct
		// element type.  If we are here, must be a cross BSON kind
		// conversion (e.g. 0x05 unmarshalling on string).
		if outt.Elem().Kind() != reflect.Uint8 {
			break
		}
		switch inv.Kind() {
		case reflect.String:
			slice := []byte(inv.String())
			out.Set(reflect.ValueOf(slice))
			return true
		case reflect.Slice:
			switch outt.Kind() {
			case reflect.Array:
				reflect.Copy(out, inv)
			case reflect.Slice:
				out.SetBytes(inv.Bytes())
			}
			return true
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch inv.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			out.SetInt(inv.Int())
			return true
		case reflect.Float32, reflect.Float64:
			out.SetInt(int64(inv.Float()))
			return true
		case reflect.Bool:
			if inv.Bool() {
				out.SetInt(1)
			} else {
				out.SetInt(0)
			}
			return true
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			panic("can't happen: no uint types in BSON (!?)")
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		switch inv.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			out.SetUint(uint64(inv.Int()))
			return true
		case reflect.Float32, reflect.Float64:
			out.SetUint(uint64(inv.Float()))
			return true
		case reflect.Bool:
			if inv.Bool() {
				out.SetUint(1)
			} else {
				out.SetUint(0)
			}
			return true
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			panic("Can't happen. No uint types in BSON.")
		}
	case reflect.Float32, reflect.Float64:
		switch inv.Kind() {
		case reflect.Float32, reflect.Float64:
			out.SetFloat(inv.Float())
			return true
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			out.SetFloat(float64(inv.Int()))
			return true
		case reflect.Bool:
			if inv.Bool() {
				out.SetFloat(1)
			} else {
				out.SetFloat(0)
			}
			return true
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			panic("Can't happen. No uint types in BSON?")
		}
	case reflect.Bool:
		switch inv.Kind() {
		case reflect.Bool:
			out.SetBool(inv.Bool())
			return true
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			out.SetBool(inv.Int() != 0)
			return true
		case reflect.Float32, reflect.Float64:
			out.SetBool(inv.Float() != 0)
			return true
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			panic("Can't happen. No uint types in BSON?")
		}
	case reflect.Struct:
		if outt == typeURL && inv.Kind() == reflect.String {
			u, err := url.Parse(inv.String())
			if err != nil {
				panic(err)
			}
			out.Set(reflect.ValueOf(u).Elem())
			return true
		}
		if outt == typeBinary {
			if b, ok := in.([]byte); ok {
				out.Set(reflect.ValueOf(Binary{Data: b}))
				return true
			}
		}
	}

	return false
}

// --------------------------------------------------------------------------
// Parsers of basic types.

func (d *decoder) readRegEx() RegEx {
	re := RegEx{}
	re.Pattern = d.readCStr()
	re.Options = d.readCStr()
	return re
}

func (d *decoder) readBinary() Binary {
	l := d.readInt32()
	b := Binary{}
	b.Kind = d.readByte()
	if b.Kind == BinaryBinaryOld && l > 4 {
		// Weird obsolete format with redundant length.
		rl := d.readInt32()
		if rl != l-4 {
			corrupted()
		}
		l = rl
	}
	b.Data = d.readBytes(l)
	return b
}

func (d *decoder) readStr() string {
	l := d.readInt32()
	b := d.readBytes(l - 1)
	if d.readByte() != '\x00' {
		corrupted()
	}
	return string(b)
}

func (d *decoder) readCStr() string {
	start := d.i
	end := start
	l := len(d.in)
	for ; end != l; end++ {
		if d.in[end] == '\x00' {
			break
		}
	}
	d.i = end + 1
	if d.i > l {
		corrupted()
	}
	return string(d.in[start:end])
}

func (d *decoder) readBool() bool {
	b := d.readByte()
	if b == 0 {
		return false
	}
	if b == 1 {
		return true
	}
	panic(fmt.Sprintf("encoded boolean must be 1 or 0, found %d", b))
}

func (d *decoder) readFloat64() float64 {
	return math.Float64frombits(uint64(d.readInt64()))
}

func (d *decoder) readInt32() int32 {
	b := d.readBytes(4)
	return int32((uint32(b[0]) << 0) |
		(uint32(b[1]) << 8) |
		(uint32(b[2]) << 16) |
		(uint32(b[3]) << 24))
}

func getSize(offset int, b []byte) (int, error) {
	if offset+4 > len(b) {
		return 0, io.ErrUnexpectedEOF
	}
	return int((uint32(b[offset]) << 0) |
		(uint32(b[offset+1]) << 8) |
		(uint32(b[offset+2]) << 16) |
		(uint32(b[offset+3]) << 24)), nil
}

func (d *decoder) readInt64() int64 {
	b := d.readBytes(8)
	return int64((uint64(b[0]) << 0) |
		(uint64(b[1]) << 8) |
		(uint64(b[2]) << 16) |
		(uint64(b[3]) << 24) |
		(uint64(b[4]) << 32) |
		(uint64(b[5]) << 40) |
		(uint64(b[6]) << 48) |
		(uint64(b[7]) << 56))
}

func (d *decoder) readByte() byte {
	i := d.i
	d.i++
	if d.i > len(d.in) {
		corrupted()
	}
	return d.in[i]
}

func (d *decoder) readBytes(length int32) []byte {
	if length < 0 {
		corrupted()
	}
	start := d.i
	d.i += int(length)
	if d.i < start || d.i > len(d.in) {
		corrupted()
	}
	return d.in[start : start+int(length)]
}
