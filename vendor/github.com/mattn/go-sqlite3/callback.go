// Copyright (C) 2014 Yasuhiro Matsumoto <mattn.jp@gmail.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package sqlite3

// You can't export a Go function to C and have definitions in the C
// preamble in the same file, so we have to have callbackTrampoline in
// its own file. Because we need a separate file anyway, the support
// code for SQLite custom functions is in here.

/*
#ifndef USE_LIBSQLITE3
#include <sqlite3-binding.h>
#else
#include <sqlite3.h>
#endif
#include <stdlib.h>

void _sqlite3_result_text(sqlite3_context* ctx, const char* s);
void _sqlite3_result_blob(sqlite3_context* ctx, const void* b, int l);
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"unsafe"
)

//export callbackTrampoline
func callbackTrampoline(ctx *C.sqlite3_context, argc int, argv **C.sqlite3_value) {
	args := (*[(math.MaxInt32 - 1) / unsafe.Sizeof((*C.sqlite3_value)(nil))]*C.sqlite3_value)(unsafe.Pointer(argv))[:argc:argc]
	fi := lookupHandle(uintptr(C.sqlite3_user_data(ctx))).(*functionInfo)
	fi.Call(ctx, args)
}

//export stepTrampoline
func stepTrampoline(ctx *C.sqlite3_context, argc C.int, argv **C.sqlite3_value) {
	args := (*[(math.MaxInt32 - 1) / unsafe.Sizeof((*C.sqlite3_value)(nil))]*C.sqlite3_value)(unsafe.Pointer(argv))[:int(argc):int(argc)]
	ai := lookupHandle(uintptr(C.sqlite3_user_data(ctx))).(*aggInfo)
	ai.Step(ctx, args)
}

//export doneTrampoline
func doneTrampoline(ctx *C.sqlite3_context) {
	handle := uintptr(C.sqlite3_user_data(ctx))
	ai := lookupHandle(handle).(*aggInfo)
	ai.Done(ctx)
}

//export compareTrampoline
func compareTrampoline(handlePtr uintptr, la C.int, a *C.char, lb C.int, b *C.char) C.int {
	cmp := lookupHandle(handlePtr).(func(string, string) int)
	return C.int(cmp(C.GoStringN(a, la), C.GoStringN(b, lb)))
}

//export commitHookTrampoline
func commitHookTrampoline(handle uintptr) int {
	callback := lookupHandle(handle).(func() int)
	return callback()
}

//export rollbackHookTrampoline
func rollbackHookTrampoline(handle uintptr) {
	callback := lookupHandle(handle).(func())
	callback()
}

//export updateHookTrampoline
func updateHookTrampoline(handle uintptr, op int, db *C.char, table *C.char, rowid int64) {
	callback := lookupHandle(handle).(func(int, string, string, int64))
	callback(op, C.GoString(db), C.GoString(table), rowid)
}

// Use handles to avoid passing Go pointers to C.

type handleVal struct {
	db  *SQLiteConn
	val interface{}
}

var handleLock sync.Mutex
var handleVals = make(map[uintptr]handleVal)
var handleIndex uintptr = 100

func newHandle(db *SQLiteConn, v interface{}) uintptr {
	handleLock.Lock()
	defer handleLock.Unlock()
	i := handleIndex
	handleIndex++
	handleVals[i] = handleVal{db, v}
	return i
}

func lookupHandle(handle uintptr) interface{} {
	handleLock.Lock()
	defer handleLock.Unlock()
	r, ok := handleVals[handle]
	if !ok {
		if handle >= 100 && handle < handleIndex {
			panic("deleted handle")
		} else {
			panic("invalid handle")
		}
	}
	return r.val
}

func deleteHandles(db *SQLiteConn) {
	handleLock.Lock()
	defer handleLock.Unlock()
	for handle, val := range handleVals {
		if val.db == db {
			delete(handleVals, handle)
		}
	}
}

// This is only here so that tests can refer to it.
type callbackArgRaw C.sqlite3_value

type callbackArgConverter func(*C.sqlite3_value) (reflect.Value, error)

type callbackArgCast struct {
	f   callbackArgConverter
	typ reflect.Type
}

func (c callbackArgCast) Run(v *C.sqlite3_value) (reflect.Value, error) {
	val, err := c.f(v)
	if err != nil {
		return reflect.Value{}, err
	}
	if !val.Type().ConvertibleTo(c.typ) {
		return reflect.Value{}, fmt.Errorf("cannot convert %s to %s", val.Type(), c.typ)
	}
	return val.Convert(c.typ), nil
}

func callbackArgInt64(v *C.sqlite3_value) (reflect.Value, error) {
	if C.sqlite3_value_type(v) != C.SQLITE_INTEGER {
		return reflect.Value{}, fmt.Errorf("argument must be an INTEGER")
	}
	return reflect.ValueOf(int64(C.sqlite3_value_int64(v))), nil
}

func callbackArgBool(v *C.sqlite3_value) (reflect.Value, error) {
	if C.sqlite3_value_type(v) != C.SQLITE_INTEGER {
		return reflect.Value{}, fmt.Errorf("argument must be an INTEGER")
	}
	i := int64(C.sqlite3_value_int64(v))
	val := false
	if i != 0 {
		val = true
	}
	return reflect.ValueOf(val), nil
}

func callbackArgFloat64(v *C.sqlite3_value) (reflect.Value, error) {
	if C.sqlite3_value_type(v) != C.SQLITE_FLOAT {
		return reflect.Value{}, fmt.Errorf("argument must be a FLOAT")
	}
	return reflect.ValueOf(float64(C.sqlite3_value_double(v))), nil
}

func callbackArgBytes(v *C.sqlite3_value) (reflect.Value, error) {
	switch C.sqlite3_value_type(v) {
	case C.SQLITE_BLOB:
		l := C.sqlite3_value_bytes(v)
		p := C.sqlite3_value_blob(v)
		return reflect.ValueOf(C.GoBytes(p, l)), nil
	case C.SQLITE_TEXT:
		l := C.sqlite3_value_bytes(v)
		c := unsafe.Pointer(C.sqlite3_value_text(v))
		return reflect.ValueOf(C.GoBytes(c, l)), nil
	default:
		return reflect.Value{}, fmt.Errorf("argument must be BLOB or TEXT")
	}
}

func callbackArgString(v *C.sqlite3_value) (reflect.Value, error) {
	switch C.sqlite3_value_type(v) {
	case C.SQLITE_BLOB:
		l := C.sqlite3_value_bytes(v)
		p := (*C.char)(C.sqlite3_value_blob(v))
		return reflect.ValueOf(C.GoStringN(p, l)), nil
	case C.SQLITE_TEXT:
		c := (*C.char)(unsafe.Pointer(C.sqlite3_value_text(v)))
		return reflect.ValueOf(C.GoString(c)), nil
	default:
		return reflect.Value{}, fmt.Errorf("argument must be BLOB or TEXT")
	}
}

func callbackArgGeneric(v *C.sqlite3_value) (reflect.Value, error) {
	switch C.sqlite3_value_type(v) {
	case C.SQLITE_INTEGER:
		return callbackArgInt64(v)
	case C.SQLITE_FLOAT:
		return callbackArgFloat64(v)
	case C.SQLITE_TEXT:
		return callbackArgString(v)
	case C.SQLITE_BLOB:
		return callbackArgBytes(v)
	case C.SQLITE_NULL:
		// Interpret NULL as a nil byte slice.
		var ret []byte
		return reflect.ValueOf(ret), nil
	default:
		panic("unreachable")
	}
}

func callbackArg(typ reflect.Type) (callbackArgConverter, error) {
	switch typ.Kind() {
	case reflect.Interface:
		if typ.NumMethod() != 0 {
			return nil, errors.New("the only supported interface type is interface{}")
		}
		return callbackArgGeneric, nil
	case reflect.Slice:
		if typ.Elem().Kind() != reflect.Uint8 {
			return nil, errors.New("the only supported slice type is []byte")
		}
		return callbackArgBytes, nil
	case reflect.String:
		return callbackArgString, nil
	case reflect.Bool:
		return callbackArgBool, nil
	case reflect.Int64:
		return callbackArgInt64, nil
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Int, reflect.Uint:
		c := callbackArgCast{callbackArgInt64, typ}
		return c.Run, nil
	case reflect.Float64:
		return callbackArgFloat64, nil
	case reflect.Float32:
		c := callbackArgCast{callbackArgFloat64, typ}
		return c.Run, nil
	default:
		return nil, fmt.Errorf("don't know how to convert to %s", typ)
	}
}

func callbackConvertArgs(argv []*C.sqlite3_value, converters []callbackArgConverter, variadic callbackArgConverter) ([]reflect.Value, error) {
	var args []reflect.Value

	if len(argv) < len(converters) {
		return nil, fmt.Errorf("function requires at least %d arguments", len(converters))
	}

	for i, arg := range argv[:len(converters)] {
		v, err := converters[i](arg)
		if err != nil {
			return nil, err
		}
		args = append(args, v)
	}

	if variadic != nil {
		for _, arg := range argv[len(converters):] {
			v, err := variadic(arg)
			if err != nil {
				return nil, err
			}
			args = append(args, v)
		}
	}
	return args, nil
}

type callbackRetConverter func(*C.sqlite3_context, reflect.Value) error

func callbackRetInteger(ctx *C.sqlite3_context, v reflect.Value) error {
	switch v.Type().Kind() {
	case reflect.Int64:
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Int, reflect.Uint:
		v = v.Convert(reflect.TypeOf(int64(0)))
	case reflect.Bool:
		b := v.Interface().(bool)
		if b {
			v = reflect.ValueOf(int64(1))
		} else {
			v = reflect.ValueOf(int64(0))
		}
	default:
		return fmt.Errorf("cannot convert %s to INTEGER", v.Type())
	}

	C.sqlite3_result_int64(ctx, C.sqlite3_int64(v.Interface().(int64)))
	return nil
}

func callbackRetFloat(ctx *C.sqlite3_context, v reflect.Value) error {
	switch v.Type().Kind() {
	case reflect.Float64:
	case reflect.Float32:
		v = v.Convert(reflect.TypeOf(float64(0)))
	default:
		return fmt.Errorf("cannot convert %s to FLOAT", v.Type())
	}

	C.sqlite3_result_double(ctx, C.double(v.Interface().(float64)))
	return nil
}

func callbackRetBlob(ctx *C.sqlite3_context, v reflect.Value) error {
	if v.Type().Kind() != reflect.Slice || v.Type().Elem().Kind() != reflect.Uint8 {
		return fmt.Errorf("cannot convert %s to BLOB", v.Type())
	}
	i := v.Interface()
	if i == nil || len(i.([]byte)) == 0 {
		C.sqlite3_result_null(ctx)
	} else {
		bs := i.([]byte)
		C._sqlite3_result_blob(ctx, unsafe.Pointer(&bs[0]), C.int(len(bs)))
	}
	return nil
}

func callbackRetText(ctx *C.sqlite3_context, v reflect.Value) error {
	if v.Type().Kind() != reflect.String {
		return fmt.Errorf("cannot convert %s to TEXT", v.Type())
	}
	C._sqlite3_result_text(ctx, C.CString(v.Interface().(string)))
	return nil
}

func callbackRet(typ reflect.Type) (callbackRetConverter, error) {
	switch typ.Kind() {
	case reflect.Slice:
		if typ.Elem().Kind() != reflect.Uint8 {
			return nil, errors.New("the only supported slice type is []byte")
		}
		return callbackRetBlob, nil
	case reflect.String:
		return callbackRetText, nil
	case reflect.Bool, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Int, reflect.Uint:
		return callbackRetInteger, nil
	case reflect.Float32, reflect.Float64:
		return callbackRetFloat, nil
	default:
		return nil, fmt.Errorf("don't know how to convert to %s", typ)
	}
}

func callbackError(ctx *C.sqlite3_context, err error) {
	cstr := C.CString(err.Error())
	defer C.free(unsafe.Pointer(cstr))
	C.sqlite3_result_error(ctx, cstr, -1)
}

// Test support code. Tests are not allowed to import "C", so we can't
// declare any functions that use C.sqlite3_value.
func callbackSyntheticForTests(v reflect.Value, err error) callbackArgConverter {
	return func(*C.sqlite3_value) (reflect.Value, error) {
		return v, err
	}
}
