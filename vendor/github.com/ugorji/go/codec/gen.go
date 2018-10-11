// +build codecgen.exec

// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"go/format"
	"io"
	"io/ioutil"
	"math/rand"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/template"
	"time"
	"unicode"
	"unicode/utf8"
)

// ---------------------------------------------------
// codecgen supports the full cycle of reflection-based codec:
//    - RawExt
//    - Raw
//    - Builtins
//    - Extensions
//    - (Binary|Text|JSON)(Unm|M)arshal
//    - generic by-kind
//
// This means that, for dynamic things, we MUST use reflection to at least get the reflect.Type.
// In those areas, we try to only do reflection or interface-conversion when NECESSARY:
//    - Extensions, only if Extensions are configured.
//
// However, codecgen doesn't support the following:
//   - Canonical option. (codecgen IGNORES it currently)
//     This is just because it has not been implemented.
//
// During encode/decode, Selfer takes precedence.
// A type implementing Selfer will know how to encode/decode itself statically.
//
// The following field types are supported:
//     array: [n]T
//     slice: []T
//     map: map[K]V
//     primitive: [u]int[n], float(32|64), bool, string
//     struct
//
// ---------------------------------------------------
// Note that a Selfer cannot call (e|d).(En|De)code on itself,
// as this will cause a circular reference, as (En|De)code will call Selfer methods.
// Any type that implements Selfer must implement completely and not fallback to (En|De)code.
//
// In addition, code in this file manages the generation of fast-path implementations of
// encode/decode of slices/maps of primitive keys/values.
//
// Users MUST re-generate their implementations whenever the code shape changes.
// The generated code will panic if it was generated with a version older than the supporting library.
// ---------------------------------------------------
//
// codec framework is very feature rich.
// When encoding or decoding into an interface, it depends on the runtime type of the interface.
// The type of the interface may be a named type, an extension, etc.
// Consequently, we fallback to runtime codec for encoding/decoding interfaces.
// In addition, we fallback for any value which cannot be guaranteed at runtime.
// This allows us support ANY value, including any named types, specifically those which
// do not implement our interfaces (e.g. Selfer).
//
// This explains some slowness compared to other code generation codecs (e.g. msgp).
// This reduction in speed is only seen when your refers to interfaces,
// e.g. type T struct { A interface{}; B []interface{}; C map[string]interface{} }
//
// codecgen will panic if the file was generated with an old version of the library in use.
//
// Note:
//   It was a conscious decision to have gen.go always explicitly call EncodeNil or TryDecodeAsNil.
//   This way, there isn't a function call overhead just to see that we should not enter a block of code.
//
// Note:
//   codecgen-generated code depends on the variables defined by fast-path.generated.go.
//   consequently, you cannot run with tags "codecgen notfastpath".

// GenVersion is the current version of codecgen.
//
// NOTE: Increment this value each time codecgen changes fundamentally.
// Fundamental changes are:
//   - helper methods change (signature change, new ones added, some removed, etc)
//   - codecgen command line changes
//
// v1: Initial Version
// v2:
// v3: Changes for Kubernetes:
//     changes in signature of some unpublished helper methods and codecgen cmdline arguments.
// v4: Removed separator support from (en|de)cDriver, and refactored codec(gen)
// v5: changes to support faster json decoding. Let encoder/decoder maintain state of collections.
// v6: removed unsafe from gen, and now uses codecgen.exec tag
const genVersion = 8

const (
	genCodecPkg        = "codec1978"
	genTempVarPfx      = "yy"
	genTopLevelVarName = "x"

	// ignore canBeNil parameter, and always set to true.
	// This is because nil can appear anywhere, so we should always check.
	genAnythingCanBeNil = true

	// if genUseOneFunctionForDecStructMap, make a single codecDecodeSelferFromMap function;
	// else make codecDecodeSelferFromMap{LenPrefix,CheckBreak} so that conditionals
	// are not executed a lot.
	//
	// From testing, it didn't make much difference in runtime, so keep as true (one function only)
	genUseOneFunctionForDecStructMap = true
)

type genStructMapStyle uint8

const (
	genStructMapStyleConsolidated genStructMapStyle = iota
	genStructMapStyleLenPrefix
	genStructMapStyleCheckBreak
)

var (
	genAllTypesSamePkgErr  = errors.New("All types must be in the same package")
	genExpectArrayOrMapErr = errors.New("unexpected type. Expecting array/map/slice")
	genBase64enc           = base64.NewEncoding("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789__")
	genQNameRegex          = regexp.MustCompile(`[A-Za-z_.]+`)
)

// genRunner holds some state used during a Gen run.
type genRunner struct {
	w io.Writer      // output
	c uint64         // counter used for generating varsfx
	t []reflect.Type // list of types to run selfer on

	tc reflect.Type     // currently running selfer on this type
	te map[uintptr]bool // types for which the encoder has been created
	td map[uintptr]bool // types for which the decoder has been created
	cp string           // codec import path

	im  map[string]reflect.Type // imports to add
	imn map[string]string       // package names of imports to add
	imc uint64                  // counter for import numbers

	is map[reflect.Type]struct{} // types seen during import search
	bp string                    // base PkgPath, for which we are generating for

	cpfx string // codec package prefix

	tm map[reflect.Type]struct{} // types for which enc/dec must be generated
	ts []reflect.Type            // types for which enc/dec must be generated

	xs string // top level variable/constant suffix
	hn string // fn helper type name

	ti *TypeInfos
	// rr *rand.Rand // random generator for file-specific types

	nx bool // no extensions
}

// Gen will write a complete go file containing Selfer implementations for each
// type passed. All the types must be in the same package.
//
// Library users: DO NOT USE IT DIRECTLY. IT WILL CHANGE CONTINOUSLY WITHOUT NOTICE.
func Gen(w io.Writer, buildTags, pkgName, uid string, noExtensions bool,
	ti *TypeInfos, typ ...reflect.Type) {
	// All types passed to this method do not have a codec.Selfer method implemented directly.
	// codecgen already checks the AST and skips any types that define the codec.Selfer methods.
	// Consequently, there's no need to check and trim them if they implement codec.Selfer

	if len(typ) == 0 {
		return
	}
	x := genRunner{
		w:   w,
		t:   typ,
		te:  make(map[uintptr]bool),
		td:  make(map[uintptr]bool),
		im:  make(map[string]reflect.Type),
		imn: make(map[string]string),
		is:  make(map[reflect.Type]struct{}),
		tm:  make(map[reflect.Type]struct{}),
		ts:  []reflect.Type{},
		bp:  genImportPath(typ[0]),
		xs:  uid,
		ti:  ti,
		nx:  noExtensions,
	}
	if x.ti == nil {
		x.ti = defTypeInfos
	}
	if x.xs == "" {
		rr := rand.New(rand.NewSource(time.Now().UnixNano()))
		x.xs = strconv.FormatInt(rr.Int63n(9999), 10)
	}

	// gather imports first:
	x.cp = genImportPath(reflect.TypeOf(x))
	x.imn[x.cp] = genCodecPkg
	for _, t := range typ {
		// fmt.Printf("###########: PkgPath: '%v', Name: '%s'\n", genImportPath(t), t.Name())
		if genImportPath(t) != x.bp {
			panic(genAllTypesSamePkgErr)
		}
		x.genRefPkgs(t)
	}
	if buildTags != "" {
		x.line("// +build " + buildTags)
		x.line("")
	}
	x.line(`

// ************************************************************
// DO NOT EDIT.
// THIS FILE IS AUTO-GENERATED BY codecgen.
// ************************************************************

`)
	x.line("package " + pkgName)
	x.line("")
	x.line("import (")
	if x.cp != x.bp {
		x.cpfx = genCodecPkg + "."
		x.linef("%s \"%s\"", genCodecPkg, x.cp)
	}
	// use a sorted set of im keys, so that we can get consistent output
	imKeys := make([]string, 0, len(x.im))
	for k, _ := range x.im {
		imKeys = append(imKeys, k)
	}
	sort.Strings(imKeys)
	for _, k := range imKeys { // for k, _ := range x.im {
		x.linef("%s \"%s\"", x.imn[k], k)
	}
	// add required packages
	for _, k := range [...]string{"reflect", "runtime", "fmt", "errors"} {
		if _, ok := x.im[k]; !ok {
			x.line("\"" + k + "\"")
		}
	}
	x.line(")")
	x.line("")

	x.line("const (")
	x.linef("// ----- content types ----")
	x.linef("codecSelferC_UTF8%s = %v", x.xs, int64(c_UTF8))
	x.linef("codecSelferC_RAW%s = %v", x.xs, int64(c_RAW))
	x.linef("// ----- value types used ----")
	x.linef("codecSelferValueTypeArray%s = %v", x.xs, int64(valueTypeArray))
	x.linef("codecSelferValueTypeMap%s = %v", x.xs, int64(valueTypeMap))
	x.linef("// ----- containerStateValues ----")
	x.linef("codecSelfer_containerMapKey%s = %v", x.xs, int64(containerMapKey))
	x.linef("codecSelfer_containerMapValue%s = %v", x.xs, int64(containerMapValue))
	x.linef("codecSelfer_containerMapEnd%s = %v", x.xs, int64(containerMapEnd))
	x.linef("codecSelfer_containerArrayElem%s = %v", x.xs, int64(containerArrayElem))
	x.linef("codecSelfer_containerArrayEnd%s = %v", x.xs, int64(containerArrayEnd))
	x.line(")")
	x.line("var (")
	x.line("codecSelferBitsize" + x.xs + " = uint8(reflect.TypeOf(uint(0)).Bits())")
	x.line("codecSelferOnlyMapOrArrayEncodeToStructErr" + x.xs + " = errors.New(`only encoded map or array can be decoded into a struct`)")
	x.line(")")
	x.line("")

	x.hn = "codecSelfer" + x.xs
	x.line("type " + x.hn + " struct{}")
	x.line("")

	x.varsfxreset()
	x.line("func init() {")
	x.linef("if %sGenVersion != %v {", x.cpfx, genVersion)
	x.line("_, file, _, _ := runtime.Caller(0)")
	x.line(`err := fmt.Errorf("codecgen version mismatch: current: %v, need %v. Re-generate file: %v", `)
	x.linef(`%v, %sGenVersion, file)`, genVersion, x.cpfx)
	x.line("panic(err)")
	x.linef("}")
	x.line("if false { // reference the types, but skip this branch at build/run time")
	var n int
	// for k, t := range x.im {
	for _, k := range imKeys {
		t := x.im[k]
		x.linef("var v%v %s.%s", n, x.imn[k], t.Name())
		n++
	}
	if n > 0 {
		x.out("_")
		for i := 1; i < n; i++ {
			x.out(", _")
		}
		x.out(" = v0")
		for i := 1; i < n; i++ {
			x.outf(", v%v", i)
		}
	}
	x.line("} ") // close if false
	x.line("}")  // close init
	x.line("")

	// generate rest of type info
	for _, t := range typ {
		x.tc = t
		x.selfer(true)
		x.selfer(false)
	}

	for _, t := range x.ts {
		rtid := rt2id(t)
		// generate enc functions for all these slice/map types.
		x.varsfxreset()
		x.linef("func (x %s) enc%s(v %s%s, e *%sEncoder) {", x.hn, x.genMethodNameT(t), x.arr2str(t, "*"), x.genTypeName(t), x.cpfx)
		x.genRequiredMethodVars(true)
		switch t.Kind() {
		case reflect.Array, reflect.Slice, reflect.Chan:
			x.encListFallback("v", t)
		case reflect.Map:
			x.encMapFallback("v", t)
		default:
			panic(genExpectArrayOrMapErr)
		}
		x.line("}")
		x.line("")

		// generate dec functions for all these slice/map types.
		x.varsfxreset()
		x.linef("func (x %s) dec%s(v *%s, d *%sDecoder) {", x.hn, x.genMethodNameT(t), x.genTypeName(t), x.cpfx)
		x.genRequiredMethodVars(false)
		switch t.Kind() {
		case reflect.Array, reflect.Slice, reflect.Chan:
			x.decListFallback("v", rtid, t)
		case reflect.Map:
			x.decMapFallback("v", rtid, t)
		default:
			panic(genExpectArrayOrMapErr)
		}
		x.line("}")
		x.line("")
	}

	x.line("")
}

func (x *genRunner) checkForSelfer(t reflect.Type, varname string) bool {
	// return varname != genTopLevelVarName && t != x.tc
	// the only time we checkForSelfer is if we are not at the TOP of the generated code.
	return varname != genTopLevelVarName
}

func (x *genRunner) arr2str(t reflect.Type, s string) string {
	if t.Kind() == reflect.Array {
		return s
	}
	return ""
}

func (x *genRunner) genRequiredMethodVars(encode bool) {
	x.line("var h " + x.hn)
	if encode {
		x.line("z, r := " + x.cpfx + "GenHelperEncoder(e)")
	} else {
		x.line("z, r := " + x.cpfx + "GenHelperDecoder(d)")
	}
	x.line("_, _, _ = h, z, r")
}

func (x *genRunner) genRefPkgs(t reflect.Type) {
	if _, ok := x.is[t]; ok {
		return
	}
	// fmt.Printf(">>>>>>: PkgPath: '%v', Name: '%s'\n", genImportPath(t), t.Name())
	x.is[t] = struct{}{}
	tpkg, tname := genImportPath(t), t.Name()
	if tpkg != "" && tpkg != x.bp && tpkg != x.cp && tname != "" && tname[0] >= 'A' && tname[0] <= 'Z' {
		if _, ok := x.im[tpkg]; !ok {
			x.im[tpkg] = t
			if idx := strings.LastIndex(tpkg, "/"); idx < 0 {
				x.imn[tpkg] = tpkg
			} else {
				x.imc++
				x.imn[tpkg] = "pkg" + strconv.FormatUint(x.imc, 10) + "_" + genGoIdentifier(tpkg[idx+1:], false)
			}
		}
	}
	switch t.Kind() {
	case reflect.Array, reflect.Slice, reflect.Ptr, reflect.Chan:
		x.genRefPkgs(t.Elem())
	case reflect.Map:
		x.genRefPkgs(t.Elem())
		x.genRefPkgs(t.Key())
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if fname := t.Field(i).Name; fname != "" && fname[0] >= 'A' && fname[0] <= 'Z' {
				x.genRefPkgs(t.Field(i).Type)
			}
		}
	}
}

func (x *genRunner) line(s string) {
	x.out(s)
	if len(s) == 0 || s[len(s)-1] != '\n' {
		x.out("\n")
	}
}

func (x *genRunner) varsfx() string {
	x.c++
	return strconv.FormatUint(x.c, 10)
}

func (x *genRunner) varsfxreset() {
	x.c = 0
}

func (x *genRunner) out(s string) {
	if _, err := io.WriteString(x.w, s); err != nil {
		panic(err)
	}
}

func (x *genRunner) linef(s string, params ...interface{}) {
	x.line(fmt.Sprintf(s, params...))
}

func (x *genRunner) outf(s string, params ...interface{}) {
	x.out(fmt.Sprintf(s, params...))
}

func (x *genRunner) genTypeName(t reflect.Type) (n string) {
	// defer func() { fmt.Printf(">>>> ####: genTypeName: t: %v, name: '%s'\n", t, n) }()

	// if the type has a PkgPath, which doesn't match the current package,
	// then include it.
	// We cannot depend on t.String() because it includes current package,
	// or t.PkgPath because it includes full import path,
	//
	var ptrPfx string
	for t.Kind() == reflect.Ptr {
		ptrPfx += "*"
		t = t.Elem()
	}
	if tn := t.Name(); tn != "" {
		return ptrPfx + x.genTypeNamePrim(t)
	}
	switch t.Kind() {
	case reflect.Map:
		return ptrPfx + "map[" + x.genTypeName(t.Key()) + "]" + x.genTypeName(t.Elem())
	case reflect.Slice:
		return ptrPfx + "[]" + x.genTypeName(t.Elem())
	case reflect.Array:
		return ptrPfx + "[" + strconv.FormatInt(int64(t.Len()), 10) + "]" + x.genTypeName(t.Elem())
	case reflect.Chan:
		return ptrPfx + t.ChanDir().String() + " " + x.genTypeName(t.Elem())
	default:
		if t == intfTyp {
			return ptrPfx + "interface{}"
		} else {
			return ptrPfx + x.genTypeNamePrim(t)
		}
	}
}

func (x *genRunner) genTypeNamePrim(t reflect.Type) (n string) {
	if t.Name() == "" {
		return t.String()
	} else if genImportPath(t) == "" || genImportPath(t) == genImportPath(x.tc) {
		return t.Name()
	} else {
		return x.imn[genImportPath(t)] + "." + t.Name()
		// return t.String() // best way to get the package name inclusive
	}
}

func (x *genRunner) genZeroValueR(t reflect.Type) string {
	// if t is a named type, w
	switch t.Kind() {
	case reflect.Ptr, reflect.Interface, reflect.Chan, reflect.Func,
		reflect.Slice, reflect.Map, reflect.Invalid:
		return "nil"
	case reflect.Bool:
		return "false"
	case reflect.String:
		return `""`
	case reflect.Struct, reflect.Array:
		return x.genTypeName(t) + "{}"
	default: // all numbers
		return "0"
	}
}

func (x *genRunner) genMethodNameT(t reflect.Type) (s string) {
	return genMethodNameT(t, x.tc)
}

func (x *genRunner) selfer(encode bool) {
	t := x.tc
	t0 := t
	// always make decode use a pointer receiver,
	// and structs always use a ptr receiver (encode|decode)
	isptr := !encode || t.Kind() == reflect.Struct
	x.varsfxreset()
	fnSigPfx := "func (x "
	if isptr {
		fnSigPfx += "*"
	}
	fnSigPfx += x.genTypeName(t)

	x.out(fnSigPfx)
	if isptr {
		t = reflect.PtrTo(t)
	}
	if encode {
		x.line(") CodecEncodeSelf(e *" + x.cpfx + "Encoder) {")
		x.genRequiredMethodVars(true)
		// x.enc(genTopLevelVarName, t)
		x.encVar(genTopLevelVarName, t)
	} else {
		x.line(") CodecDecodeSelf(d *" + x.cpfx + "Decoder) {")
		x.genRequiredMethodVars(false)
		// do not use decVar, as there is no need to check TryDecodeAsNil
		// or way to elegantly handle that, and also setting it to a
		// non-nil value doesn't affect the pointer passed.
		// x.decVar(genTopLevelVarName, t, false)
		x.dec(genTopLevelVarName, t0)
	}
	x.line("}")
	x.line("")

	if encode || t0.Kind() != reflect.Struct {
		return
	}

	// write is containerMap
	if genUseOneFunctionForDecStructMap {
		x.out(fnSigPfx)
		x.line(") codecDecodeSelfFromMap(l int, d *" + x.cpfx + "Decoder) {")
		x.genRequiredMethodVars(false)
		x.decStructMap(genTopLevelVarName, "l", rt2id(t0), t0, genStructMapStyleConsolidated)
		x.line("}")
		x.line("")
	} else {
		x.out(fnSigPfx)
		x.line(") codecDecodeSelfFromMapLenPrefix(l int, d *" + x.cpfx + "Decoder) {")
		x.genRequiredMethodVars(false)
		x.decStructMap(genTopLevelVarName, "l", rt2id(t0), t0, genStructMapStyleLenPrefix)
		x.line("}")
		x.line("")

		x.out(fnSigPfx)
		x.line(") codecDecodeSelfFromMapCheckBreak(l int, d *" + x.cpfx + "Decoder) {")
		x.genRequiredMethodVars(false)
		x.decStructMap(genTopLevelVarName, "l", rt2id(t0), t0, genStructMapStyleCheckBreak)
		x.line("}")
		x.line("")
	}

	// write containerArray
	x.out(fnSigPfx)
	x.line(") codecDecodeSelfFromArray(l int, d *" + x.cpfx + "Decoder) {")
	x.genRequiredMethodVars(false)
	x.decStructArray(genTopLevelVarName, "l", "return", rt2id(t0), t0)
	x.line("}")
	x.line("")

}

// used for chan, array, slice, map
func (x *genRunner) xtraSM(varname string, encode bool, t reflect.Type) {
	if encode {
		x.linef("h.enc%s((%s%s)(%s), e)", x.genMethodNameT(t), x.arr2str(t, "*"), x.genTypeName(t), varname)
	} else {
		x.linef("h.dec%s((*%s)(%s), d)", x.genMethodNameT(t), x.genTypeName(t), varname)
	}
	x.registerXtraT(t)
}

func (x *genRunner) registerXtraT(t reflect.Type) {
	// recursively register the types
	if _, ok := x.tm[t]; ok {
		return
	}
	var tkey reflect.Type
	switch t.Kind() {
	case reflect.Chan, reflect.Slice, reflect.Array:
	case reflect.Map:
		tkey = t.Key()
	default:
		return
	}
	x.tm[t] = struct{}{}
	x.ts = append(x.ts, t)
	// check if this refers to any xtra types eg. a slice of array: add the array
	x.registerXtraT(t.Elem())
	if tkey != nil {
		x.registerXtraT(tkey)
	}
}

// encVar will encode a variable.
// The parameter, t, is the reflect.Type of the variable itself
func (x *genRunner) encVar(varname string, t reflect.Type) {
	// fmt.Printf(">>>>>> varname: %s, t: %v\n", varname, t)
	var checkNil bool
	switch t.Kind() {
	case reflect.Ptr, reflect.Interface, reflect.Slice, reflect.Map, reflect.Chan:
		checkNil = true
	}
	if checkNil {
		x.linef("if %s == nil { r.EncodeNil() } else { ", varname)
	}
	switch t.Kind() {
	case reflect.Ptr:
		switch t.Elem().Kind() {
		case reflect.Struct, reflect.Array:
			x.enc(varname, genNonPtr(t))
		default:
			i := x.varsfx()
			x.line(genTempVarPfx + i + " := *" + varname)
			x.enc(genTempVarPfx+i, genNonPtr(t))
		}
	case reflect.Struct, reflect.Array:
		i := x.varsfx()
		x.line(genTempVarPfx + i + " := &" + varname)
		x.enc(genTempVarPfx+i, t)
	default:
		x.enc(varname, t)
	}

	if checkNil {
		x.line("}")
	}

}

// enc will encode a variable (varname) of type t,
// except t is of kind reflect.Struct or reflect.Array, wherein varname is of type ptrTo(T) (to prevent copying)
func (x *genRunner) enc(varname string, t reflect.Type) {
	rtid := rt2id(t)
	// We call CodecEncodeSelf if one of the following are honored:
	//   - the type already implements Selfer, call that
	//   - the type has a Selfer implementation just created, use that
	//   - the type is in the list of the ones we will generate for, but it is not currently being generated

	mi := x.varsfx()
	tptr := reflect.PtrTo(t)
	tk := t.Kind()
	if x.checkForSelfer(t, varname) {
		if tk == reflect.Array || tk == reflect.Struct { // varname is of type *T
			if tptr.Implements(selferTyp) || t.Implements(selferTyp) {
				x.line(varname + ".CodecEncodeSelf(e)")
				return
			}
		} else { // varname is of type T
			if t.Implements(selferTyp) {
				x.line(varname + ".CodecEncodeSelf(e)")
				return
			} else if tptr.Implements(selferTyp) {
				x.linef("%ssf%s := &%s", genTempVarPfx, mi, varname)
				x.linef("%ssf%s.CodecEncodeSelf(e)", genTempVarPfx, mi)
				return
			}
		}

		if _, ok := x.te[rtid]; ok {
			x.line(varname + ".CodecEncodeSelf(e)")
			return
		}
	}

	inlist := false
	for _, t0 := range x.t {
		if t == t0 {
			inlist = true
			if x.checkForSelfer(t, varname) {
				x.line(varname + ".CodecEncodeSelf(e)")
				return
			}
			break
		}
	}

	var rtidAdded bool
	if t == x.tc {
		x.te[rtid] = true
		rtidAdded = true
	}

	// check if
	//   - type is RawExt, Raw
	//   - the type implements (Text|JSON|Binary)(Unm|M)arshal
	x.linef("%sm%s := z.EncBinary()", genTempVarPfx, mi)
	x.linef("_ = %sm%s", genTempVarPfx, mi)
	x.line("if false {")           //start if block
	defer func() { x.line("}") }() //end if block

	if t == rawTyp {
		x.linef("} else { z.EncRaw(%v)", varname)
		return
	}
	if t == rawExtTyp {
		x.linef("} else { r.EncodeRawExt(%v, e)", varname)
		return
	}
	// HACK: Support for Builtins.
	//       Currently, only Binc supports builtins, and the only builtin type is time.Time.
	//       Have a method that returns the rtid for time.Time if Handle is Binc.
	if t == timeTyp {
		vrtid := genTempVarPfx + "m" + x.varsfx()
		x.linef("} else if %s := z.TimeRtidIfBinc(); %s != 0 { ", vrtid, vrtid)
		x.linef("r.EncodeBuiltin(%s, %s)", vrtid, varname)
	}
	// only check for extensions if the type is named, and has a packagePath.
	if !x.nx && genImportPath(t) != "" && t.Name() != "" {
		// first check if extensions are configued, before doing the interface conversion
		x.linef("} else if z.HasExtensions() && z.EncExt(%s) {", varname)
	}
	if tk == reflect.Array || tk == reflect.Struct { // varname is of type *T
		if t.Implements(binaryMarshalerTyp) || tptr.Implements(binaryMarshalerTyp) {
			x.linef("} else if %sm%s { z.EncBinaryMarshal(%v) ", genTempVarPfx, mi, varname)
		}
		if t.Implements(jsonMarshalerTyp) || tptr.Implements(jsonMarshalerTyp) {
			x.linef("} else if !%sm%s && z.IsJSONHandle() { z.EncJSONMarshal(%v) ", genTempVarPfx, mi, varname)
		} else if t.Implements(textMarshalerTyp) || tptr.Implements(textMarshalerTyp) {
			x.linef("} else if !%sm%s { z.EncTextMarshal(%v) ", genTempVarPfx, mi, varname)
		}
	} else { // varname is of type T
		if t.Implements(binaryMarshalerTyp) {
			x.linef("} else if %sm%s { z.EncBinaryMarshal(%v) ", genTempVarPfx, mi, varname)
		} else if tptr.Implements(binaryMarshalerTyp) {
			x.linef("} else if %sm%s { z.EncBinaryMarshal(&%v) ", genTempVarPfx, mi, varname)
		}
		if t.Implements(jsonMarshalerTyp) {
			x.linef("} else if !%sm%s && z.IsJSONHandle() { z.EncJSONMarshal(%v) ", genTempVarPfx, mi, varname)
		} else if tptr.Implements(jsonMarshalerTyp) {
			x.linef("} else if !%sm%s && z.IsJSONHandle() { z.EncJSONMarshal(&%v) ", genTempVarPfx, mi, varname)
		} else if t.Implements(textMarshalerTyp) {
			x.linef("} else if !%sm%s { z.EncTextMarshal(%v) ", genTempVarPfx, mi, varname)
		} else if tptr.Implements(textMarshalerTyp) {
			x.linef("} else if !%sm%s { z.EncTextMarshal(&%v) ", genTempVarPfx, mi, varname)
		}
	}
	x.line("} else {")

	switch t.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		x.line("r.EncodeInt(int64(" + varname + "))")
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		x.line("r.EncodeUint(uint64(" + varname + "))")
	case reflect.Float32:
		x.line("r.EncodeFloat32(float32(" + varname + "))")
	case reflect.Float64:
		x.line("r.EncodeFloat64(float64(" + varname + "))")
	case reflect.Bool:
		x.line("r.EncodeBool(bool(" + varname + "))")
	case reflect.String:
		x.line("r.EncodeString(codecSelferC_UTF8" + x.xs + ", string(" + varname + "))")
	case reflect.Chan:
		x.xtraSM(varname, true, t)
		// x.encListFallback(varname, rtid, t)
	case reflect.Array:
		x.xtraSM(varname, true, t)
	case reflect.Slice:
		// if nil, call dedicated function
		// if a []uint8, call dedicated function
		// if a known fastpath slice, call dedicated function
		// else write encode function in-line.
		// - if elements are primitives or Selfers, call dedicated function on each member.
		// - else call Encoder.encode(XXX) on it.
		if rtid == uint8SliceTypId {
			x.line("r.EncodeStringBytes(codecSelferC_RAW" + x.xs + ", []byte(" + varname + "))")
		} else if fastpathAV.index(rtid) != -1 {
			g := x.newGenV(t)
			x.line("z.F." + g.MethodNamePfx("Enc", false) + "V(" + varname + ", e)")
		} else {
			x.xtraSM(varname, true, t)
			// x.encListFallback(varname, rtid, t)
		}
	case reflect.Map:
		// if nil, call dedicated function
		// if a known fastpath map, call dedicated function
		// else write encode function in-line.
		// - if elements are primitives or Selfers, call dedicated function on each member.
		// - else call Encoder.encode(XXX) on it.
		// x.line("if " + varname + " == nil { \nr.EncodeNil()\n } else { ")
		if fastpathAV.index(rtid) != -1 {
			g := x.newGenV(t)
			x.line("z.F." + g.MethodNamePfx("Enc", false) + "V(" + varname + ", e)")
		} else {
			x.xtraSM(varname, true, t)
			// x.encMapFallback(varname, rtid, t)
		}
	case reflect.Struct:
		if !inlist {
			delete(x.te, rtid)
			x.line("z.EncFallback(" + varname + ")")
			break
		}
		x.encStruct(varname, rtid, t)
	default:
		if rtidAdded {
			delete(x.te, rtid)
		}
		x.line("z.EncFallback(" + varname + ")")
	}
}

func (x *genRunner) encZero(t reflect.Type) {
	switch t.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		x.line("r.EncodeInt(0)")
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		x.line("r.EncodeUint(0)")
	case reflect.Float32:
		x.line("r.EncodeFloat32(0)")
	case reflect.Float64:
		x.line("r.EncodeFloat64(0)")
	case reflect.Bool:
		x.line("r.EncodeBool(false)")
	case reflect.String:
		x.line("r.EncodeString(codecSelferC_UTF8" + x.xs + `, "")`)
	default:
		x.line("r.EncodeNil()")
	}
}

func (x *genRunner) encStruct(varname string, rtid uintptr, t reflect.Type) {
	// Use knowledge from structfieldinfo (mbs, encodable fields. Ignore omitempty. )
	// replicate code in kStruct i.e. for each field, deref type to non-pointer, and call x.enc on it

	// if t === type currently running selfer on, do for all
	ti := x.ti.get(rtid, t)
	i := x.varsfx()
	sepVarname := genTempVarPfx + "sep" + i
	numfieldsvar := genTempVarPfx + "q" + i
	ti2arrayvar := genTempVarPfx + "r" + i
	struct2arrvar := genTempVarPfx + "2arr" + i

	x.line(sepVarname + " := !z.EncBinary()")
	x.linef("%s := z.EncBasicHandle().StructToArray", struct2arrvar)
	tisfi := ti.sfip // always use sequence from file. decStruct expects same thing.
	// due to omitEmpty, we need to calculate the
	// number of non-empty things we write out first.
	// This is required as we need to pre-determine the size of the container,
	// to support length-prefixing.
	if ti.anyOmitEmpty {
		x.linef("var %s [%v]bool", numfieldsvar, len(tisfi))
		x.linef("_ = %s", numfieldsvar)
	}
	x.linef("_, _ = %s, %s", sepVarname, struct2arrvar)
	x.linef("const %s bool = %v", ti2arrayvar, ti.toArray)
	var nn int
	if ti.anyOmitEmpty {
		for j, si := range tisfi {
			if !si.omitEmpty {
				nn++
				continue
			}
			var t2 reflect.StructField
			var omitline string
			{
				t2typ := t
				varname3 := varname
				for ij, ix := range si.is {
					if uint8(ij) == si.nis {
						break
					}
					for t2typ.Kind() == reflect.Ptr {
						t2typ = t2typ.Elem()
					}
					t2 = t2typ.Field(int(ix))
					t2typ = t2.Type
					varname3 = varname3 + "." + t2.Name
					if t2typ.Kind() == reflect.Ptr {
						omitline += varname3 + " != nil && "
					}
				}
			}
			// never check omitEmpty on a struct type, as it may contain uncomparable map/slice/etc.
			// also, for maps/slices/arrays, check if len ! 0 (not if == zero value)
			switch t2.Type.Kind() {
			case reflect.Struct:
				omitline += " true"
			case reflect.Map, reflect.Slice, reflect.Array, reflect.Chan:
				omitline += "len(" + varname + "." + t2.Name + ") != 0"
			default:
				omitline += varname + "." + t2.Name + " != " + x.genZeroValueR(t2.Type)
			}
			x.linef("%s[%v] = %s", numfieldsvar, j, omitline)
		}
	}
	// x.linef("var %snn%s int", genTempVarPfx, i)
	x.linef("if %s || %s {", ti2arrayvar, struct2arrvar) // if ti.toArray {
	x.linef("r.WriteArrayStart(%d)", len(tisfi))
	x.linef("} else {") // if not ti.toArray
	if ti.anyOmitEmpty {
		x.linef("var %snn%s = %v", genTempVarPfx, i, nn)
		x.linef("for _, b := range %s { if b { %snn%s++ } }", numfieldsvar, genTempVarPfx, i)
		x.linef("r.WriteMapStart(%snn%s)", genTempVarPfx, i)
		x.linef("%snn%s = %v", genTempVarPfx, i, 0)
	} else {
		x.linef("r.WriteMapStart(%d)", len(tisfi))
	}
	x.line("}") // close if not StructToArray

	for j, si := range tisfi {
		i := x.varsfx()
		isNilVarName := genTempVarPfx + "n" + i
		var labelUsed bool
		var t2 reflect.StructField
		{
			t2typ := t
			varname3 := varname
			for ij, ix := range si.is {
				if uint8(ij) == si.nis {
					break
				}
				// fmt.Printf("%%%% %v, ix: %v\n", t2typ, ix)
				for t2typ.Kind() == reflect.Ptr {
					t2typ = t2typ.Elem()
				}
				t2 = t2typ.Field(int(ix))
				t2typ = t2.Type
				varname3 = varname3 + "." + t2.Name
				if t2typ.Kind() == reflect.Ptr {
					if !labelUsed {
						x.line("var " + isNilVarName + " bool")
					}
					x.line("if " + varname3 + " == nil { " + isNilVarName + " = true ")
					x.line("goto LABEL" + i)
					x.line("}")
					labelUsed = true
					// "varname3 = new(" + x.genTypeName(t3.Elem()) + ") }")
				}
			}
			// t2 = t.FieldByIndex(si.is)
		}
		if labelUsed {
			x.line("LABEL" + i + ":")
		}
		// if the type of the field is a Selfer, or one of the ones

		x.linef("if %s || %s {", ti2arrayvar, struct2arrvar) // if ti.toArray
		if labelUsed {
			x.linef("if %s { r.WriteArrayElem(); r.EncodeNil() } else { ", isNilVarName)
			// x.linef("if %s { z.EncSendContainerState(codecSelfer_containerArrayElem%s); r.EncodeNil() } else { ", isNilVarName, x.xs)
		}
		x.line("r.WriteArrayElem()") // x.linef("z.EncSendContainerState(codecSelfer_containerArrayElem%s)", x.xs)
		if si.omitEmpty {
			x.linef("if %s[%v] {", numfieldsvar, j)
		}
		x.encVar(varname+"."+t2.Name, t2.Type)
		if si.omitEmpty {
			x.linef("} else {")
			x.encZero(t2.Type)
			x.linef("}")
		}
		if labelUsed {
			x.line("}")
		}

		x.linef("} else {") // if not ti.toArray

		if si.omitEmpty {
			x.linef("if %s[%v] {", numfieldsvar, j)
		}
		x.line("r.WriteMapElemKey()") // x.linef("z.EncSendContainerState(codecSelfer_containerMapKey%s)", x.xs)
		x.line("r.EncodeString(codecSelferC_UTF8" + x.xs + ", string(\"" + si.encName + "\"))")
		x.line("r.WriteMapElemValue()") // x.linef("z.EncSendContainerState(codecSelfer_containerMapValue%s)", x.xs)
		if labelUsed {
			x.line("if " + isNilVarName + " { r.EncodeNil() } else { ")
			x.encVar(varname+"."+t2.Name, t2.Type)
			x.line("}")
		} else {
			x.encVar(varname+"."+t2.Name, t2.Type)
		}
		if si.omitEmpty {
			x.line("}")
		}
		x.linef("} ") // end if/else ti.toArray
	}
	x.linef("if %s || %s {", ti2arrayvar, struct2arrvar) // if ti.toArray {
	x.line("r.WriteArrayEnd()")                          // x.linef("z.EncSendContainerState(codecSelfer_containerArrayEnd%s)", x.xs)
	x.line("} else {")
	x.line("r.WriteMapEnd()") // x.linef("z.EncSendContainerState(codecSelfer_containerMapEnd%s)", x.xs)
	x.line("}")

}

func (x *genRunner) encListFallback(varname string, t reflect.Type) {
	if t.AssignableTo(uint8SliceTyp) {
		x.linef("r.EncodeStringBytes(codecSelferC_RAW%s, []byte(%s))", x.xs, varname)
		return
	}
	if t.Kind() == reflect.Array && t.Elem().Kind() == reflect.Uint8 {
		x.linef("r.EncodeStringBytes(codecSelferC_RAW%s, ([%v]byte(%s))[:])", x.xs, t.Len(), varname)
		return
	}
	i := x.varsfx()
	g := genTempVarPfx
	x.line("r.WriteArrayStart(len(" + varname + "))")
	if t.Kind() == reflect.Chan {
		x.linef("for %si%s, %si2%s := 0, len(%s); %si%s < %si2%s; %si%s++ {", g, i, g, i, varname, g, i, g, i, g, i)
		x.line("r.WriteArrayElem()") // x.linef("z.EncSendContainerState(codecSelfer_containerArrayElem%s)", x.xs)
		x.linef("%sv%s := <-%s", g, i, varname)
	} else {
		// x.linef("for %si%s, %sv%s := range %s {", genTempVarPfx, i, genTempVarPfx, i, varname)
		x.linef("for _, %sv%s := range %s {", genTempVarPfx, i, varname)
		x.line("r.WriteArrayElem()") // x.linef("z.EncSendContainerState(codecSelfer_containerArrayElem%s)", x.xs)
	}
	x.encVar(genTempVarPfx+"v"+i, t.Elem())
	x.line("}")
	x.line("r.WriteArrayEnd()") // x.linef("z.EncSendContainerState(codecSelfer_containerArrayEnd%s)", x.xs)
}

func (x *genRunner) encMapFallback(varname string, t reflect.Type) {
	// TODO: expand this to handle canonical.
	i := x.varsfx()
	x.line("r.WriteMapStart(len(" + varname + "))")
	x.linef("for %sk%s, %sv%s := range %s {", genTempVarPfx, i, genTempVarPfx, i, varname)
	// x.line("for " + genTempVarPfx + "k" + i + ", " + genTempVarPfx + "v" + i + " := range " + varname + " {")
	x.line("r.WriteMapElemKey()") // f("z.EncSendContainerState(codecSelfer_containerMapKey%s)", x.xs)
	x.encVar(genTempVarPfx+"k"+i, t.Key())
	x.line("r.WriteMapElemValue()") // f("z.EncSendContainerState(codecSelfer_containerMapValue%s)", x.xs)
	x.encVar(genTempVarPfx+"v"+i, t.Elem())
	x.line("}")
	x.line("r.WriteMapEnd()") // f("z.EncSendContainerState(codecSelfer_containerMapEnd%s)", x.xs)
}

func (x *genRunner) decVar(varname, decodedNilVarname string, t reflect.Type, canBeNil bool) {
	// We only encode as nil if a nillable value.
	// This removes some of the wasted checks for TryDecodeAsNil.
	// We need to think about this more, to see what happens if omitempty, etc
	// cause a nil value to be stored when something is expected.
	// This could happen when decoding from a struct encoded as an array.
	// For that, decVar should be called with canNil=true, to force true as its value.
	i := x.varsfx()
	if !canBeNil {
		canBeNil = genAnythingCanBeNil || !genIsImmutable(t)
	}
	if canBeNil {
		x.line("if r.TryDecodeAsNil() {")
		if decodedNilVarname != "" {
			x.line(decodedNilVarname + " = true")
		} else if t.Kind() == reflect.Ptr {
			x.line("if " + varname + " != nil { ")

			// if varname is a field of a struct (has a dot in it),
			// then just set it to nil
			if strings.IndexByte(varname, '.') != -1 {
				x.line(varname + " = nil")
			} else {
				x.line("*" + varname + " = " + x.genZeroValueR(t.Elem()))
			}
			x.line("}")
		} else {
			x.line(varname + " = " + x.genZeroValueR(t))
		}
		x.line("} else {")
	} else {
		x.line("// cannot be nil")
	}
	if t.Kind() != reflect.Ptr {
		if x.decTryAssignPrimitive(varname, t) {
			x.line(genTempVarPfx + "v" + i + " := &" + varname)
			x.dec(genTempVarPfx+"v"+i, t)
		}
	} else {
		x.linef("if %s == nil { %s = new(%s) }", varname, varname, x.genTypeName(t.Elem()))
		// Ensure we set underlying ptr to a non-nil value (so we can deref to it later).
		// There's a chance of a **T in here which is nil.
		var ptrPfx string
		for t = t.Elem(); t.Kind() == reflect.Ptr; t = t.Elem() {
			ptrPfx += "*"
			x.linef("if %s%s == nil { %s%s = new(%s)}",
				ptrPfx, varname, ptrPfx, varname, x.genTypeName(t))
		}
		// if varname has [ in it, then create temp variable for this ptr thingie
		if strings.Index(varname, "[") >= 0 {
			varname2 := genTempVarPfx + "w" + i
			x.line(varname2 + " := " + varname)
			varname = varname2
		}

		if ptrPfx == "" {
			x.dec(varname, t)
		} else {
			x.line(genTempVarPfx + "z" + i + " := " + ptrPfx + varname)
			x.dec(genTempVarPfx+"z"+i, t)
		}

	}

	if canBeNil {
		x.line("} ")
	}
}

// dec will decode a variable (varname) of type ptrTo(t).
// t is always a basetype (i.e. not of kind reflect.Ptr).
func (x *genRunner) dec(varname string, t reflect.Type) {
	// assumptions:
	//   - the varname is to a pointer already. No need to take address of it
	//   - t is always a baseType T (not a *T, etc).
	rtid := rt2id(t)
	tptr := reflect.PtrTo(t)
	if x.checkForSelfer(t, varname) {
		if t.Implements(selferTyp) || tptr.Implements(selferTyp) {
			x.line(varname + ".CodecDecodeSelf(d)")
			return
		}
		if _, ok := x.td[rtid]; ok {
			x.line(varname + ".CodecDecodeSelf(d)")
			return
		}
	}

	inlist := false
	for _, t0 := range x.t {
		if t == t0 {
			inlist = true
			if x.checkForSelfer(t, varname) {
				x.line(varname + ".CodecDecodeSelf(d)")
				return
			}
			break
		}
	}

	var rtidAdded bool
	if t == x.tc {
		x.td[rtid] = true
		rtidAdded = true
	}

	// check if
	//   - type is Raw, RawExt
	//   - the type implements (Text|JSON|Binary)(Unm|M)arshal
	mi := x.varsfx()
	x.linef("%sm%s := z.DecBinary()", genTempVarPfx, mi)
	x.linef("_ = %sm%s", genTempVarPfx, mi)
	x.line("if false {")           //start if block
	defer func() { x.line("}") }() //end if block

	if t == rawTyp {
		x.linef("} else { *%v = z.DecRaw()", varname)
		return
	}
	if t == rawExtTyp {
		x.linef("} else { r.DecodeExt(%v, 0, nil)", varname)
		return
	}

	// HACK: Support for Builtins.
	//       Currently, only Binc supports builtins, and the only builtin type is time.Time.
	//       Have a method that returns the rtid for time.Time if Handle is Binc.
	if t == timeTyp {
		vrtid := genTempVarPfx + "m" + x.varsfx()
		x.linef("} else if %s := z.TimeRtidIfBinc(); %s != 0 { ", vrtid, vrtid)
		x.linef("r.DecodeBuiltin(%s, %s)", vrtid, varname)
	}
	// only check for extensions if the type is named, and has a packagePath.
	if !x.nx && genImportPath(t) != "" && t.Name() != "" {
		// first check if extensions are configued, before doing the interface conversion
		x.linef("} else if z.HasExtensions() && z.DecExt(%s) {", varname)
	}

	if t.Implements(binaryUnmarshalerTyp) || tptr.Implements(binaryUnmarshalerTyp) {
		x.linef("} else if %sm%s { z.DecBinaryUnmarshal(%v) ", genTempVarPfx, mi, varname)
	}
	if t.Implements(jsonUnmarshalerTyp) || tptr.Implements(jsonUnmarshalerTyp) {
		x.linef("} else if !%sm%s && z.IsJSONHandle() { z.DecJSONUnmarshal(%v)", genTempVarPfx, mi, varname)
	} else if t.Implements(textUnmarshalerTyp) || tptr.Implements(textUnmarshalerTyp) {
		x.linef("} else if !%sm%s { z.DecTextUnmarshal(%v)", genTempVarPfx, mi, varname)
	}

	x.line("} else {")

	// Since these are pointers, we cannot share, and have to use them one by one
	switch t.Kind() {
	case reflect.Int:
		x.line("*((*int)(" + varname + ")) = int(r.DecodeInt(codecSelferBitsize" + x.xs + "))")
		// x.line("z.DecInt((*int)(" + varname + "))")
	case reflect.Int8:
		x.line("*((*int8)(" + varname + ")) = int8(r.DecodeInt(8))")
		// x.line("z.DecInt8((*int8)(" + varname + "))")
	case reflect.Int16:
		x.line("*((*int16)(" + varname + ")) = int16(r.DecodeInt(16))")
		// x.line("z.DecInt16((*int16)(" + varname + "))")
	case reflect.Int32:
		x.line("*((*int32)(" + varname + ")) = int32(r.DecodeInt(32))")
		// x.line("z.DecInt32((*int32)(" + varname + "))")
	case reflect.Int64:
		x.line("*((*int64)(" + varname + ")) = int64(r.DecodeInt(64))")
		// x.line("z.DecInt64((*int64)(" + varname + "))")

	case reflect.Uint:
		x.line("*((*uint)(" + varname + ")) = uint(r.DecodeUint(codecSelferBitsize" + x.xs + "))")
		// x.line("z.DecUint((*uint)(" + varname + "))")
	case reflect.Uint8:
		x.line("*((*uint8)(" + varname + ")) = uint8(r.DecodeUint(8))")
		// x.line("z.DecUint8((*uint8)(" + varname + "))")
	case reflect.Uint16:
		x.line("*((*uint16)(" + varname + ")) = uint16(r.DecodeUint(16))")
		//x.line("z.DecUint16((*uint16)(" + varname + "))")
	case reflect.Uint32:
		x.line("*((*uint32)(" + varname + ")) = uint32(r.DecodeUint(32))")
		//x.line("z.DecUint32((*uint32)(" + varname + "))")
	case reflect.Uint64:
		x.line("*((*uint64)(" + varname + ")) = uint64(r.DecodeUint(64))")
		//x.line("z.DecUint64((*uint64)(" + varname + "))")
	case reflect.Uintptr:
		x.line("*((*uintptr)(" + varname + ")) = uintptr(r.DecodeUint(codecSelferBitsize" + x.xs + "))")

	case reflect.Float32:
		x.line("*((*float32)(" + varname + ")) = float32(r.DecodeFloat(true))")
		//x.line("z.DecFloat32((*float32)(" + varname + "))")
	case reflect.Float64:
		x.line("*((*float64)(" + varname + ")) = float64(r.DecodeFloat(false))")
		// x.line("z.DecFloat64((*float64)(" + varname + "))")

	case reflect.Bool:
		x.line("*((*bool)(" + varname + ")) = r.DecodeBool()")
		// x.line("z.DecBool((*bool)(" + varname + "))")
	case reflect.String:
		x.line("*((*string)(" + varname + ")) = r.DecodeString()")
		// x.line("z.DecString((*string)(" + varname + "))")
	case reflect.Array, reflect.Chan:
		x.xtraSM(varname, false, t)
		// x.decListFallback(varname, rtid, true, t)
	case reflect.Slice:
		// if a []uint8, call dedicated function
		// if a known fastpath slice, call dedicated function
		// else write encode function in-line.
		// - if elements are primitives or Selfers, call dedicated function on each member.
		// - else call Encoder.encode(XXX) on it.
		if rtid == uint8SliceTypId {
			x.line("*" + varname + " = r.DecodeBytes(*(*[]byte)(" + varname + "), false)")
		} else if fastpathAV.index(rtid) != -1 {
			g := x.newGenV(t)
			x.line("z.F." + g.MethodNamePfx("Dec", false) + "X(" + varname + ", d)")
		} else {
			x.xtraSM(varname, false, t)
			// x.decListFallback(varname, rtid, false, t)
		}
	case reflect.Map:
		// if a known fastpath map, call dedicated function
		// else write encode function in-line.
		// - if elements are primitives or Selfers, call dedicated function on each member.
		// - else call Encoder.encode(XXX) on it.
		if fastpathAV.index(rtid) != -1 {
			g := x.newGenV(t)
			x.line("z.F." + g.MethodNamePfx("Dec", false) + "X(" + varname + ", d)")
		} else {
			x.xtraSM(varname, false, t)
			// x.decMapFallback(varname, rtid, t)
		}
	case reflect.Struct:
		if inlist {
			x.decStruct(varname, rtid, t)
		} else {
			// delete(x.td, rtid)
			x.line("z.DecFallback(" + varname + ", false)")
		}
	default:
		if rtidAdded {
			delete(x.te, rtid)
		}
		x.line("z.DecFallback(" + varname + ", true)")
	}
}

func (x *genRunner) decTryAssignPrimitive(varname string, t reflect.Type) (tryAsPtr bool) {
	// This should only be used for exact primitives (ie un-named types).
	// Named types may be implementations of Selfer, Unmarshaler, etc.
	// They should be handled by dec(...)

	if t.Name() != "" {
		tryAsPtr = true
		return
	}

	switch t.Kind() {
	case reflect.Int:
		x.linef("%s = r.DecodeInt(codecSelferBitsize%s)", varname, x.xs)
	case reflect.Int8:
		x.linef("%s = r.DecodeInt(8)", varname)
	case reflect.Int16:
		x.linef("%s = r.DecodeInt(16)", varname)
	case reflect.Int32:
		x.linef("%s = r.DecodeInt(32)", varname)
	case reflect.Int64:
		x.linef("%s = r.DecodeInt(64)", varname)

	case reflect.Uint:
		x.linef("%s = r.DecodeUint(codecSelferBitsize%s)", varname, x.xs)
	case reflect.Uint8:
		x.linef("%s = r.DecodeUint(8)", varname)
	case reflect.Uint16:
		x.linef("%s = r.DecodeUint(16)", varname)
	case reflect.Uint32:
		x.linef("%s = r.DecodeUint(32)", varname)
	case reflect.Uint64:
		x.linef("%s = r.DecodeUint(64)", varname)
	case reflect.Uintptr:
		x.linef("%s = r.DecodeUint(codecSelferBitsize%s)", varname, x.xs)

	case reflect.Float32:
		x.linef("%s = r.DecodeFloat(true)", varname)
	case reflect.Float64:
		x.linef("%s = r.DecodeFloat(false)", varname)

	case reflect.Bool:
		x.linef("%s = r.DecodeBool()", varname)
	case reflect.String:
		x.linef("%s = r.DecodeString()", varname)
	default:
		tryAsPtr = true
	}
	return
}

func (x *genRunner) decListFallback(varname string, rtid uintptr, t reflect.Type) {
	if t.AssignableTo(uint8SliceTyp) {
		x.line("*" + varname + " = r.DecodeBytes(*((*[]byte)(" + varname + ")), false)")
		return
	}
	if t.Kind() == reflect.Array && t.Elem().Kind() == reflect.Uint8 {
		x.linef("r.DecodeBytes( ((*[%s]byte)(%s))[:], true)", t.Len(), varname)
		return
	}
	type tstruc struct {
		TempVar   string
		Rand      string
		Varname   string
		CTyp      string
		Typ       string
		Immutable bool
		Size      int
	}
	telem := t.Elem()
	ts := tstruc{genTempVarPfx, x.varsfx(), varname, x.genTypeName(t), x.genTypeName(telem), genIsImmutable(telem), int(telem.Size())}

	funcs := make(template.FuncMap)

	funcs["decLineVar"] = func(varname string) string {
		x.decVar(varname, "", telem, false)
		return ""
	}
	// funcs["decLine"] = func(pfx string) string {
	// 	x.decVar(ts.TempVar+pfx+ts.Rand, "", reflect.PtrTo(telem), false)
	// 	return ""
	// }
	funcs["var"] = func(s string) string {
		return ts.TempVar + s + ts.Rand
	}
	funcs["zero"] = func() string {
		return x.genZeroValueR(telem)
	}
	funcs["isArray"] = func() bool {
		return t.Kind() == reflect.Array
	}
	funcs["isSlice"] = func() bool {
		return t.Kind() == reflect.Slice
	}
	funcs["isChan"] = func() bool {
		return t.Kind() == reflect.Chan
	}
	tm, err := template.New("").Funcs(funcs).Parse(genDecListTmpl)
	if err != nil {
		panic(err)
	}
	if err = tm.Execute(x.w, &ts); err != nil {
		panic(err)
	}
}

func (x *genRunner) decMapFallback(varname string, rtid uintptr, t reflect.Type) {
	type tstruc struct {
		TempVar string
		Sfx     string
		Rand    string
		Varname string
		KTyp    string
		Typ     string
		Size    int
	}
	telem := t.Elem()
	tkey := t.Key()
	ts := tstruc{
		genTempVarPfx, x.xs, x.varsfx(), varname, x.genTypeName(tkey),
		x.genTypeName(telem), int(telem.Size() + tkey.Size()),
	}

	funcs := make(template.FuncMap)
	funcs["decElemZero"] = func() string {
		return x.genZeroValueR(telem)
	}
	funcs["decElemKindImmutable"] = func() bool {
		return genIsImmutable(telem)
	}
	funcs["decElemKindPtr"] = func() bool {
		return telem.Kind() == reflect.Ptr
	}
	funcs["decElemKindIntf"] = func() bool {
		return telem.Kind() == reflect.Interface
	}
	funcs["decLineVarK"] = func(varname string) string {
		x.decVar(varname, "", tkey, false)
		return ""
	}
	funcs["decLineVar"] = func(varname, decodedNilVarname string) string {
		x.decVar(varname, decodedNilVarname, telem, false)
		return ""
	}
	// funcs["decLineK"] = func(pfx string) string {
	// 	x.decVar(ts.TempVar+pfx+ts.Rand, reflect.PtrTo(tkey), false)
	// 	return ""
	// }
	// funcs["decLine"] = func(pfx string) string {
	// 	x.decVar(ts.TempVar+pfx+ts.Rand, reflect.PtrTo(telem), false)
	// 	return ""
	// }
	funcs["var"] = func(s string) string {
		return ts.TempVar + s + ts.Rand
	}

	tm, err := template.New("").Funcs(funcs).Parse(genDecMapTmpl)
	if err != nil {
		panic(err)
	}
	if err = tm.Execute(x.w, &ts); err != nil {
		panic(err)
	}
}

func (x *genRunner) decStructMapSwitch(kName string, varname string, rtid uintptr, t reflect.Type) {
	ti := x.ti.get(rtid, t)
	tisfi := ti.sfip // always use sequence from file. decStruct expects same thing.
	x.line("switch (" + kName + ") {")
	for _, si := range tisfi {
		x.line("case \"" + si.encName + "\":")
		var t2 reflect.StructField
		{
			//we must accommodate anonymous fields, where the embedded field is a nil pointer in the value.
			// t2 = t.FieldByIndex(si.is)
			t2typ := t
			varname3 := varname
			for ij, ix := range si.is {
				if uint8(ij) == si.nis {
					break
				}
				for t2typ.Kind() == reflect.Ptr {
					t2typ = t2typ.Elem()
				}
				t2 = t2typ.Field(int(ix))
				t2typ = t2.Type
				varname3 = varname3 + "." + t2.Name
				if t2typ.Kind() == reflect.Ptr {
					x.linef("if %s == nil { %s = new(%s) }", varname3, varname3, x.genTypeName(t2typ.Elem()))
				}
			}
		}
		x.decVar(varname+"."+t2.Name, "", t2.Type, false)
	}
	x.line("default:")
	// pass the slice here, so that the string will not escape, and maybe save allocation
	x.line("z.DecStructFieldNotFound(-1, " + kName + ")")
	x.line("} // end switch " + kName)
}

func (x *genRunner) decStructMap(varname, lenvarname string, rtid uintptr, t reflect.Type, style genStructMapStyle) {
	tpfx := genTempVarPfx
	i := x.varsfx()
	kName := tpfx + "s" + i

	// x.line("var " + kName + "Arr = [32]byte{} // default string to decode into")
	// x.line("var " + kName + "Slc = " + kName + "Arr[:] // default slice to decode into")
	// use the scratch buffer to avoid allocation (most field names are < 32).

	x.line("var " + kName + "Slc = z.DecScratchBuffer() // default slice to decode into")

	x.line("_ = " + kName + "Slc")
	switch style {
	case genStructMapStyleLenPrefix:
		x.linef("for %sj%s := 0; %sj%s < %s; %sj%s++ {", tpfx, i, tpfx, i, lenvarname, tpfx, i)
	case genStructMapStyleCheckBreak:
		x.linef("for %sj%s := 0; !r.CheckBreak(); %sj%s++ {", tpfx, i, tpfx, i)
	default: // 0, otherwise.
		x.linef("var %shl%s bool = %s >= 0", tpfx, i, lenvarname) // has length
		x.linef("for %sj%s := 0; ; %sj%s++ {", tpfx, i, tpfx, i)
		x.linef("if %shl%s { if %sj%s >= %s { break }", tpfx, i, tpfx, i, lenvarname)
		x.line("} else { if r.CheckBreak() { break }; }")
	}
	x.line("r.ReadMapElemKey()") // f("z.DecSendContainerState(codecSelfer_containerMapKey%s)", x.xs)
	x.line(kName + "Slc = r.DecodeStringAsBytes()")
	// let string be scoped to this loop alone, so it doesn't escape.
	x.line(kName + " := string(" + kName + "Slc)")
	x.line("r.ReadMapElemValue()") // f("z.DecSendContainerState(codecSelfer_containerMapValue%s)", x.xs)
	x.decStructMapSwitch(kName, varname, rtid, t)

	x.line("} // end for " + tpfx + "j" + i)
	x.line("r.ReadMapEnd()") // f("z.DecSendContainerState(codecSelfer_containerMapEnd%s)", x.xs)
}

func (x *genRunner) decStructArray(varname, lenvarname, breakString string, rtid uintptr, t reflect.Type) {
	tpfx := genTempVarPfx
	i := x.varsfx()
	ti := x.ti.get(rtid, t)
	tisfi := ti.sfip // always use sequence from file. decStruct expects same thing.
	x.linef("var %sj%s int", tpfx, i)
	x.linef("var %sb%s bool", tpfx, i)                        // break
	x.linef("var %shl%s bool = %s >= 0", tpfx, i, lenvarname) // has length
	for _, si := range tisfi {
		var t2 reflect.StructField
		{
			//we must accommodate anonymous fields, where the embedded field is a nil pointer in the value.
			// t2 = t.FieldByIndex(si.is)
			t2typ := t
			varname3 := varname
			for ij, ix := range si.is {
				if uint8(ij) == si.nis {
					break
				}
				for t2typ.Kind() == reflect.Ptr {
					t2typ = t2typ.Elem()
				}
				t2 = t2typ.Field(int(ix))
				t2typ = t2.Type
				varname3 = varname3 + "." + t2.Name
				if t2typ.Kind() == reflect.Ptr {
					x.linef("if %s == nil { %s = new(%s) }", varname3, varname3, x.genTypeName(t2typ.Elem()))
				}
			}
		}

		x.linef("%sj%s++; if %shl%s { %sb%s = %sj%s > %s } else { %sb%s = r.CheckBreak() }",
			tpfx, i, tpfx, i, tpfx, i,
			tpfx, i, lenvarname, tpfx, i)
		x.linef("if %sb%s { r.ReadArrayEnd(); %s }", tpfx, i, breakString)
		// x.linef("if %sb%s { z.DecSendContainerState(codecSelfer_containerArrayEnd%s); %s }", tpfx, i, x.xs, breakString)
		x.line("r.ReadArrayElem()") // f("z.DecSendContainerState(codecSelfer_containerArrayElem%s)", x.xs)
		x.decVar(varname+"."+t2.Name, "", t2.Type, true)
	}
	// read remaining values and throw away.
	x.line("for {")
	x.linef("%sj%s++; if %shl%s { %sb%s = %sj%s > %s } else { %sb%s = r.CheckBreak() }",
		tpfx, i, tpfx, i, tpfx, i,
		tpfx, i, lenvarname, tpfx, i)
	x.linef("if %sb%s { break }", tpfx, i)
	x.line("r.ReadArrayElem()") // f("z.DecSendContainerState(codecSelfer_containerArrayElem%s)", x.xs)
	x.linef(`z.DecStructFieldNotFound(%sj%s - 1, "")`, tpfx, i)
	x.line("}")
	x.line("r.ReadArrayEnd()") // f("z.DecSendContainerState(codecSelfer_containerArrayEnd%s)", x.xs)
}

func (x *genRunner) decStruct(varname string, rtid uintptr, t reflect.Type) {
	// if container is map
	i := x.varsfx()
	x.linef("%sct%s := r.ContainerType()", genTempVarPfx, i)
	x.linef("if %sct%s == codecSelferValueTypeMap%s {", genTempVarPfx, i, x.xs)
	x.line(genTempVarPfx + "l" + i + " := r.ReadMapStart()")
	x.linef("if %sl%s == 0 {", genTempVarPfx, i)
	x.line("r.ReadMapEnd()") // f("z.DecSendContainerState(codecSelfer_containerMapEnd%s)", x.xs)
	if genUseOneFunctionForDecStructMap {
		x.line("} else { ")
		x.linef("x.codecDecodeSelfFromMap(%sl%s, d)", genTempVarPfx, i)
	} else {
		x.line("} else if " + genTempVarPfx + "l" + i + " > 0 { ")
		x.line("x.codecDecodeSelfFromMapLenPrefix(" + genTempVarPfx + "l" + i + ", d)")
		x.line("} else {")
		x.line("x.codecDecodeSelfFromMapCheckBreak(" + genTempVarPfx + "l" + i + ", d)")
	}
	x.line("}")

	// else if container is array
	x.linef("} else if %sct%s == codecSelferValueTypeArray%s {", genTempVarPfx, i, x.xs)
	x.line(genTempVarPfx + "l" + i + " := r.ReadArrayStart()")
	x.linef("if %sl%s == 0 {", genTempVarPfx, i)
	x.line("r.ReadArrayEnd()") // f("z.DecSendContainerState(codecSelfer_containerArrayEnd%s)", x.xs)
	x.line("} else { ")
	x.linef("x.codecDecodeSelfFromArray(%sl%s, d)", genTempVarPfx, i)
	x.line("}")
	// else panic
	x.line("} else { ")
	x.line("panic(codecSelferOnlyMapOrArrayEncodeToStructErr" + x.xs + ")")
	x.line("} ")
}

// --------

type genV struct {
	// genV is either a primitive (Primitive != "") or a map (MapKey != "") or a slice
	MapKey    string
	Elem      string
	Primitive string
	Size      int
}

func (x *genRunner) newGenV(t reflect.Type) (v genV) {
	switch t.Kind() {
	case reflect.Slice, reflect.Array:
		te := t.Elem()
		v.Elem = x.genTypeName(te)
		v.Size = int(te.Size())
	case reflect.Map:
		te, tk := t.Elem(), t.Key()
		v.Elem = x.genTypeName(te)
		v.MapKey = x.genTypeName(tk)
		v.Size = int(te.Size() + tk.Size())
	default:
		panic("unexpected type for newGenV. Requires map or slice type")
	}
	return
}

func (x *genV) MethodNamePfx(prefix string, prim bool) string {
	var name []byte
	if prefix != "" {
		name = append(name, prefix...)
	}
	if prim {
		name = append(name, genTitleCaseName(x.Primitive)...)
	} else {
		if x.MapKey == "" {
			name = append(name, "Slice"...)
		} else {
			name = append(name, "Map"...)
			name = append(name, genTitleCaseName(x.MapKey)...)
		}
		name = append(name, genTitleCaseName(x.Elem)...)
	}
	return string(name)

}

// genImportPath returns import path of a non-predeclared named typed, or an empty string otherwise.
//
// This handles the misbehaviour that occurs when 1.5-style vendoring is enabled,
// where PkgPath returns the full path, including the vendoring pre-fix that should have been stripped.
// We strip it here.
func genImportPath(t reflect.Type) (s string) {
	s = t.PkgPath()
	if genCheckVendor {
		// HACK: always handle vendoring. It should be typically on in go 1.6, 1.7
		s = stripVendor(s)
	}
	return
}

// A go identifier is (letter|_)[letter|number|_]*
func genGoIdentifier(s string, checkFirstChar bool) string {
	b := make([]byte, 0, len(s))
	t := make([]byte, 4)
	var n int
	for i, r := range s {
		if checkFirstChar && i == 0 && !unicode.IsLetter(r) {
			b = append(b, '_')
		}
		// r must be unicode_letter, unicode_digit or _
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			n = utf8.EncodeRune(t, r)
			b = append(b, t[:n]...)
		} else {
			b = append(b, '_')
		}
	}
	return string(b)
}

func genNonPtr(t reflect.Type) reflect.Type {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	return t
}

func genTitleCaseName(s string) string {
	switch s {
	case "interface{}", "interface {}":
		return "Intf"
	default:
		return strings.ToUpper(s[0:1]) + s[1:]
	}
}

func genMethodNameT(t reflect.Type, tRef reflect.Type) (n string) {
	var ptrPfx string
	for t.Kind() == reflect.Ptr {
		ptrPfx += "Ptrto"
		t = t.Elem()
	}
	tstr := t.String()
	if tn := t.Name(); tn != "" {
		if tRef != nil && genImportPath(t) == genImportPath(tRef) {
			return ptrPfx + tn
		} else {
			if genQNameRegex.MatchString(tstr) {
				return ptrPfx + strings.Replace(tstr, ".", "_", 1000)
			} else {
				return ptrPfx + genCustomTypeName(tstr)
			}
		}
	}
	switch t.Kind() {
	case reflect.Map:
		return ptrPfx + "Map" + genMethodNameT(t.Key(), tRef) + genMethodNameT(t.Elem(), tRef)
	case reflect.Slice:
		return ptrPfx + "Slice" + genMethodNameT(t.Elem(), tRef)
	case reflect.Array:
		return ptrPfx + "Array" + strconv.FormatInt(int64(t.Len()), 10) + genMethodNameT(t.Elem(), tRef)
	case reflect.Chan:
		var cx string
		switch t.ChanDir() {
		case reflect.SendDir:
			cx = "ChanSend"
		case reflect.RecvDir:
			cx = "ChanRecv"
		default:
			cx = "Chan"
		}
		return ptrPfx + cx + genMethodNameT(t.Elem(), tRef)
	default:
		if t == intfTyp {
			return ptrPfx + "Interface"
		} else {
			if tRef != nil && genImportPath(t) == genImportPath(tRef) {
				if t.Name() != "" {
					return ptrPfx + t.Name()
				} else {
					return ptrPfx + genCustomTypeName(tstr)
				}
			} else {
				// best way to get the package name inclusive
				// return ptrPfx + strings.Replace(tstr, ".", "_", 1000)
				// return ptrPfx + genBase64enc.EncodeToString([]byte(tstr))
				if t.Name() != "" && genQNameRegex.MatchString(tstr) {
					return ptrPfx + strings.Replace(tstr, ".", "_", 1000)
				} else {
					return ptrPfx + genCustomTypeName(tstr)
				}
			}
		}
	}
}

// genCustomNameForType base64encodes the t.String() value in such a way
// that it can be used within a function name.
func genCustomTypeName(tstr string) string {
	len2 := genBase64enc.EncodedLen(len(tstr))
	bufx := make([]byte, len2)
	genBase64enc.Encode(bufx, []byte(tstr))
	for i := len2 - 1; i >= 0; i-- {
		if bufx[i] == '=' {
			len2--
		} else {
			break
		}
	}
	return string(bufx[:len2])
}

func genIsImmutable(t reflect.Type) (v bool) {
	return isImmutableKind(t.Kind())
}

type genInternal struct {
	Version int
	Values  []genV
}

func (x genInternal) FastpathLen() (l int) {
	for _, v := range x.Values {
		if v.Primitive == "" {
			l++
		}
	}
	return
}

func genInternalZeroValue(s string) string {
	switch s {
	case "interface{}", "interface {}":
		return "nil"
	case "bool":
		return "false"
	case "string":
		return `""`
	default:
		return "0"
	}
}

func genInternalEncCommandAsString(s string, vname string) string {
	switch s {
	case "uint", "uint8", "uint16", "uint32", "uint64":
		return "ee.EncodeUint(uint64(" + vname + "))"
	case "int", "int8", "int16", "int32", "int64":
		return "ee.EncodeInt(int64(" + vname + "))"
	case "string":
		return "ee.EncodeString(c_UTF8, " + vname + ")"
	case "float32":
		return "ee.EncodeFloat32(" + vname + ")"
	case "float64":
		return "ee.EncodeFloat64(" + vname + ")"
	case "bool":
		return "ee.EncodeBool(" + vname + ")"
	case "symbol":
		return "ee.EncodeSymbol(" + vname + ")"
	default:
		return "e.encode(" + vname + ")"
	}
}

func genInternalDecCommandAsString(s string) string {
	switch s {
	case "uint":
		return "uint(dd.DecodeUint(uintBitsize))"
	case "uint8":
		return "uint8(dd.DecodeUint(8))"
	case "uint16":
		return "uint16(dd.DecodeUint(16))"
	case "uint32":
		return "uint32(dd.DecodeUint(32))"
	case "uint64":
		return "dd.DecodeUint(64)"
	case "uintptr":
		return "uintptr(dd.DecodeUint(uintBitsize))"
	case "int":
		return "int(dd.DecodeInt(intBitsize))"
	case "int8":
		return "int8(dd.DecodeInt(8))"
	case "int16":
		return "int16(dd.DecodeInt(16))"
	case "int32":
		return "int32(dd.DecodeInt(32))"
	case "int64":
		return "dd.DecodeInt(64)"

	case "string":
		return "dd.DecodeString()"
	case "float32":
		return "float32(dd.DecodeFloat(true))"
	case "float64":
		return "dd.DecodeFloat(false)"
	case "bool":
		return "dd.DecodeBool()"
	default:
		panic(errors.New("gen internal: unknown type for decode: " + s))
	}
}

func genInternalSortType(s string, elem bool) string {
	for _, v := range [...]string{"int", "uint", "float", "bool", "string"} {
		if strings.HasPrefix(s, v) {
			if elem {
				if v == "int" || v == "uint" || v == "float" {
					return v + "64"
				} else {
					return v
				}
			}
			return v + "Slice"
		}
	}
	panic("sorttype: unexpected type: " + s)
}

func stripVendor(s string) string {
	// HACK: Misbehaviour occurs in go 1.5. May have to re-visit this later.
	// if s contains /vendor/ OR startsWith vendor/, then return everything after it.
	const vendorStart = "vendor/"
	const vendorInline = "/vendor/"
	if i := strings.LastIndex(s, vendorInline); i >= 0 {
		s = s[i+len(vendorInline):]
	} else if strings.HasPrefix(s, vendorStart) {
		s = s[len(vendorStart):]
	}
	return s
}

// var genInternalMu sync.Mutex
var genInternalV = genInternal{Version: genVersion}
var genInternalTmplFuncs template.FuncMap
var genInternalOnce sync.Once

func genInternalInit() {
	types := [...]string{
		"interface{}",
		"string",
		"float32",
		"float64",
		"uint",
		"uint8",
		"uint16",
		"uint32",
		"uint64",
		"uintptr",
		"int",
		"int8",
		"int16",
		"int32",
		"int64",
		"bool",
	}
	// keep as slice, so it is in specific iteration order.
	// Initial order was uint64, string, interface{}, int, int64
	mapvaltypes := [...]string{
		"interface{}",
		"string",
		"uint",
		"uint8",
		"uint16",
		"uint32",
		"uint64",
		"uintptr",
		"int",
		"int8",
		"int16",
		"int32",
		"int64",
		"float32",
		"float64",
		"bool",
	}
	wordSizeBytes := int(intBitsize) / 8

	mapvaltypes2 := map[string]int{
		"interface{}": 2 * wordSizeBytes,
		"string":      2 * wordSizeBytes,
		"uint":        1 * wordSizeBytes,
		"uint8":       1,
		"uint16":      2,
		"uint32":      4,
		"uint64":      8,
		"uintptr":     1 * wordSizeBytes,
		"int":         1 * wordSizeBytes,
		"int8":        1,
		"int16":       2,
		"int32":       4,
		"int64":       8,
		"float32":     4,
		"float64":     8,
		"bool":        1,
	}
	var gt = genInternal{Version: genVersion}

	// For each slice or map type, there must be a (symmetrical) Encode and Decode fast-path function
	for _, s := range types {
		gt.Values = append(gt.Values, genV{Primitive: s, Size: mapvaltypes2[s]})
		if s != "uint8" { // do not generate fast path for slice of bytes. Treat specially already.
			gt.Values = append(gt.Values, genV{Elem: s, Size: mapvaltypes2[s]})
		}
		if _, ok := mapvaltypes2[s]; !ok {
			gt.Values = append(gt.Values, genV{MapKey: s, Elem: s, Size: 2 * mapvaltypes2[s]})
		}
		for _, ms := range mapvaltypes {
			gt.Values = append(gt.Values, genV{MapKey: s, Elem: ms, Size: mapvaltypes2[s] + mapvaltypes2[ms]})
		}
	}

	funcs := make(template.FuncMap)
	// funcs["haspfx"] = strings.HasPrefix
	funcs["encmd"] = genInternalEncCommandAsString
	funcs["decmd"] = genInternalDecCommandAsString
	funcs["zerocmd"] = genInternalZeroValue
	funcs["hasprefix"] = strings.HasPrefix
	funcs["sorttype"] = genInternalSortType

	genInternalV = gt
	genInternalTmplFuncs = funcs
}

// genInternalGoFile is used to generate source files from templates.
// It is run by the program author alone.
// Unfortunately, it has to be exported so that it can be called from a command line tool.
// *** DO NOT USE ***
func genInternalGoFile(r io.Reader, w io.Writer) (err error) {
	genInternalOnce.Do(genInternalInit)

	gt := genInternalV

	t := template.New("").Funcs(genInternalTmplFuncs)

	tmplstr, err := ioutil.ReadAll(r)
	if err != nil {
		return
	}

	if t, err = t.Parse(string(tmplstr)); err != nil {
		return
	}

	var out bytes.Buffer
	err = t.Execute(&out, gt)
	if err != nil {
		return
	}

	bout, err := format.Source(out.Bytes())
	if err != nil {
		w.Write(out.Bytes()) // write out if error, so we can still see.
		// w.Write(bout) // write out if error, as much as possible, so we can still see.
		return
	}
	w.Write(bout)
	return
}
