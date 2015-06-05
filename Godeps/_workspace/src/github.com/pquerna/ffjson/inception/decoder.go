/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ffjsoninception

import (
	"fmt"
	"github.com/pquerna/ffjson/shared"
	"reflect"
)

var validValues []string = []string{
	"FFTok_left_brace",
	"FFTok_left_bracket",
	"FFTok_integer",
	"FFTok_double",
	"FFTok_string",
	"FFTok_bool",
	"FFTok_null",
}

func CreateUnmarshalJSON(ic *Inception, si *StructInfo) error {
	out := ""
	ic.OutputImports[`fflib "github.com/pquerna/ffjson/fflib/v1"`] = true
	if len(si.Fields) > 0 {
		ic.OutputImports[`"bytes"`] = true
	}
	ic.OutputImports[`"fmt"`] = true

	out += tplStr(decodeTpl["header"], header{
		IC: ic,
		SI: si,
	})

	out += tplStr(decodeTpl["ujFunc"], ujFunc{
		SI:          si,
		IC:          ic,
		ValidValues: validValues,
	})

	ic.OutputFuncs = append(ic.OutputFuncs, out)

	return nil
}

func handleField(ic *Inception, name string, typ reflect.Type, ptr bool) string {
	return handleFieldAddr(ic, name, false, typ, ptr)
}

func handleFieldAddr(ic *Inception, name string, takeAddr bool, typ reflect.Type, ptr bool) string {
	out := ""
	out += fmt.Sprintf("/* handler: %s type=%v kind=%v */\n", name, typ, typ.Kind())

	umlx := typ.Implements(unmarshalFasterType) || typeInInception(ic, typ, shared.MustDecoder)
	umlx = umlx || reflect.PtrTo(typ).Implements(unmarshalFasterType)

	umlstd := typ.Implements(unmarshalerType) || reflect.PtrTo(typ).Implements(unmarshalerType)

	out += tplStr(decodeTpl["handleUnmarshaler"], handleUnmarshaler{
		IC:                   ic,
		Name:                 name,
		Typ:                  typ,
		Ptr:                  reflect.Ptr,
		TakeAddr:             takeAddr || ptr,
		UnmarshalJSONFFLexer: umlx,
		Unmarshaler:          umlstd,
	})

	if umlx || umlstd {
		return out
	}

	// TODO(pquerna): generic handling of token type mismatching struct type
	switch typ.Kind() {
	case reflect.Int,
		reflect.Int8,
		reflect.Int16,
		reflect.Int32,
		reflect.Int64:
		out += getAllowTokens(typ.Name(), "FFTok_integer", "FFTok_null")
		out += getNumberHandler(ic, name, takeAddr || ptr, typ, "ParseInt")

	case reflect.Uint,
		reflect.Uint8,
		reflect.Uint16,
		reflect.Uint32,
		reflect.Uint64:
		out += getAllowTokens(typ.Name(), "FFTok_integer", "FFTok_null")
		out += getNumberHandler(ic, name, takeAddr || ptr, typ, "ParseUint")

	case reflect.Float32,
		reflect.Float64:
		out += getAllowTokens(typ.Name(), "FFTok_double", "FFTok_integer", "FFTok_null")
		out += getNumberHandler(ic, name, takeAddr || ptr, typ, "ParseFloat")

	case reflect.Bool:
		ic.OutputImports[`"bytes"`] = true
		ic.OutputImports[`"errors"`] = true
		out += tplStr(decodeTpl["handleBool"], handleBool{
			Name:     name,
			Typ:      typ,
			TakeAddr: takeAddr || ptr,
		})

	case reflect.Ptr:
		out += tplStr(decodeTpl["handlePtr"], handlePtr{
			IC:   ic,
			Name: name,
			Typ:  typ,
		})

	case reflect.Array,
		reflect.Slice:
		if typ.Kind() == reflect.Slice && typ.Elem().Kind() == reflect.Uint8 {
			ic.OutputImports[`"encoding/base64"`] = true
			useReflectToSet := false
			if typ.Elem().Name() != "byte" {
				ic.OutputImports[`"reflect"`] = true
				useReflectToSet = true
			}

			out += tplStr(decodeTpl["handleByteArray"], handleArray{
				IC:              ic,
				Name:            name,
				Typ:             typ,
				Ptr:             reflect.Ptr,
				UseReflectToSet: useReflectToSet,
			})
		} else if typ.Elem().Kind() == reflect.Struct && typ.Elem().Name() != "" {
			out += tplStr(decodeTpl["handleArray"], handleArray{
				IC:   ic,
				Name: name,
				Typ:  typ,
				Ptr:  reflect.Ptr,
			})
		} else if (typ.Elem().Kind() == reflect.Struct || typ.Elem().Kind() == reflect.Map) ||
			typ.Elem().Kind() == reflect.Array || typ.Elem().Kind() == reflect.Slice &&
			typ.Elem().Name() == "" {
			ic.OutputImports[`"encoding/json"`] = true
			out += tplStr(decodeTpl["handleFallback"], handleFallback{
				Name: name,
				Typ:  typ,
				Kind: typ.Kind(),
			})
		} else {
			out += tplStr(decodeTpl["handleArray"], handleArray{
				IC:   ic,
				Name: name,
				Typ:  typ,
				Ptr:  reflect.Ptr,
			})
		}

	case reflect.String:
		out += tplStr(decodeTpl["handleString"], handleString{
			IC:       ic,
			Name:     name,
			Typ:      typ,
			TakeAddr: takeAddr || ptr,
		})
	case reflect.Interface:
		ic.OutputImports[`"encoding/json"`] = true
		out += tplStr(decodeTpl["handleFallback"], handleFallback{
			Name: name,
			Typ:  typ,
			Kind: typ.Kind(),
		})
	default:
		ic.OutputImports[`"encoding/json"`] = true
		out += tplStr(decodeTpl["handleFallback"], handleFallback{
			Name: name,
			Typ:  typ,
			Kind: typ.Kind(),
		})
	}

	return out
}

func getAllowTokens(name string, tokens ...string) string {
	return tplStr(decodeTpl["allowTokens"], allowTokens{
		Name:   name,
		Tokens: tokens,
	})
}

func getNumberHandler(ic *Inception, name string, takeAddr bool, typ reflect.Type, parsefunc string) string {
	return tplStr(decodeTpl["handlerNumeric"], handlerNumeric{
		IC:        ic,
		Name:      name,
		ParseFunc: parsefunc,
		TakeAddr:  takeAddr,
		Typ:       typ,
	})
}

func getNumberSize(typ reflect.Type) string {
	return fmt.Sprintf("%d", typ.Bits())
}

func getType(ic *Inception, name string, typ reflect.Type) string {
	s := typ.Name()

	if typ.PkgPath() != "" && typ.PkgPath() != ic.PackagePath {
		ic.OutputImports[`"`+typ.PkgPath()+`"`] = true
		s = typ.String()
	}

	if s == "" {
		switch typ.Kind() {
		case reflect.Interface:
			return "interface{}"
		case reflect.Slice:
			return "[]" + typ.Elem().String()
		}
		panic("non-numeric type " + typ.String() + " passed in w/o name: " + name)
	}

	return s
}
