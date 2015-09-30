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
	"reflect"
	"text/template"
)

var decodeTpl map[string]*template.Template

func init() {
	decodeTpl = make(map[string]*template.Template)

	funcs := map[string]string{
		"handlerNumeric":    handlerNumericTxt,
		"allowTokens":       allowTokensTxt,
		"handleFallback":    handleFallbackTxt,
		"handleString":      handleStringTxt,
		"handleArray":       handleArrayTxt,
		"handleByteArray":   handleByteArrayTxt,
		"handleBool":        handleBoolTxt,
		"handlePtr":         handlePtrTxt,
		"header":            headerTxt,
		"ujFunc":            ujFuncTxt,
		"handleUnmarshaler": handleUnmarshalerTxt,
	}

	tplFuncs := template.FuncMap{
		"getAllowTokens":  getAllowTokens,
		"getNumberSize":   getNumberSize,
		"getType":         getType,
		"handleField":     handleField,
		"handleFieldAddr": handleFieldAddr,
	}

	for k, v := range funcs {
		decodeTpl[k] = template.Must(template.New(k).Funcs(tplFuncs).Parse(v))
	}
}

type handlerNumeric struct {
	IC        *Inception
	Name      string
	ParseFunc string
	Typ       reflect.Type
	TakeAddr  bool
}

var handlerNumericTxt = `
{
	{{$ic := .IC}}

	if tok == fflib.FFTok_null {
		{{if eq .TakeAddr true}}
		{{.Name}} = nil
		{{end}}
	} else {
		{{if eq .ParseFunc "ParseFloat" }}
		tval, err := fflib.{{ .ParseFunc}}(fs.Output.Bytes(), {{getNumberSize .Typ}})
		{{else}}
		tval, err := fflib.{{ .ParseFunc}}(fs.Output.Bytes(), 10, {{getNumberSize .Typ}})
		{{end}}

		if err != nil {
			return fs.WrapErr(err)
		}
		{{if eq .TakeAddr true}}
		ttypval := {{getType $ic .Name .Typ}}(tval)
		{{.Name}} = &ttypval
		{{else}}
		{{.Name}} = {{getType $ic .Name .Typ}}(tval)
		{{end}}
	}
}
`

type allowTokens struct {
	Name   string
	Tokens []string
}

var allowTokensTxt = `
{
	if {{range $index, $element := .Tokens}}{{if ne $index 0 }}&&{{end}} tok != fflib.{{$element}}{{end}} {
		return fs.WrapErr(fmt.Errorf("cannot unmarshal %s into Go value for {{.Name}}", tok))
	}
}
`

type handleFallback struct {
	Name string
	Typ  reflect.Type
	Kind reflect.Kind
}

var handleFallbackTxt = `
{
	/* Falling back. type={{printf "%v" .Typ}} kind={{printf "%v" .Kind}} */
	tbuf, err := fs.CaptureField(tok)
	if err != nil {
		return fs.WrapErr(err)
	}

	err = json.Unmarshal(tbuf, &{{.Name}})
	if err != nil {
		return fs.WrapErr(err)
	}
}
`

type handleString struct {
	IC       *Inception
	Name     string
	Typ      reflect.Type
	TakeAddr bool
}

var handleStringTxt = `
{
	{{$ic := .IC}}

	{{getAllowTokens .Typ.Name "FFTok_string" "FFTok_null"}}
	if tok == fflib.FFTok_null {
	{{if eq .TakeAddr true}}
		{{.Name}} = nil
	{{end}}
	} else {
	{{if eq .TakeAddr true}}
		var tval {{getType $ic .Name .Typ}}
		tval = {{getType $ic .Name .Typ}}(fs.Output.String())
		{{.Name}} = &tval
	{{else}}
		{{.Name}} = {{getType $ic .Name .Typ}}(fs.Output.String())
	{{end}}
	}
}
`

type handleArray struct {
	IC              *Inception
	Name            string
	Typ             reflect.Type
	Ptr             reflect.Kind
	UseReflectToSet bool
}

var handleArrayTxt = `
{
	{{$ic := .IC}}
	{{getAllowTokens .Typ.Name "FFTok_left_brace" "FFTok_null"}}
	if tok == fflib.FFTok_null {
		{{.Name}} = nil
	} else {


		{{if eq .Typ.Elem.Kind .Ptr }}
			{{.Name}} = make([]*{{getType $ic .Name .Typ.Elem.Elem}}, 0)
		{{else}}
			{{.Name}} = make([]{{getType $ic .Name .Typ.Elem}}, 0)
		{{end}}

		wantVal := true

		for {
		{{$ptr := false}}
		{{if eq .Typ.Elem.Kind .Ptr }}
			{{$ptr := true}}
			var v *{{getType $ic .Name .Typ.Elem.Elem}}
		{{else}}
			var v {{getType $ic .Name .Typ.Elem}}
		{{end}}

			tok = fs.Scan()
			if tok == fflib.FFTok_error {
				goto tokerror
			}
			if tok == fflib.FFTok_right_brace {
				break
			}

			if tok == fflib.FFTok_comma {
				if wantVal == true {
					// TODO(pquerna): this isn't an ideal error message, this handles 
					// things like [,,,] as an array value.
					return fs.WrapErr(fmt.Errorf("wanted value token, but got token: %v", tok))
				}
				continue
			} else {
				wantVal = true
			}

			{{handleField .IC "v" .Typ.Elem $ptr}}
			{{.Name}} = append({{.Name}}, v)
			wantVal = false
		}
	}
}
`

var handleByteArrayTxt = `
{
	{{getAllowTokens .Typ.Name "FFTok_string" "FFTok_null"}}
	if tok == fflib.FFTok_null {
		{{.Name}} = nil
	} else {
		b := make([]byte, base64.StdEncoding.DecodedLen(fs.Output.Len()))
		n, err := base64.StdEncoding.Decode(b, fs.Output.Bytes())
		if err != nil {
			return fs.WrapErr(err)
		}
		{{if eq .UseReflectToSet true}}
			v := reflect.ValueOf(&{{.Name}}).Elem()
			v.SetBytes(b[0:n])
		{{else}}
			{{.Name}} = append([]byte(), b[0:n]...)
		{{end}}
	}
}
`

type handleBool struct {
	Name     string
	Typ      reflect.Type
	TakeAddr bool
}

var handleBoolTxt = `
{
	{{getAllowTokens .Typ.Name "FFTok_bool" "FFTok_null"}}
	if tok == fflib.FFTok_null {
		{{if eq .TakeAddr true}}
		{{.Name}} = nil
		{{end}}
	} else {
		tmpb := fs.Output.Bytes()

		{{if eq .TakeAddr true}}
		var tval bool
		{{end}}

		if bytes.Compare([]byte{'t', 'r', 'u', 'e'}, tmpb) == 0 {
		{{if eq .TakeAddr true}}
			tval = true
		{{else}}
			{{.Name}} = true
		{{end}}
		} else if bytes.Compare([]byte{'f', 'a', 'l', 's', 'e'}, tmpb) == 0 {
		{{if eq .TakeAddr true}}
			tval = false
		{{else}}
			{{.Name}} = false
		{{end}}
		} else {
			err = errors.New("unexpected bytes for true/false value")
			return fs.WrapErr(err)
		}

		{{if eq .TakeAddr true}}
		{{.Name}} = &tval
		{{end}}
	}
}
`

type handlePtr struct {
	IC   *Inception
	Name string
	Typ  reflect.Type
}

var handlePtrTxt = `
{
	{{$ic := .IC}}

	if tok == fflib.FFTok_null {
		{{.Name}} = nil
	} else {
		if {{.Name}} == nil {
			{{.Name}} = new({{getType $ic .Typ.Elem.Name .Typ.Elem}})
		}

		{{handleFieldAddr .IC .Name true .Typ.Elem false}}
	}
}
`

type header struct {
	IC *Inception
	SI *StructInfo
}

var headerTxt = `
const (
	ffj_t_{{.SI.Name}}base = iota
	ffj_t_{{.SI.Name}}no_such_key
	{{with $si := .SI}}
		{{range $index, $field := $si.Fields}}
			{{if ne $field.JsonName "-"}}
		ffj_t_{{$si.Name}}_{{$field.Name}}
			{{end}}
		{{end}}
	{{end}}
)

{{with $si := .SI}}
	{{range $index, $field := $si.Fields}}
		{{if ne $field.JsonName "-"}}
var ffj_key_{{$si.Name}}_{{$field.Name}} = []byte({{$field.JsonName}})
		{{end}}
	{{end}}
{{end}}

`

type ujFunc struct {
	IC          *Inception
	SI          *StructInfo
	ValidValues []string
}

var ujFuncTxt = `
{{$si := .SI}}
{{$ic := .IC}}

func (uj *{{.SI.Name}}) UnmarshalJSON(input []byte) error {
	fs := fflib.NewFFLexer(input)
    return uj.UnmarshalJSONFFLexer(fs, fflib.FFParse_map_start)
}

func (uj *{{.SI.Name}}) UnmarshalJSONFFLexer(fs *fflib.FFLexer, state fflib.FFParseState) error {
	var err error = nil
	currentKey := ffj_t_{{.SI.Name}}base
	_ = currentKey
	tok := fflib.FFTok_init
	wantedTok := fflib.FFTok_init

mainparse:
	for {
		tok = fs.Scan()
		//	println(fmt.Sprintf("debug: tok: %v  state: %v", tok, state))
		if tok == fflib.FFTok_error {
			goto tokerror
		}

		switch state {

		case fflib.FFParse_map_start:
			if tok != fflib.FFTok_left_bracket {
				wantedTok = fflib.FFTok_left_bracket
				goto wrongtokenerror
			}
			state = fflib.FFParse_want_key
			continue

		case fflib.FFParse_after_value:
			if tok == fflib.FFTok_comma {
				state = fflib.FFParse_want_key
			} else if tok == fflib.FFTok_right_bracket {
				goto done
			} else {
				wantedTok = fflib.FFTok_comma
				goto wrongtokenerror
			}

		case fflib.FFParse_want_key:
			// json {} ended. goto exit. woo.
			if tok == fflib.FFTok_right_bracket {
				goto done
			}
			if tok != fflib.FFTok_string {
				wantedTok = fflib.FFTok_string
				goto wrongtokenerror
			}

			kn := fs.Output.Bytes()
			if len(kn) <= 0 {
				// "" case. hrm.
				currentKey = ffj_t_{{.SI.Name}}no_such_key
				state = fflib.FFParse_want_colon
				goto mainparse
			} else {
				switch kn[0] {
				{{range $byte, $fields := $si.FieldsByFirstByte}}
				case '{{$byte}}':
					{{range $index, $field := $fields}}
						{{if ne $index 0 }}} else if {{else}}if {{end}} bytes.Equal(ffj_key_{{$si.Name}}_{{$field.Name}}, kn) {
						currentKey = ffj_t_{{$si.Name}}_{{$field.Name}}
						state = fflib.FFParse_want_colon
						goto mainparse
					{{end}} }
				{{end}}
				}
				{{range $index, $field := $si.ReverseFields}}
				if {{$field.FoldFuncName}}(ffj_key_{{$si.Name}}_{{$field.Name}}, kn) {
					currentKey = ffj_t_{{$si.Name}}_{{$field.Name}}
					state = fflib.FFParse_want_colon
					goto mainparse
				}
				{{end}}
				currentKey = ffj_t_{{.SI.Name}}no_such_key
				state = fflib.FFParse_want_colon
				goto mainparse
			}

		case fflib.FFParse_want_colon:
			if tok != fflib.FFTok_colon {
				wantedTok = fflib.FFTok_colon
				goto wrongtokenerror
			}
			state = fflib.FFParse_want_value
			continue
		case fflib.FFParse_want_value:

			if {{range $index, $v := .ValidValues}}{{if ne $index 0 }}||{{end}}tok == fflib.{{$v}}{{end}} {
				switch currentKey {
				{{range $index, $field := $si.Fields}}
				case ffj_t_{{$si.Name}}_{{$field.Name}}:
					goto handle_{{$field.Name}}
				{{end}}
				case ffj_t_{{$si.Name}}no_such_key:
					err = fs.SkipField(tok)
					if err != nil {
						return fs.WrapErr(err)
					}
					state = fflib.FFParse_after_value
					goto mainparse
				}
			} else {
				goto wantedvalue
			}
		}
	}

{{range $index, $field := $si.Fields}}
handle_{{$field.Name}}:
	{{with $fieldName := $field.Name | printf "uj.%s"}}
		{{handleField $ic $fieldName $field.Typ $field.Pointer}}
		state = fflib.FFParse_after_value
		goto mainparse
	{{end}}
{{end}}

wantedvalue:
	return fs.WrapErr(fmt.Errorf("wanted value token, but got token: %v", tok))
wrongtokenerror:
	return fs.WrapErr(fmt.Errorf("ffjson: wanted token: %v, but got token: %v output=%s", wantedTok, tok, fs.Output.String()))
tokerror:
	if fs.BigError != nil {
		return fs.WrapErr(fs.BigError)
	}
	err = fs.Error.ToError()
	if err != nil {
		return fs.WrapErr(err)
	}
	panic("ffjson-generated: unreachable, please report bug.")
done:
	return nil
}

`

type handleUnmarshaler struct {
	IC                   *Inception
	Name                 string
	Typ                  reflect.Type
	Ptr                  reflect.Kind
	TakeAddr             bool
	UnmarshalJSONFFLexer bool
	Unmarshaler          bool
}

var handleUnmarshalerTxt = `
	{{$ic := .IC}}

	{{if eq .UnmarshalJSONFFLexer true}}
	{
		if tok == fflib.FFTok_null {
				{{if eq .Typ.Kind .Ptr }}
					{{.Name}} = nil
				{{end}}
				{{if eq .TakeAddr true }}
					{{.Name}} = nil
				{{end}}
				state = fflib.FFParse_after_value
				goto mainparse
		}
		{{if eq .Typ.Kind .Ptr }}
			if {{.Name}} == nil {
				{{.Name}} = new({{getType $ic .Typ.Elem.Name .Typ.Elem}})
			}
		{{end}}
		{{if eq .TakeAddr true }}
			if {{.Name}} == nil {
				{{.Name}} = new({{getType $ic .Typ.Name .Typ}})
			}
		{{end}}
		err = {{.Name}}.UnmarshalJSONFFLexer(fs, fflib.FFParse_want_key)
		if err != nil {
			return err
		}
		state = fflib.FFParse_after_value
	}
	{{else}}
	{{if eq .Unmarshaler true}}
	{
		if tok == fflib.FFTok_null {
			{{if eq .TakeAddr true }}
				{{.Name}} = nil
			{{end}}
			state = fflib.FFParse_after_value
			goto mainparse
		}

		tbuf, err := fs.CaptureField(tok)
		if err != nil {
			return fs.WrapErr(err)
		}

		{{if eq .TakeAddr true }}
		if {{.Name}} == nil {
			{{.Name}} = new({{getType $ic .Typ.Name .Typ}})
		}
		{{end}}
		err = {{.Name}}.UnmarshalJSON(tbuf)
		if err != nil {
			return fs.WrapErr(err)
		}
		state = fflib.FFParse_after_value
	}
	{{end}}
	{{end}}
`
