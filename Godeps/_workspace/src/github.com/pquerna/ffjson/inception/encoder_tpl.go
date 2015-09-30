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
	"text/template"
)

var encodeTpl map[string]*template.Template

func init() {
	encodeTpl = make(map[string]*template.Template)

	funcs := map[string]string{
		"handleMarshaler": handleMarshalerTxt,
	}
	tplFuncs := template.FuncMap{}

	for k, v := range funcs {
		encodeTpl[k] = template.Must(template.New(k).Funcs(tplFuncs).Parse(v))
	}
}

type handleMarshaler struct {
	IC             *Inception
	Name           string
	MarshalJSONBuf bool
	Marshaler      bool
}

var handleMarshalerTxt = `
	{{if eq .MarshalJSONBuf true}}
	{
		err = {{.Name}}.MarshalJSONBuf(buf)
		if err != nil {
			return err
		}
	}
	{{else if eq .Marshaler true}}
	{
		obj, err = {{.Name}}.MarshalJSON()
		if err != nil {
			return err
		}
		buf.Write(obj)
	}
	{{end}}
`
