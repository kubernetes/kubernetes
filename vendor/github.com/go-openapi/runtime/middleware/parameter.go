// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package middleware

import (
	"encoding"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strconv"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/swag"
	"github.com/go-openapi/validate"
)

const defaultMaxMemory = 32 << 20

var textUnmarshalType = reflect.TypeOf(new(encoding.TextUnmarshaler)).Elem()

func newUntypedParamBinder(param spec.Parameter, spec *spec.Swagger, formats strfmt.Registry) *untypedParamBinder {
	binder := new(untypedParamBinder)
	binder.Name = param.Name
	binder.parameter = &param
	binder.formats = formats
	if param.In != "body" {
		binder.validator = validate.NewParamValidator(&param, formats)
	} else {
		binder.validator = validate.NewSchemaValidator(param.Schema, spec, param.Name, formats)
	}

	return binder
}

type untypedParamBinder struct {
	parameter *spec.Parameter
	formats   strfmt.Registry
	Name      string
	validator validate.EntityValidator
}

func (p *untypedParamBinder) Type() reflect.Type {
	return p.typeForSchema(p.parameter.Type, p.parameter.Format, p.parameter.Items)
}

func (p *untypedParamBinder) typeForSchema(tpe, format string, items *spec.Items) reflect.Type {
	switch tpe {
	case "boolean":
		return reflect.TypeOf(true)

	case "string":
		if tt, ok := p.formats.GetType(format); ok {
			return tt
		}
		return reflect.TypeOf("")

	case "integer":
		switch format {
		case "int8":
			return reflect.TypeOf(int8(0))
		case "int16":
			return reflect.TypeOf(int16(0))
		case "int32":
			return reflect.TypeOf(int32(0))
		case "int64":
			return reflect.TypeOf(int64(0))
		default:
			return reflect.TypeOf(int64(0))
		}

	case "number":
		switch format {
		case "float":
			return reflect.TypeOf(float32(0))
		case "double":
			return reflect.TypeOf(float64(0))
		}

	case "array":
		if items == nil {
			return nil
		}
		itemsType := p.typeForSchema(items.Type, items.Format, items.Items)
		if itemsType == nil {
			return nil
		}
		return reflect.MakeSlice(reflect.SliceOf(itemsType), 0, 0).Type()

	case "file":
		return reflect.TypeOf(&runtime.File{}).Elem()

	case "object":
		return reflect.TypeOf(map[string]interface{}{})
	}
	return nil
}

func (p *untypedParamBinder) allowsMulti() bool {
	return p.parameter.In == "query" || p.parameter.In == "formData"
}

func (p *untypedParamBinder) readValue(values runtime.Gettable, target reflect.Value) ([]string, bool, bool, error) {
	name, in, cf, tpe := p.parameter.Name, p.parameter.In, p.parameter.CollectionFormat, p.parameter.Type
	if tpe == "array" {
		if cf == "multi" {
			if !p.allowsMulti() {
				return nil, false, false, errors.InvalidCollectionFormat(name, in, cf)
			}
			vv, hasKey, _ := values.GetOK(name)
			return vv, false, hasKey, nil
		}

		v, hk, hv := values.GetOK(name)
		if !hv {
			return nil, false, hk, nil
		}
		d, c, e := p.readFormattedSliceFieldValue(v[len(v)-1], target)
		return d, c, hk, e
	}

	vv, hk, _ := values.GetOK(name)
	return vv, false, hk, nil
}

func (p *untypedParamBinder) Bind(request *http.Request, routeParams RouteParams, consumer runtime.Consumer, target reflect.Value) error {
	// fmt.Println("binding", p.name, "as", p.Type())
	switch p.parameter.In {
	case "query":
		data, custom, hasKey, err := p.readValue(runtime.Values(request.URL.Query()), target)
		if err != nil {
			return err
		}
		if custom {
			return nil
		}

		return p.bindValue(data, hasKey, target)

	case "header":
		data, custom, hasKey, err := p.readValue(runtime.Values(request.Header), target)
		if err != nil {
			return err
		}
		if custom {
			return nil
		}
		return p.bindValue(data, hasKey, target)

	case "path":
		data, custom, hasKey, err := p.readValue(routeParams, target)
		if err != nil {
			return err
		}
		if custom {
			return nil
		}
		return p.bindValue(data, hasKey, target)

	case "formData":
		var err error
		var mt string

		mt, _, e := runtime.ContentType(request.Header)
		if e != nil {
			// because of the interface conversion go thinks the error is not nil
			// so we first check for nil and then set the err var if it's not nil
			err = e
		}

		if err != nil {
			return errors.InvalidContentType("", []string{"multipart/form-data", "application/x-www-form-urlencoded"})
		}

		if mt != "multipart/form-data" && mt != "application/x-www-form-urlencoded" {
			return errors.InvalidContentType(mt, []string{"multipart/form-data", "application/x-www-form-urlencoded"})
		}

		if mt == "multipart/form-data" {
			if err := request.ParseMultipartForm(defaultMaxMemory); err != nil {
				return errors.NewParseError(p.Name, p.parameter.In, "", err)
			}
		}

		if err := request.ParseForm(); err != nil {
			return errors.NewParseError(p.Name, p.parameter.In, "", err)
		}

		if p.parameter.Type == "file" {
			file, header, err := request.FormFile(p.parameter.Name)
			if err != nil {
				return errors.NewParseError(p.Name, p.parameter.In, "", err)
			}
			target.Set(reflect.ValueOf(runtime.File{Data: file, Header: header}))
			return nil
		}

		if request.MultipartForm != nil {
			data, custom, hasKey, err := p.readValue(runtime.Values(request.MultipartForm.Value), target)
			if err != nil {
				return err
			}
			if custom {
				return nil
			}
			return p.bindValue(data, hasKey, target)
		}
		data, custom, hasKey, err := p.readValue(runtime.Values(request.PostForm), target)
		if err != nil {
			return err
		}
		if custom {
			return nil
		}
		return p.bindValue(data, hasKey, target)

	case "body":
		newValue := reflect.New(target.Type())
		if !runtime.HasBody(request) {
			if p.parameter.Default != nil {
				target.Set(reflect.ValueOf(p.parameter.Default))
			}

			return nil
		}
		if err := consumer.Consume(request.Body, newValue.Interface()); err != nil {
			if err == io.EOF && p.parameter.Default != nil {
				target.Set(reflect.ValueOf(p.parameter.Default))
				return nil
			}
			tpe := p.parameter.Type
			if p.parameter.Format != "" {
				tpe = p.parameter.Format
			}
			return errors.InvalidType(p.Name, p.parameter.In, tpe, nil)
		}
		target.Set(reflect.Indirect(newValue))
		return nil
	default:
		return errors.New(500, fmt.Sprintf("invalid parameter location %q", p.parameter.In))
	}
}

func (p *untypedParamBinder) bindValue(data []string, hasKey bool, target reflect.Value) error {
	if p.parameter.Type == "array" {
		return p.setSliceFieldValue(target, p.parameter.Default, data, hasKey)
	}
	var d string
	if len(data) > 0 {
		d = data[len(data)-1]
	}
	return p.setFieldValue(target, p.parameter.Default, d, hasKey)
}

func (p *untypedParamBinder) setFieldValue(target reflect.Value, defaultValue interface{}, data string, hasKey bool) error {
	tpe := p.parameter.Type
	if p.parameter.Format != "" {
		tpe = p.parameter.Format
	}

	if (!hasKey || (!p.parameter.AllowEmptyValue && data == "")) && p.parameter.Required && p.parameter.Default == nil {
		return errors.Required(p.Name, p.parameter.In)
	}

	ok, err := p.tryUnmarshaler(target, defaultValue, data)
	if err != nil {
		return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
	}
	if ok {
		return nil
	}

	defVal := reflect.Zero(target.Type())
	if defaultValue != nil {
		defVal = reflect.ValueOf(defaultValue)
	}

	if tpe == "byte" {
		if data == "" {
			if target.CanSet() {
				target.SetBytes(defVal.Bytes())
			}
			return nil
		}

		b, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			b, err = base64.URLEncoding.DecodeString(data)
			if err != nil {
				return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
			}
		}
		if target.CanSet() {
			target.SetBytes(b)
		}
		return nil
	}

	switch target.Kind() {
	case reflect.Bool:
		if data == "" {
			if target.CanSet() {
				target.SetBool(defVal.Bool())
			}
			return nil
		}
		b, err := swag.ConvertBool(data)
		if err != nil {
			return err
		}
		if target.CanSet() {
			target.SetBool(b)
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if data == "" {
			if target.CanSet() {
				target.SetInt(defVal.Int())
			}
			return nil
		}
		i, err := strconv.ParseInt(data, 10, 64)
		if err != nil {
			return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
		}
		if target.OverflowInt(i) {
			return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
		}
		if target.CanSet() {
			target.SetInt(i)
		}

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if data == "" {
			if target.CanSet() {
				target.SetUint(defVal.Uint())
			}
			return nil
		}
		u, err := strconv.ParseUint(data, 10, 64)
		if err != nil {
			return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
		}
		if target.OverflowUint(u) {
			return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
		}
		if target.CanSet() {
			target.SetUint(u)
		}

	case reflect.Float32, reflect.Float64:
		if data == "" {
			if target.CanSet() {
				target.SetFloat(defVal.Float())
			}
			return nil
		}
		f, err := strconv.ParseFloat(data, 64)
		if err != nil {
			return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
		}
		if target.OverflowFloat(f) {
			return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
		}
		if target.CanSet() {
			target.SetFloat(f)
		}

	case reflect.String:
		value := data
		if value == "" {
			value = defVal.String()
		}
		// validate string
		if target.CanSet() {
			target.SetString(value)
		}

	case reflect.Ptr:
		if data == "" && defVal.Kind() == reflect.Ptr {
			if target.CanSet() {
				target.Set(defVal)
			}
			return nil
		}
		newVal := reflect.New(target.Type().Elem())
		if err := p.setFieldValue(reflect.Indirect(newVal), defVal, data, hasKey); err != nil {
			return err
		}
		if target.CanSet() {
			target.Set(newVal)
		}

	default:
		return errors.InvalidType(p.Name, p.parameter.In, tpe, data)
	}
	return nil
}

func (p *untypedParamBinder) tryUnmarshaler(target reflect.Value, defaultValue interface{}, data string) (bool, error) {
	if !target.CanSet() {
		return false, nil
	}
	// When a type implements encoding.TextUnmarshaler we'll use that instead of reflecting some more
	if reflect.PtrTo(target.Type()).Implements(textUnmarshalType) {
		if defaultValue != nil && len(data) == 0 {
			target.Set(reflect.ValueOf(defaultValue))
			return true, nil
		}
		value := reflect.New(target.Type())
		if err := value.Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(data)); err != nil {
			return true, err
		}
		target.Set(reflect.Indirect(value))
		return true, nil
	}
	return false, nil
}

func (p *untypedParamBinder) readFormattedSliceFieldValue(data string, target reflect.Value) ([]string, bool, error) {
	ok, err := p.tryUnmarshaler(target, p.parameter.Default, data)
	if err != nil {
		return nil, true, err
	}
	if ok {
		return nil, true, nil
	}

	return swag.SplitByFormat(data, p.parameter.CollectionFormat), false, nil
}

func (p *untypedParamBinder) setSliceFieldValue(target reflect.Value, defaultValue interface{}, data []string, hasKey bool) error {
	sz := len(data)
	if (!hasKey || (!p.parameter.AllowEmptyValue && (sz == 0 || (sz == 1 && data[0] == "")))) && p.parameter.Required && defaultValue == nil {
		return errors.Required(p.Name, p.parameter.In)
	}

	defVal := reflect.Zero(target.Type())
	if defaultValue != nil {
		defVal = reflect.ValueOf(defaultValue)
	}

	if !target.CanSet() {
		return nil
	}
	if sz == 0 {
		target.Set(defVal)
		return nil
	}

	value := reflect.MakeSlice(reflect.SliceOf(target.Type().Elem()), sz, sz)

	for i := 0; i < sz; i++ {
		if err := p.setFieldValue(value.Index(i), nil, data[i], hasKey); err != nil {
			return err
		}
	}

	target.Set(value)

	return nil
}
