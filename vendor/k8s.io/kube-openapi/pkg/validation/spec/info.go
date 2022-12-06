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

package spec

import (
	"encoding/json"
	"strings"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
)

// Extensions vendor specific extensions
type Extensions map[string]interface{}

// Add adds a value to these extensions
func (e Extensions) Add(key string, value interface{}) {
	realKey := strings.ToLower(key)
	e[realKey] = value
}

// GetString gets a string value from the extensions
func (e Extensions) GetString(key string) (string, bool) {
	if v, ok := e[strings.ToLower(key)]; ok {
		str, ok := v.(string)
		return str, ok
	}
	return "", false
}

// GetBool gets a string value from the extensions
func (e Extensions) GetBool(key string) (bool, bool) {
	if v, ok := e[strings.ToLower(key)]; ok {
		str, ok := v.(bool)
		return str, ok
	}
	return false, false
}

// GetStringSlice gets a string value from the extensions
func (e Extensions) GetStringSlice(key string) ([]string, bool) {
	if v, ok := e[strings.ToLower(key)]; ok {
		arr, isSlice := v.([]interface{})
		if !isSlice {
			return nil, false
		}
		var strs []string
		for _, iface := range arr {
			str, isString := iface.(string)
			if !isString {
				return nil, false
			}
			strs = append(strs, str)
		}
		return strs, ok
	}
	return nil, false
}

// GetObject gets the object value from the extensions.
// out must be a json serializable type; the json go struct
// tags of out are used to populate it.
func (e Extensions) GetObject(key string, out interface{}) error {
	// This json serialization/deserialization could be replaced with
	// an approach using reflection if the optimization becomes justified.
	if v, ok := e[strings.ToLower(key)]; ok {
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		err = json.Unmarshal(b, out)
		if err != nil {
			return err
		}
	}
	return nil
}

func (e Extensions) sanitize() {
	for k := range e {
		if !isExtensionKey(k) {
			delete(e, k)
		}
	}
}

func (e Extensions) sanitizeWithExtra() (extra map[string]any) {
	for k, v := range e {
		if !isExtensionKey(k) {
			if extra == nil {
				extra = make(map[string]any)
			}
			extra[k] = v
			delete(e, k)
		}
	}
	return extra
}

func isExtensionKey(k string) bool {
	return len(k) > 1 && (k[0] == 'x' || k[0] == 'X') && k[1] == '-'
}

// VendorExtensible composition block.
type VendorExtensible struct {
	Extensions Extensions
}

// AddExtension adds an extension to this extensible object
func (v *VendorExtensible) AddExtension(key string, value interface{}) {
	if value == nil {
		return
	}
	if v.Extensions == nil {
		v.Extensions = make(map[string]interface{})
	}
	v.Extensions.Add(key, value)
}

// MarshalJSON marshals the extensions to json
func (v VendorExtensible) MarshalJSON() ([]byte, error) {
	toser := make(map[string]interface{})
	for k, v := range v.Extensions {
		lk := strings.ToLower(k)
		if strings.HasPrefix(lk, "x-") {
			toser[k] = v
		}
	}
	return json.Marshal(toser)
}

// UnmarshalJSON for this extensible object
func (v *VendorExtensible) UnmarshalJSON(data []byte) error {
	var d map[string]interface{}
	if err := json.Unmarshal(data, &d); err != nil {
		return err
	}
	for k, vv := range d {
		lk := strings.ToLower(k)
		if strings.HasPrefix(lk, "x-") {
			if v.Extensions == nil {
				v.Extensions = map[string]interface{}{}
			}
			v.Extensions[k] = vv
		}
	}
	return nil
}

// InfoProps the properties for an info definition
type InfoProps struct {
	Description    string       `json:"description,omitempty"`
	Title          string       `json:"title,omitempty"`
	TermsOfService string       `json:"termsOfService,omitempty"`
	Contact        *ContactInfo `json:"contact,omitempty"`
	License        *License     `json:"license,omitempty"`
	Version        string       `json:"version,omitempty"`
}

// Info object provides metadata about the API.
// The metadata can be used by the clients if needed, and can be presented in the Swagger-UI for convenience.
//
// For more information: http://goo.gl/8us55a#infoObject
type Info struct {
	VendorExtensible
	InfoProps
}

// MarshalJSON marshal this to JSON
func (i Info) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(i.InfoProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(i.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// UnmarshalJSON marshal this from JSON
func (i *Info) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, i)
	}

	if err := json.Unmarshal(data, &i.InfoProps); err != nil {
		return err
	}
	return json.Unmarshal(data, &i.VendorExtensible)
}

func (i *Info) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		Extensions
		InfoProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	x.Extensions.sanitize()
	if len(x.Extensions) == 0 {
		x.Extensions = nil
	}
	i.VendorExtensible.Extensions = x.Extensions
	i.InfoProps = x.InfoProps
	return nil
}
