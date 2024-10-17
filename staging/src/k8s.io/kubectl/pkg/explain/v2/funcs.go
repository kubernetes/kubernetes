/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v2

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"text/template"

	"github.com/go-openapi/jsonreference"
	"k8s.io/kubectl/pkg/util/term"
)

type explainError string

func (e explainError) Error() string {
	return string(e)
}

func WithBuiltinTemplateFuncs(tmpl *template.Template) *template.Template {
	return tmpl.Funcs(map[string]interface{}{
		"throw": func(e string, args ...any) (string, error) {
			errString := fmt.Sprintf(e, args...)
			return "", explainError(errString)
		},
		"toJson": func(obj any) (string, error) {
			res, err := json.Marshal(obj)
			return string(res), err
		},
		"toPrettyJson": func(obj any) (string, error) {
			res, err := json.MarshalIndent(obj, "", "    ")
			if err != nil {
				return "", err
			}
			return string(res), err
		},
		"fail": func(message string) (string, error) {
			return "", errors.New(message)
		},
		"wrap": func(l int, s string) (string, error) {
			buf := bytes.NewBuffer(nil)
			writer := term.NewWordWrapWriter(buf, uint(l))
			_, err := writer.Write([]byte(s))
			if err != nil {
				return "", err
			}
			return buf.String(), nil
		},
		"split": func(s string, sep string) []string {
			return strings.Split(s, sep)
		},
		"join": func(sep string, strs ...string) string {
			return strings.Join(strs, sep)
		},
		"include": func(name string, data interface{}) (string, error) {
			buf := bytes.NewBuffer(nil)
			if err := tmpl.ExecuteTemplate(buf, name, data); err != nil {
				return "", err
			}
			return buf.String(), nil
		},
		"ternary": func(a, b any, condition bool) any {
			if condition {
				return a
			}
			return b
		},
		"first": func(list any) (any, error) {
			if list == nil {
				return nil, errors.New("list is empty")
			}

			tp := reflect.TypeOf(list).Kind()
			switch tp {
			case reflect.Slice, reflect.Array:
				l2 := reflect.ValueOf(list)

				l := l2.Len()
				if l == 0 {
					return nil, errors.New("list is empty")
				}

				return l2.Index(0).Interface(), nil
			default:
				return nil, fmt.Errorf("first cannot be used on type: %T", list)
			}
		},
		"last": func(list any) (any, error) {
			if list == nil {
				return nil, errors.New("list is empty")
			}

			tp := reflect.TypeOf(list).Kind()
			switch tp {
			case reflect.Slice, reflect.Array:
				l2 := reflect.ValueOf(list)

				l := l2.Len()
				if l == 0 {
					return nil, errors.New("list is empty")
				}

				return l2.Index(l - 1).Interface(), nil
			default:
				return nil, fmt.Errorf("last cannot be used on type: %T", list)
			}
		},
		"indent": func(amount int, spaceString, str string) string {
			pad := strings.Repeat(spaceString, amount)
			return pad + strings.Replace(str, "\n", "\n"+pad, -1)
		},
		"dict": func(keysAndValues ...any) (map[string]any, error) {
			if len(keysAndValues)%2 != 0 {
				return nil, errors.New("expected even # of arguments")
			}

			res := map[string]any{}
			for i := 0; i+1 < len(keysAndValues); i = i + 2 {
				if key, ok := keysAndValues[i].(string); ok {
					res[key] = keysAndValues[i+1]
				} else {
					return nil, fmt.Errorf("key of type %T is not a string as expected", key)
				}
			}

			return res, nil
		},
		"contains": func(list any, value any) bool {
			if list == nil {
				return false
			}

			val := reflect.ValueOf(list)
			switch val.Kind() {
			case reflect.Array:
			case reflect.Slice:
				for i := 0; i < val.Len(); i++ {
					cur := val.Index(i)
					if cur.CanInterface() && reflect.DeepEqual(cur.Interface(), value) {
						return true
					}
				}
				return false
			default:
				return false
			}
			return false
		},
		"set": func(dict map[string]any, keysAndValues ...any) (any, error) {
			if len(keysAndValues)%2 != 0 {
				return nil, errors.New("expected even number of arguments")
			}

			copyDict := make(map[string]any, len(dict))
			for k, v := range dict {
				copyDict[k] = v
			}

			for i := 0; i < len(keysAndValues); i += 2 {
				key, ok := keysAndValues[i].(string)
				if !ok {
					return nil, errors.New("keys must be strings")
				}

				copyDict[key] = keysAndValues[i+1]
			}

			return copyDict, nil
		},
		"list": func(values ...any) ([]any, error) {
			return values, nil
		},
		"add": func(value, operand int) int {
			return value + operand
		},
		"sub": func(value, operand int) int {
			return value - operand
		},
		"mul": func(value, operand int) int {
			return value * operand
		},
		"resolveRef": func(refAny any, document map[string]any) map[string]any {
			refString, ok := refAny.(string)
			if !ok {
				// if passed nil, or wrong type just treat the same
				// way as unresolved reference (makes for easier templates)
				return nil
			}

			// Resolve field path encoded by the ref
			ref, err := jsonreference.New(refString)
			if err != nil {
				// Unrecognized ref format.
				return nil
			}

			if !ref.HasFragmentOnly {
				// Downloading is not supported. Treat as not found
				return nil
			}

			fragment := ref.GetURL().Fragment
			components := strings.Split(fragment, "/")
			cur := document

			for _, k := range components {
				if len(k) == 0 {
					// first component is usually empty (#/components/) , etc
					continue
				}

				next, ok := cur[k].(map[string]any)
				if !ok {
					return nil
				}

				cur = next
			}
			return cur
		},
	})
}
