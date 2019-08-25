/*
Copyright 2018 The Kubernetes Authors.

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

package transformers

import (
	"fmt"
	"log"
	"strings"
)

type mutateFunc func(interface{}) (interface{}, error)

func MutateField(
	m map[string]interface{},
	pathToField []string,
	createIfNotPresent bool,
	fns ...mutateFunc) error {
	if len(pathToField) == 0 {
		return nil
	}

	firstPathSegment, isArray := getFirstPathSegment(pathToField)

	_, found := m[firstPathSegment]
	if !found {
		if !createIfNotPresent || isArray {
			return nil
		}
		m[firstPathSegment] = map[string]interface{}{}
	}

	if len(pathToField) == 1 {
		var err error
		for _, fn := range fns {
			m[firstPathSegment], err = fn(m[firstPathSegment])
			if err != nil {
				return err
			}
		}
		return nil
	}

	v := m[firstPathSegment]
	newPathToField := pathToField[1:]
	switch typedV := v.(type) {
	case nil:
		log.Printf(
			"nil value at `%s` ignored in mutation attempt",
			strings.Join(pathToField, "."))
		return nil
	case map[string]interface{}:
		return MutateField(typedV, newPathToField, createIfNotPresent, fns...)
	case []interface{}:
		for i := range typedV {
			item := typedV[i]
			typedItem, ok := item.(map[string]interface{})
			if !ok {
				return fmt.Errorf("%#v is expected to be %T", item, typedItem)
			}
			err := MutateField(typedItem, newPathToField, createIfNotPresent, fns...)
			if err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("%#v is not expected to be a primitive type", typedV)
	}
}

func getFirstPathSegment(pathToField []string) (string, bool) {
	if strings.HasSuffix(pathToField[0], "[]") {
		return pathToField[0][:len(pathToField[0])-2], true
	}
	return pathToField[0], false
}
