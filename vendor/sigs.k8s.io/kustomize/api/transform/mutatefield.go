// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transform

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
