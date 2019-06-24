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

package internal

import (
	"encoding/json"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
)

type fieldsList []string

// fieldsToSet creates a set paths from an input trie of fields
func fieldsToSet(f fieldsList) (*fieldpath.SetAsList, error) {
	elements := make([]fieldpath.LeveledElement, len(f))

	for i, s := range f {
		if err := NewLeveledElement(s, &elements[i]); err != nil {
			return nil, err
		}
	}

	return fieldpath.NewSetAsListFromList(elements), nil
}

func setToFields(set *fieldpath.SetAsList) (fieldsList, error) {
	l := make(fieldsList, len(set.List()))
	for i, lpe := range set.List() {
		s, err := LeveledElementString(&lpe)
		if err != nil {
			return nil, err
		}
		l[i] = s
	}
	return l, nil
}

var EmptyFields metav1.Fields = func() metav1.Fields {
	fm, err := setToFields(fieldpath.NewSetAsList())
	if err != nil {
		panic("should never happen")
	}
	f, err := json.Marshal(fm)
	if err != nil {
		panic("should never happen")
	}
	return metav1.Fields(f)
}()

// FieldsToSet creates a set paths from an input trie of fields
func FieldsToSet(f metav1.Fields) (*fieldpath.SetAsList, error) {
	fl := fieldsList{}
	err := json.Unmarshal([]byte(f), &fl)
	if err != nil {
		return nil, err
	}
	return fieldsToSet(fl)
}

// SetToFields creates a trie of fields from an input set of paths
func SetToFields(s *fieldpath.SetAsList) (metav1.Fields, error) {
	fm, err := setToFields(s)
	if err != nil {
		return EmptyFields, err
	}
	f, err := json.Marshal(fm)
	if err != nil {
		return EmptyFields, err
	}
	return metav1.Fields(f), nil
}
