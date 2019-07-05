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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
)

func newFields() metav1.Fields {
	return metav1.Fields{Map: map[string]metav1.Fields{}}
}

func fieldsSet(f metav1.Fields, path fieldpath.Path, set *fieldpath.Set) error {
	if len(f.Map) == 0 {
		set.Insert(path)
	}
	for k := range f.Map {
		if k == "." {
			set.Insert(path)
			continue
		}
		pe, err := NewPathElement(k)
		if err != nil {
			return err
		}
		path = append(path, pe)
		err = fieldsSet(f.Map[k], path, set)
		if err != nil {
			return err
		}
		path = path[:len(path)-1]
	}
	return nil
}

// FieldsToSet creates a set paths from an input trie of fields
func FieldsToSet(f metav1.Fields) (fieldpath.Set, error) {
	set := fieldpath.Set{}
	return set, fieldsSet(f, fieldpath.Path{}, &set)
}

func removeUselessDots(f metav1.Fields) metav1.Fields {
	if _, ok := f.Map["."]; ok && len(f.Map) == 1 {
		delete(f.Map, ".")
		return f
	}
	for k, tf := range f.Map {
		f.Map[k] = removeUselessDots(tf)
	}
	return f
}

// SetToFields creates a trie of fields from an input set of paths
func SetToFields(s fieldpath.Set) (metav1.Fields, error) {
	var err error
	f := newFields()
	s.Iterate(func(path fieldpath.Path) {
		if err != nil {
			return
		}
		tf := f
		for _, pe := range path {
			var str string
			str, err = PathElementString(pe)
			if err != nil {
				break
			}
			if _, ok := tf.Map[str]; ok {
				tf = tf.Map[str]
			} else {
				tf.Map[str] = newFields()
				tf = tf.Map[str]
			}
		}
		tf.Map["."] = newFields()
	})
	f = removeUselessDots(f)
	return f, err
}
