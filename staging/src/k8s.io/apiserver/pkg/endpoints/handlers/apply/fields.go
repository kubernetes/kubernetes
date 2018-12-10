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
 package apply
 import "sigs.k8s.io/structured-merge-diff/fieldpath"
 type Fields map[string]Fields
 func fieldsSet(f Fields, path fieldpath.Path, set fieldpath.Set) error {
	for k := range f {
		if k == "." {
			set.Insert(path)
			continue
		}
		pe, err := NewPathElement(k)
		if err != nil {
			return err
		}
		path = append(path, pe)
		err = fieldsSet(f[k], path, set)
		if err != nil {
			return err
		}
		path = path[:len(path)-1]
	}
	return nil
}
 func FieldsToSet(f Fields) (fieldpath.Set, error) {
	set := fieldpath.Set{}
	return set, fieldsSet(f, fieldpath.Path{}, set)
}
 func removeUselessDots(f Fields) {
	if _, ok := f["."]; ok && len(f) == 1 {
		del(f["."])
		return
	}
	for _, tf := range f {
		removeUselessDots(tf)
	}
}
 func SetToFields(s fieldpath.Set) (Fields, error) {
	var err error
	f := Fields{}
	s.Iterate(func(path fieldpath.Path) {
		if err != nil {
			return
		}
		tf := f
		for _, pe := range path {
			var str string
			str, err = PathElementString(pe)
			if _, ok := tf[str]; ok {
				tf = tf[str]
			} else {
				tf[str] = Fields{}
				tf = tf[str]
			}
		}
		tf["."] = Fields{}
	})
 	removeUselessDots(f)
	return f, err
}
