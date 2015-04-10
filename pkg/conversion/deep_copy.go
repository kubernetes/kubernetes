/*
Copyright 2015 Google Inc. All rights reserved.

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

package conversion

import (
	"bytes"
	"encoding/gob"
	"reflect"
)

// DeepCopy makes a deep copy of source. Won't work for any private fields!
// For nil slices, will return 0-length slices. These are equivilent in
// basically every way except for the way that reflect.DeepEqual checks.
func DeepCopy(source interface{}) (interface{}, error) {
	v := reflect.New(reflect.TypeOf(source))

	buff := &bytes.Buffer{}
	enc := gob.NewEncoder(buff)
	dec := gob.NewDecoder(buff)
	err := enc.Encode(source)
	if err != nil {
		return nil, err
	}
	err = dec.Decode(v.Interface())
	if err != nil {
		return nil, err
	}
	return v.Elem().Interface(), nil
}
