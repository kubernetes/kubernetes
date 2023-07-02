/*
Copyright 2023 The Kubernetes Authors.

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

package cache

import (
	"math/rand"
	"strings"
	"testing"
)

func TestObjectNames(t *testing.T) {
	chars := "abcdefghi/"
	for count := 1; count <= 100; count++ {
		var encodedB strings.Builder
		for index := 0; index < 10; index++ {
			encodedB.WriteByte(chars[rand.Intn(len(chars))])
		}
		encodedS := encodedB.String()
		parts := strings.Split(encodedS, "/")
		on, err := ParseObjectName(encodedS)
		expectError := len(parts) > 2
		if expectError != (err != nil) {
			t.Errorf("Wrong error; expected=%v, got=%v", expectError, err)
		}
		if expectError || err != nil {
			continue
		}
		var expectedObjectName ObjectName
		if len(parts) == 2 {
			expectedObjectName = ObjectName{Namespace: parts[0], Name: parts[1]}
		} else {
			expectedObjectName = ObjectName{Name: encodedS}
		}
		if on != expectedObjectName {
			t.Errorf("Parse failed, expected=%+v, got=%+v", expectedObjectName, on)
		}
		recoded := on.String()
		if encodedS[0] == '/' {
			recoded = "/" + recoded
		}
		if encodedS != recoded {
			t.Errorf("Parse().String() was not identity, original=%q, final=%q", encodedS, recoded)
		}
	}
}
