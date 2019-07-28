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

package mount

import (
	"reflect"
	"testing"
)

func TestIsBind(t *testing.T) {
	tests := []struct {
		mountOption         []string
		isBind              bool
		expectedBindOpts    []string
		expectedRemountOpts []string
	}{
		{
			[]string{"vers=2", "ro", "_netdev"},
			false,
			[]string{},
			[]string{},
		},
		{

			[]string{"bind", "vers=2", "ro", "_netdev"},
			true,
			[]string{"bind", "_netdev"},
			[]string{"bind", "remount", "vers=2", "ro", "_netdev"},
		},
	}
	for _, test := range tests {
		bind, bindOpts, bindRemountOpts := IsBind(test.mountOption)
		if bind != test.isBind {
			t.Errorf("Expected bind to be %v but got %v", test.isBind, bind)
		}
		if test.isBind {
			if !reflect.DeepEqual(test.expectedBindOpts, bindOpts) {
				t.Errorf("Expected bind mount options to be %+v got %+v", test.expectedBindOpts, bindOpts)
			}
			if !reflect.DeepEqual(test.expectedRemountOpts, bindRemountOpts) {
				t.Errorf("Expected remount options to be %+v got %+v", test.expectedRemountOpts, bindRemountOpts)
			}
		}

	}
}
