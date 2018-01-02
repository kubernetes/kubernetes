/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package mo

import (
	"reflect"
	"testing"
)

func TestLoadAll(*testing.T) {
	for _, typ := range t {
		newTypeInfo(typ)
	}
}

// The virtual machine managed object has about 500 nested properties.
// It's likely to be indicative of the function's performance in general.
func BenchmarkLoadVirtualMachine(b *testing.B) {
	vmtyp := reflect.TypeOf((*VirtualMachine)(nil)).Elem()
	for i := 0; i < b.N; i++ {
		newTypeInfo(vmtyp)
	}
}
