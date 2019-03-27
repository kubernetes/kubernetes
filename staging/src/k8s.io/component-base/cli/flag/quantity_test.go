/*
Copyright 2019 The Kubernetes Authors.

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

package flag

import (
	"fmt"
	"testing"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestQuantityFlagValue_Set(t *testing.T) {
	q := new(resource.Quantity)
	qf := NewQuantityFlagValue(q)
	qf.Set("1Ki")
	if e, a := "1Ki", qf.String(); e != a {
		t.Errorf("Unexpected result: want %v got %v", e, a)
	}
	if e, a := int64(1<<10), q.Value(); e != a {
		t.Errorf("Unexpected quantity value: want %v got %v", e, a)
	}
}

func TestQuantityFlagValue_Type(t *testing.T) {
	qfv := NewQuantityFlagValue(nil)
	if e, a := "quantity", qfv.Type(); e != a {
		t.Errorf("Unexpected type: want %v got %v", e, a)
	}
}

func ExampleQuantityFlag() {
	flagValue := QuantityFlag("amount", "500m", "how much is needed")
	pflag.Set("amount", "750m")
	fmt.Println(flagValue.MilliValue())
	// Output: 750
}

func ExampleNewQuantityFlagValue() {
	var myQuantity resource.Quantity
	fs := pflag.NewFlagSet("test", pflag.ExitOnError)
	fs.Var(NewQuantityFlagValue(&myQuantity), "amount", "how much is needed")
	fs.Set("amount", "5Mi")
	fmt.Println(myQuantity.Value())
	// Output: 5242880
}
