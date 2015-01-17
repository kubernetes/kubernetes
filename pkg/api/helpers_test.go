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

package api

import (
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"

	"speter.net/go/exp/math/dec/inf"
)

func TestConversionError(t *testing.T) {
	var i int
	var s string
	i = 3
	s = "foo"
	c := ConversionError{
		In: &i, Out: &s,
		Message: "Can't make x into y, silly",
	}
	var e error
	e = &c // ensure it implements error
	msg := e.Error()
	t.Logf("Message is %v", msg)
	for _, part := range []string{"3", "int", "string", "Can't"} {
		if !strings.Contains(msg, part) {
			t.Errorf("didn't find %v", part)
		}
	}
}

func TestSemantic(t *testing.T) {
	table := []struct {
		a, b        interface{}
		shouldEqual bool
	}{
		{resource.MustParse("0"), resource.Quantity{}, true},
		{resource.Quantity{}, resource.MustParse("0"), true},
		{resource.Quantity{}, resource.MustParse("1m"), false},
		{
			resource.Quantity{inf.NewDec(5, 0), resource.BinarySI},
			resource.Quantity{inf.NewDec(5, 0), resource.DecimalSI},
			true,
		},
		{resource.MustParse("2m"), resource.MustParse("1m"), false},
	}

	for index, item := range table {
		if e, a := item.shouldEqual, Semantic.DeepEqual(item.a, item.b); e != a {
			t.Errorf("expected %v, got %v.", index, e, a)
		}
	}
}
