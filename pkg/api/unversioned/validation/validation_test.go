/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/util/validation/field"
)

func TestValidateLabels(t *testing.T) {
	successCases := []map[string]string{
		{"simple": "bar"},
		{"now-with-dashes": "bar"},
		{"1-starts-with-num": "bar"},
		{"1234": "bar"},
		{"simple/simple": "bar"},
		{"now-with-dashes/simple": "bar"},
		{"now-with-dashes/now-with-dashes": "bar"},
		{"now.with.dots/simple": "bar"},
		{"now-with.dashes-and.dots/simple": "bar"},
		{"1-num.2-num/3-num": "bar"},
		{"1234/5678": "bar"},
		{"1.2.3.4/5678": "bar"},
		{"UpperCaseAreOK123": "bar"},
		{"goodvalue": "123_-.BaR"},
	}
	for i := range successCases {
		errs := ValidateLabels(successCases[i], field.NewPath("field"))
		if len(errs) != 0 {
			t.Errorf("case[%d] expected success, got %#v", i, errs)
		}
	}

	labelNameErrorCases := []map[string]string{
		{"nospecialchars^=@": "bar"},
		{"cantendwithadash-": "bar"},
		{"only/one/slash": "bar"},
		{strings.Repeat("a", 254): "bar"},
	}
	for i := range labelNameErrorCases {
		errs := ValidateLabels(labelNameErrorCases[i], field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		} else {
			detail := errs[0].Detail
			if detail != qualifiedNameErrorMsg {
				t.Errorf("error detail %s should be equal %s", detail, qualifiedNameErrorMsg)
			}
		}
	}

	labelValueErrorCases := []map[string]string{
		{"toolongvalue": strings.Repeat("a", 64)},
		{"backslashesinvalue": "some\\bad\\value"},
		{"nocommasallowed": "bad,value"},
		{"strangecharsinvalue": "?#$notsogood"},
	}
	for i := range labelValueErrorCases {
		errs := ValidateLabels(labelValueErrorCases[i], field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		} else {
			detail := errs[0].Detail
			if detail != labelValueErrorMsg {
				t.Errorf("error detail %s should be equal %s", detail, labelValueErrorMsg)
			}
		}
	}
}
