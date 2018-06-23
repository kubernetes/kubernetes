/*
Copyright 2016 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
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

	namePartErrMsg := "name part must consist of"
	nameErrMsg := "a qualified name must consist of"
	labelErrMsg := "a valid label must be an empty string or consist of"
	maxLengthErrMsg := "must be no more than"

	labelNameErrorCases := []struct {
		labels map[string]string
		expect string
	}{
		{map[string]string{"nospecialchars^=@": "bar"}, namePartErrMsg},
		{map[string]string{"cantendwithadash-": "bar"}, namePartErrMsg},
		{map[string]string{"only/one/slash": "bar"}, nameErrMsg},
		{map[string]string{strings.Repeat("a", 254): "bar"}, maxLengthErrMsg},
	}
	for i := range labelNameErrorCases {
		errs := ValidateLabels(labelNameErrorCases[i].labels, field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d]: expected failure", i)
		} else {
			if !strings.Contains(errs[0].Detail, labelNameErrorCases[i].expect) {
				t.Errorf("case[%d]: error details do not include %q: %q", i, labelNameErrorCases[i].expect, errs[0].Detail)
			}
		}
	}

	labelValueErrorCases := []struct {
		labels map[string]string
		expect string
	}{
		{map[string]string{"toolongvalue": strings.Repeat("a", 64)}, maxLengthErrMsg},
		{map[string]string{"backslashesinvalue": "some\\bad\\value"}, labelErrMsg},
		{map[string]string{"nocommasallowed": "bad,value"}, labelErrMsg},
		{map[string]string{"strangecharsinvalue": "?#$notsogood"}, labelErrMsg},
	}
	for i := range labelValueErrorCases {
		errs := ValidateLabels(labelValueErrorCases[i].labels, field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d]: expected failure", i)
		} else {
			if !strings.Contains(errs[0].Detail, labelValueErrorCases[i].expect) {
				t.Errorf("case[%d]: error details do not include %q: %q", i, labelValueErrorCases[i].expect, errs[0].Detail)
			}
		}
	}
}

func TestValidateLabelSelector(t *testing.T) {
	successCases := []struct {
		name     string
		selector metav1.LabelSelector
	}{
		{
			name: "test when two keys and values in MatchLabels while one key and two values in MatchExpressions",
			selector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id":  "app2",
					"id2": "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
				},
			},
		},
		{
			name: "test when one key and two values in MatchExpressions but no MatchLabels",
			selector: metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "key2", Operator: "In", Values: []string{"value1", "value2"}},
				},
			},
		},
		{
			name: "test when one key and one value in MatchLabels but no MatchExpressions",
			selector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id": "app2",
				},
			},
		},
		{
			name: "test when MatchLabels and MatchExpressions exist while Operator NotIn",
			selector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id":  "app2",
					"id2": "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "NotIn", Values: []string{"bar1", "bar2"}},
				},
			},
		},
		{
			name: "test when MatchLabels and MatchExpressions exist while Operator Exists",
			selector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id": "app2",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "Exists"},
				},
			},
		},
		{
			name: "test when MatchLabels and MatchExpressions exist while Operator DoesNotExist",
			selector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id": "app2",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "DoesNotExist"},
				},
			},
		},
	}
	for i := range successCases {
		errs := ValidateLabelSelector(&successCases[i].selector, field.NewPath("field"))
		if len(errs) != 0 {
			t.Errorf("successCases: case[%d]:%v expected success, got %#v", i, successCases[i].name, errs)
		}
	}

	namePartErrMsg := "name part must consist of"
	nameErrMsg := "a qualified name must consist of"
	labelErrMsg := "a valid label must be an empty string or consist of"
	maxLengthErrMsg := "must be no more than"

	matchLabelErrorCases := []struct {
		labelselector metav1.LabelSelector
		expect        string
	}{
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"ids^=@": "app2",
					"id2":    "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
				},
			},
			expect: namePartErrMsg,
		},
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"only/one/slas": "app2",
					"id2":           "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
				},
			},
			expect: nameErrMsg,
		},
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"only": "app2",
					"good": "app3=@",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
				},
			},
			expect: labelErrMsg,
		},
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"only": "app2",
					"good": strings.Repeat("a", 64),
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
				},
			},
			expect: maxLengthErrMsg,
		},
	}
	for i := range matchLabelErrorCases {
		errs := ValidateLabelSelector(&matchLabelErrorCases[i].labelselector, field.NewPath("field"))
		if len(errs) == 0 {
			t.Errorf("MatchLabelErrorCases: case[%d]:%v expected failure", i, matchLabelErrorCases[i])
		} else {
			if !strings.Contains(errs[0].Detail, matchLabelErrorCases[i].expect) {
				t.Errorf("MatchLabelErrorCases: case[%d]:%v error details do not include %q: %q", i, matchLabelErrorCases[i], matchLabelErrorCases[i].expect, errs[0].Detail)
			}
		}
	}

	opInOrNotInErrMsg := "must be specified when `operator` is 'In' or 'NotIn'"
	opExistsOrNotExistsErrMsg := "may not be specified when `operator` is 'Exists' or 'DoesNotExist'"
	opNotValidErrMsg := "not a valid selector operator"

	matchExpressionErrorCases := []struct {
		labelselector metav1.LabelSelector
		expect        string
	}{
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"ids": "app2",
					"id2": "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{}},
				},
			},
			expect: opInOrNotInErrMsg,
		},
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"only": "app2",
					"id2":  "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "NotIn", Values: []string{}},
				},
			},
			expect: opInOrNotInErrMsg,
		},
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"only": "app2",
					"good": "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "Exists", Values: []string{"bar1", "bar2"}},
				},
			},
			expect: opExistsOrNotExistsErrMsg,
		},
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"only": "app2",
					"good": "app1",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "DoesNotExist", Values: []string{"bar1", "bar2"}},
				},
			},
			expect: opExistsOrNotExistsErrMsg,
		},
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"ids": "app2",
					"id2": "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "Out", Values: []string{"bar1", "bar2"}},
				},
			},
			expect: opNotValidErrMsg,
		},
		{
			labelselector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"ids": "app2",
					"id2": "app3",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo^=@", Operator: "In", Values: []string{"bar1", "bar2"}},
				},
			},
			expect: namePartErrMsg,
		},
	}
	for i := range matchExpressionErrorCases {
		errs := ValidateLabelSelector(&matchExpressionErrorCases[i].labelselector, field.NewPath("field"))
		if len(errs) == 0 {
			t.Errorf("MatchExpressionErrorCases: case[%d]:%v expected failure", i, matchExpressionErrorCases[i])
		} else {
			if !strings.Contains(errs[0].Detail, matchExpressionErrorCases[i].expect) {
				t.Errorf("MatchExpressionErrorCases: case[%d]:%v error details do not include %q: %q", i, matchExpressionErrorCases[i], matchExpressionErrorCases[i].expect, errs[0].Detail)
			}
		}
	}
}

func TestValidateDeleteOptions(t *testing.T) {
	falseVar := false
	foregroundPolicy := metav1.DeletePropagationForeground
	backgroundPolicy := metav1.DeletePropagationBackground
	orphanPolicy := metav1.DeletePropagationOrphan
	var invaildPolicy metav1.DeletionPropagation = "testinvaild"

	successCases := []metav1.DeleteOptions{
		{OrphanDependents: &falseVar},
		{PropagationPolicy: &orphanPolicy},
		{PropagationPolicy: &foregroundPolicy},
		{PropagationPolicy: &backgroundPolicy},
	}
	for i := range successCases {
		errs := ValidateDeleteOptions(&successCases[i])
		if len(errs) != 0 {
			t.Errorf("successCases:case[%d] expected success, got %#v", i, errs)
		}
	}

	opBothSetErrMsg := "OrphanDependents and DeletionPropagation cannot be both set"
	opNotVaildErrMsg := "DeletionPropagation need to be one of"
	filedCases := []struct {
		deleteoptions metav1.DeleteOptions
		expect        string
	}{
		{
			deleteoptions: metav1.DeleteOptions{OrphanDependents: &falseVar, PropagationPolicy: &orphanPolicy},
			expect:        opBothSetErrMsg,
		},
		{
			deleteoptions: metav1.DeleteOptions{OrphanDependents: &falseVar, PropagationPolicy: &backgroundPolicy},
			expect:        opBothSetErrMsg,
		},
		{
			deleteoptions: metav1.DeleteOptions{OrphanDependents: &falseVar, PropagationPolicy: &foregroundPolicy},
			expect:        opBothSetErrMsg,
		},
		{
			deleteoptions: metav1.DeleteOptions{PropagationPolicy: &invaildPolicy},
			expect:        opNotVaildErrMsg,
		},
	}
	for i := range filedCases {
		errs := ValidateDeleteOptions(&filedCases[i].deleteoptions)
		if len(errs) == 0 {
			t.Errorf("filedCases: case[%d]:%v expected failure", i, filedCases[i])
		} else {
			if !strings.Contains(errs[0].Detail, filedCases[i].expect) {
				t.Errorf("filedCases: case[%d]:%v error details do not include %q: %q", i, filedCases[i], filedCases[i].expect, errs[0].Detail)
			}
		}
	}
}
