// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transformers

import (
	"reflect"
	"testing"

	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resmaptest"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

func TestVarRef(t *testing.T) {
	type given struct {
		varMap map[string]interface{}
		fs     []config.FieldSpec
		res    resmap.ResMap
	}
	type expected struct {
		res    resmap.ResMap
		unused []string
	}
	testCases := []struct {
		description string
		given       given
		expected    expected
	}{
		{
			description: "var replacement in map[string]",
			given: given{
				varMap: map[string]interface{}{
					"FOO": "replacementForFoo",
					"BAR": "replacementForBar",
					"BAZ": int64(5),
					"BOO": true,
				},
				fs: []config.FieldSpec{
					{Gvk: gvk.Gvk{Version: "v1", Kind: "ConfigMap"}, Path: "data"},
				},
				res: resmaptest_test.NewRmBuilder(t, rf).
					Add(map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "ConfigMap",
						"metadata": map[string]interface{}{
							"name": "cm1",
						},
						"data": map[string]interface{}{
							"item1": "$(FOO)",
							"item2": "bla",
							"item3": "$(BAZ)",
							"item4": "$(BAZ)+$(BAZ)",
							"item5": "$(BOO)",
							"item6": "if $(BOO)",
						},
					}).ResMap(),
			},
			expected: expected{
				res: resmaptest_test.NewRmBuilder(t, rf).
					Add(map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "ConfigMap",
						"metadata": map[string]interface{}{
							"name": "cm1",
						},
						"data": map[string]interface{}{
							"item1": "replacementForFoo",
							"item2": "bla",
							"item3": int64(5),
							"item4": "5+5",
							"item5": true,
							"item6": "if true",
						}}).ResMap(),
				unused: []string{"BAR"},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// arrange
			tr := NewRefVarTransformer(tc.given.varMap, tc.given.fs)

			// act
			err := tr.Transform(tc.given.res)

			// assert
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			a, e := tc.given.res, tc.expected.res
			if !reflect.DeepEqual(a, e) {
				err = e.ErrorIfNotEqualLists(a)
				t.Fatalf("actual doesn't match expected: \nACTUAL:\n%v\nEXPECTED:\n%v\nERR: %v", a, e, err)
			}

		})
	}
}
