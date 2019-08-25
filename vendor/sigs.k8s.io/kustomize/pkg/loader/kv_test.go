// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package loader

import (
	"reflect"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/kustomize/pkg/validators"
)

func TestKeyValuesFromLines(t *testing.T) {
	tests := []struct {
		desc          string
		content       string
		expectedPairs []types.Pair
		expectedErr   bool
	}{
		{
			desc: "valid kv content parse",
			content: `
		k1=v1
		k2=v2
		`,
			expectedPairs: []types.Pair{
				{Key: "k1", Value: "v1"},
				{Key: "k2", Value: "v2"},
			},
			expectedErr: false,
		},
		{
			desc: "content with comments",
			content: `
		k1=v1
		#k2=v2
		`,
			expectedPairs: []types.Pair{
				{Key: "k1", Value: "v1"},
			},
			expectedErr: false,
		},
		// TODO: add negative testcases
	}

	l := NewFileLoaderAtRoot(
		validators.MakeFakeValidator(), fs.MakeFakeFS())
	for _, test := range tests {
		pairs, err := l.keyValuesFromLines([]byte(test.content))
		if test.expectedErr && err == nil {
			t.Fatalf("%s should not return error", test.desc)
		}
		if !reflect.DeepEqual(pairs, test.expectedPairs) {
			t.Errorf("%s should succeed, got:%v exptected:%v", test.desc, pairs, test.expectedPairs)
		}
	}
}

func TestKeyValuesFromFileSources(t *testing.T) {
	tests := []struct {
		description string
		sources     []string
		expected    []types.Pair
	}{
		{
			description: "create kvs from file sources",
			sources:     []string{"files/app-init.ini"},
			expected: []types.Pair{
				{
					Key:   "app-init.ini",
					Value: "FOO=bar",
				},
			},
		},
	}

	fSys := fs.MakeFakeFS()
	fSys.WriteFile("/files/app-init.ini", []byte("FOO=bar"))
	l := NewFileLoaderAtRoot(validators.MakeFakeValidator(), fSys)
	for _, tc := range tests {
		kvs, err := l.keyValuesFromFileSources(tc.sources)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(kvs, tc.expected) {
			t.Fatalf("in testcase: %q updated:\n%#v\ndoesn't match expected:\n%#v\n", tc.description, kvs, tc.expected)
		}
	}
}
