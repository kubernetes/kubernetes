/*
Copyright 2021 The Kubernetes Authors.

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

package storage

import (
	"testing"
)

func Test_metaTransaction(t *testing.T) {
	const initial = 10
	var temp int

	tests := []struct {
		name  string
		mt    metaTransaction
		start int
		want  int
	}{{
		name: "commit and revert match",
		mt: metaTransaction{
			callbackTransaction{
				commit: func() {
					temp = temp + 1
				},
				revert: func() {
					temp = temp - 1
				},
			},
		},
		want: 10,
	}, {
		name: "commit and revert match multiple times",
		mt: metaTransaction{
			callbackTransaction{
				commit: func() {
					temp = temp + 1
				},
				revert: func() {
					temp = temp - 1
				},
			},
			callbackTransaction{
				commit: func() {
					temp = temp + 2
				},
				revert: func() {
					temp = temp - 2
				},
			},
			callbackTransaction{
				commit: func() {
					temp = temp + 3
				},
				revert: func() {
					temp = temp - 3
				},
			},
		},
		want: 10,
	}, {
		name: "missing revert",
		mt: metaTransaction{
			callbackTransaction{
				commit: func() {
					temp = temp + 1
				},
				revert: func() {
					temp = temp - 1
				},
			},
			callbackTransaction{
				commit: func() {
					temp = temp + 2
				},
			},
			callbackTransaction{
				commit: func() {
					temp = temp + 3
				},
				revert: func() {
					temp = temp - 3
				},
			},
		},
		want: 12,
	}, {
		name: "missing commit",
		mt: metaTransaction{
			callbackTransaction{
				commit: func() {
					temp = temp + 1
				},
				revert: func() {
					temp = temp - 1
				},
			},
			callbackTransaction{
				revert: func() {
					temp = temp - 2
				},
			},
			callbackTransaction{
				commit: func() {
					temp = temp + 3
				},
				revert: func() {
					temp = temp - 3
				},
			},
		},
		want: 8,
	}, {
		name: "commit and revert match multiple but different order",
		mt: metaTransaction{
			callbackTransaction{
				commit: func() {
					temp = temp + 1
				},
				revert: func() {
					temp = temp - 2
				},
			},
			callbackTransaction{
				commit: func() {
					temp = temp + 2
				},
				revert: func() {
					temp = temp - 1
				},
			},
			callbackTransaction{
				commit: func() {
					temp = temp + 3
				},
				revert: func() {
					temp = temp - 3
				},
			},
		},
		want: 10,
	}}
	t.Parallel()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			temp = initial
			tt.mt.Commit()
			tt.mt.Revert()
			if temp != tt.want {
				t.Fatalf("expected %d got %d", tt.want, temp)
			}
		})
	}
}
