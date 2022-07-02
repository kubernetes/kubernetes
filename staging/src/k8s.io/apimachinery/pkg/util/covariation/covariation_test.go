/*
Copyright 2022 The Kubernetes Authors.

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

package covariation

import (
	"testing"
)

type a struct{}

type a1 interface {
	back1() int
}

type a2 interface {
	a1
	back2() int
}

type a3 interface {
	back3() int
}

func (a *a) back1() int { return 1 }
func (a *a) back2() int { return 2 }

func TestCovariant(t *testing.T) {
	type pair struct {
		src interface{}
		to  interface{}
	}
	tests := []struct {
		name     string
		args     pair
		verifyFn func(v interface{}) int
		want     int
		wantErr  bool
	}{
		{
			name:     "a1 to a2",
			args:     pair{src: (a1)(&a{}), to: (*a2)(nil)},
			verifyFn: func(v interface{}) int { return v.(a2).back2() },
			want:     2,
		},
		{
			name:     "a2 to a1",
			args:     pair{src: (a2)(&a{}), to: (*a1)(nil)},
			verifyFn: func(v interface{}) int { return v.(a1).back1() },
			want:     1,
		},
		{
			name:    "a1 to a3",
			args:    pair{src: (a1)(&a{}), to: (*a3)(nil)},
			wantErr: true,
		},
		{
			name:    "a1 to nil",
			args:    pair{src: (a1)(&a{}), to: nil},
			wantErr: true,
		},
		{
			name:    "nil to a1",
			args:    pair{src: nil, to: (a1)(&a{})},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Covariant(tt.args.src, tt.args.to)
			if (err != nil) != tt.wantErr {
				t.Errorf("expected error: %v, but got error: %v", tt.wantErr, err)
				return
			}
			if tt.verifyFn != nil && tt.verifyFn(got) != tt.want {
				t.Errorf("expected back: %v, but got back: %v", tt.want, tt.verifyFn(got))
			}
		})
	}
}
