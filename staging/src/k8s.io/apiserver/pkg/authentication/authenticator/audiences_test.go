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

package authenticator

import (
	"reflect"
	"testing"
)

func TestIntersect(t *testing.T) {
	cs := []struct {
		auds, tauds Audiences
		expected    Audiences
	}{
		{
			auds:     nil,
			tauds:    nil,
			expected: Audiences{},
		},
		{
			auds:     nil,
			tauds:    Audiences{"foo"},
			expected: Audiences{},
		},
		{
			auds:     Audiences{},
			tauds:    Audiences{},
			expected: Audiences{},
		},
		{
			auds:     Audiences{"foo"},
			tauds:    Audiences{},
			expected: Audiences{},
		},
		{
			auds:     Audiences{"foo"},
			tauds:    Audiences{"foo"},
			expected: Audiences{"foo"},
		},
		{
			auds:     Audiences{"foo", "bar"},
			tauds:    Audiences{"foo", "bar"},
			expected: Audiences{"foo", "bar"},
		},
		{
			auds:     Audiences{"foo", "bar"},
			tauds:    Audiences{"foo", "wat"},
			expected: Audiences{"foo"},
		},
		{
			auds:     Audiences{"foo", "bar"},
			tauds:    Audiences{"pls", "wat"},
			expected: Audiences{},
		},
	}
	for _, c := range cs {
		t.Run("auds", func(t *testing.T) {
			if got, want := c.auds.Intersect(c.tauds), c.expected; !reflect.DeepEqual(got, want) {
				t.Errorf("unexpected intersection.\ngot:\t%v\nwant:\t%v", got, want)
			}
		})
	}
}
