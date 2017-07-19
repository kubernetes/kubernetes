/*
Copyright 2017 The Kubernetes Authors.

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

package executor

import (
	"context"
	"fmt"
	"io"
	"os"
	"reflect"
	"testing"
)

func TestSort(t *testing.T) {
	cs := []struct {
		units     []*Unit
		expected  []string
		expectErr bool
	}{
		{
			units: []*Unit{
				&Unit{
					Name: "a",
				},
			},
			expected: []string{"a"},
		},
		{
			units: []*Unit{
				&Unit{
					Name: "a",
				},
				&Unit{
					Name: "b",
				},
				&Unit{
					Name: "c",
				},
			},
			expected: []string{"a", "b", "c"},
		},
		{
			units: []*Unit{
				&Unit{
					Name: "c",
				},
				&Unit{
					Name: "b",
				},
				&Unit{
					Name: "a",
				},
			},
			expected: []string{"a", "b", "c"},
		},
		{
			units: []*Unit{
				&Unit{
					Name: "a",
				},
				&Unit{
					Name: "b",
					Deps: []string{"a", "c"},
				},
				&Unit{
					Name: "c",
				},
			},
			expected: []string{"a", "c", "b"},
		},
		{
			units: []*Unit{
				&Unit{
					Name: "a",
					Deps: []string{"b"},
				},
				&Unit{
					Name: "b",
					Deps: []string{"c"},
				},
				&Unit{
					Name: "c",
					Deps: []string{"a"},
				},
			},
			expectErr: true,
		},
		{
			units: []*Unit{
				&Unit{
					Name: "a",
					Deps: []string{"a"},
				},
			},
			expectErr: true,
		},
	}

	for i, c := range cs {
		q, err := New(os.Stderr, c.units)
		if err != nil {
			if c.expectErr {
				continue
			}
			t.Errorf("[%d]unexpected err: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(c.expected, q.queue) {
			t.Errorf("[%d]\nexpected:\n\t%#v\ngot:\n\t%#v", i, c.expected, q.queue)
		}
	}
}

func TestRun(t *testing.T) {
	cs := []struct {
		units     []*Unit
		expectErr bool
	}{
		{
			units: []*Unit{
				&Unit{
					Name: "a",
					Execute: func(ctx context.Context, out io.Writer) error {
						out.Write([]byte("a\n"))
						return nil
					},
				},
				&Unit{
					Name: "b",
					Execute: func(ctx context.Context, out io.Writer) error {
						out.Write([]byte("b\n"))
						return nil
					},
					Deps: []string{"c"},
				},
				&Unit{
					Name: "c",
					Execute: func(ctx context.Context, out io.Writer) error {
						out.Write([]byte("c\n"))
						return nil
					},
				},
			},
		},
		{
			units: []*Unit{
				&Unit{
					Name: "a",
					Execute: func(ctx context.Context, out io.Writer) error {
						out.Write([]byte("a\n"))
						return fmt.Errorf("here")
					},
				},
				&Unit{
					Name: "b",
					Execute: func(ctx context.Context, out io.Writer) error {
						out.Write([]byte("b\n"))
						return nil
					},
					Deps: []string{"c"},
				},
				&Unit{
					Name: "c",
					Execute: func(ctx context.Context, out io.Writer) error {
						out.Write([]byte("c\n"))
						return nil
					},
				},
			},
			expectErr: true,
		},
	}

	for i, c := range cs {
		ctx := context.Background()
		q, err := New(os.Stderr, c.units)
		if err != nil {
			t.Errorf("[%d]unexpected err: %v", i, err)
			continue
		}
		if err := q.Run(ctx, len(c.units)); err != nil {
			if c.expectErr {
				continue
			}
			t.Errorf("[%d]unexpected err: %v", i, err)
			continue
		}
	}
}
