/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util/sets"
)

type fakeFlagSet struct {
	t   *testing.T
	set sets.String
}

func (f *fakeFlagSet) StringVar(p *string, name, value, usage string) {
	if p == nil {
		f.t.Errorf("unexpected nil pointer")
	}
	if usage == "" {
		f.t.Errorf("unexpected empty usage")
	}
	f.set.Insert(name)
}

func (f *fakeFlagSet) BoolVar(p *bool, name string, value bool, usage string) {
	if p == nil {
		f.t.Errorf("unexpected nil pointer")
	}
	if usage == "" {
		f.t.Errorf("unexpected empty usage")
	}
	f.set.Insert(name)
}

func (f *fakeFlagSet) UintVar(p *uint, name string, value uint, usage string) {
	if p == nil {
		f.t.Errorf("unexpected nil pointer")
	}
	if usage == "" {
		f.t.Errorf("unexpected empty usage")
	}
	f.set.Insert(name)
}

func (f *fakeFlagSet) DurationVar(p *time.Duration, name string, value time.Duration, usage string) {
	if p == nil {
		f.t.Errorf("unexpected nil pointer")
	}
	if usage == "" {
		f.t.Errorf("unexpected empty usage")
	}
	f.set.Insert(name)
}

func (f *fakeFlagSet) IntVar(p *int, name string, value int, usage string) {
	if p == nil {
		f.t.Errorf("unexpected nil pointer")
	}
	if usage == "" {
		f.t.Errorf("unexpected empty usage")
	}
	f.set.Insert(name)
}
