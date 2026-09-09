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

package genericclioptions

import (
	"k8s.io/cli-runtime/pkg/resource"
)

// NewSimpleFakeResourceFinder builds a super simple ResourceFinder that just iterates over the objects you provided
func NewSimpleFakeResourceFinder(infos ...*resource.Info) *FakeResourceFinder {
	return &FakeResourceFinder{
		Infos: infos,
	}
}

func (f *FakeResourceFinder) WithError(err error) *FakeResourceFinder {
	f.err = err
	return f
}

type FakeResourceFinder struct {
	Infos []*resource.Info
	err   error
}

// Do implements the interface
func (f *FakeResourceFinder) Do() resource.Visitor {
	return &fakeResourceResult{
		Infos: f.Infos,
		err:   f.err,
	}
}

type fakeResourceResult struct {
	Infos []*resource.Info
	err   error
}

// Visit just iterates over info
func (r *fakeResourceResult) Visit(fn resource.VisitorFunc) error {
	if r.err != nil {
		return r.err
	}
	for _, info := range r.Infos {
		err := fn(info, nil)
		if err != nil {
			return err
		}
	}
	return nil
}
