/*
Copyright 2014 The Kubernetes Authors.

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

package etcd

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/storage"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

func TestObjectVersioner(t *testing.T) {
	v := APIObjectVersioner{}
	if ver, err := v.ObjectResourceVersion(&storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}); err != nil || ver != 5 {
		t.Errorf("unexpected version: %d %v", ver, err)
	}
	if ver, err := v.ObjectResourceVersion(&storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "a"}}); err == nil || ver != 0 {
		t.Errorf("unexpected version: %d %v", ver, err)
	}
	obj := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "a"}}
	if err := v.UpdateObject(obj, 5); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if obj.ResourceVersion != "5" || obj.DeletionTimestamp != nil {
		t.Errorf("unexpected resource version: %#v", obj)
	}
}

func TestEtcdParseResourceVersion(t *testing.T) {
	testCases := []struct {
		Version       string
		ExpectVersion uint64
		Err           bool
	}{
		{Version: "", ExpectVersion: 0},
		{Version: "a", Err: true},
		{Version: " ", Err: true},
		{Version: "1", ExpectVersion: 1},
		{Version: "10", ExpectVersion: 10},
	}

	v := APIObjectVersioner{}
	testFuncs := []func(string) (uint64, error){
		v.ParseResourceVersion,
	}

	for _, testCase := range testCases {
		for i, f := range testFuncs {
			version, err := f(testCase.Version)
			switch {
			case testCase.Err && err == nil:
				t.Errorf("%s[%v]: unexpected non-error", testCase.Version, i)
			case testCase.Err && !storage.IsInvalidError(err):
				t.Errorf("%s[%v]: unexpected error: %v", testCase.Version, i, err)
			case !testCase.Err && err != nil:
				t.Errorf("%s[%v]: unexpected error: %v", testCase.Version, i, err)
			}
			if version != testCase.ExpectVersion {
				t.Errorf("%s[%v]: expected version %d but was %d", testCase.Version, i, testCase.ExpectVersion, version)
			}
		}
	}
}

func TestCompareResourceVersion(t *testing.T) {
	five := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}
	six := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "6"}}

	versioner := APIObjectVersioner{}

	if e, a := -1, versioner.CompareResourceVersion(five, six); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
	if e, a := 1, versioner.CompareResourceVersion(six, five); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
	if e, a := 0, versioner.CompareResourceVersion(six, six); e != a {
		t.Errorf("expected %v got %v", e, a)
	}
}
