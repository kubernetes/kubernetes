/*
Copyright The Kubernetes Authors.

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

package metrics

import (
	"testing"

	"k8s.io/ktesting"
)

func TestClientNameContext(t *testing.T) {
	tCtx := ktesting.Init(t)

	const want = "deployment-controller"

	ctx := WithClientName(tCtx, want)

	got, ok := ClientNameFrom(ctx)
	if !ok {
		tCtx.Fatal("ClientNameFrom() did not find client name")
	}
	if got != want {
		tCtx.Errorf("ClientNameFrom() = %q, want %q", got, want)
	}
}

func TestRequestInfoContext(t *testing.T) {
	tCtx := ktesting.Init(t)

	want := RequestInfo{
		Verb:     "LIST",
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
		Scope:    "namespace",
	}

	ctx := WithRequestInfo(tCtx, want)

	got, ok := RequestInfoFrom(ctx)
	if !ok {
		tCtx.Fatal("RequestInfoFrom did not find request info")
	}
	if got != want {
		tCtx.Errorf("RequestInfoFrom() = %#v, want %#v", got, want)
	}
}

func TestMetricsContextValuesNotSet(t *testing.T) {
	tCtx := ktesting.Init(t)

	if got, ok := RequestInfoFrom(tCtx); ok {
		tCtx.Errorf("RequestInfoFrom() = %#v, true; want zero value, false", got)
	}
	if got, ok := ClientNameFrom(tCtx); ok {
		tCtx.Errorf("ClientNameFrom() = %q, true; want empty string, false", got)
	}
}

func TestMetricsContextValuesCoexist(t *testing.T) {
	tCtx := ktesting.Init(t)

	wantInfo := RequestInfo{
		Verb:     "WATCH",
		Version:  "v1",
		Resource: "pods",
		Scope:    "namespace",
	}
	const wantName = "pod-informer"

	ctx := WithRequestInfo(tCtx, wantInfo)
	ctx = WithClientName(ctx, wantName)

	gotInfo, ok := RequestInfoFrom(ctx)
	if !ok {
		tCtx.Fatal("RequestInfoFrom did not find request info")
	}
	if gotInfo != wantInfo {
		tCtx.Errorf("RequestInfoFrom() = %#v, want %#v", gotInfo, wantInfo)
	}

	gotName, ok := ClientNameFrom(ctx)
	if !ok {
		tCtx.Fatal("ClientNameFrom did not find client name")
	}
	if gotName != wantName {
		tCtx.Errorf("ClientNameFrom() = %q, want %q", gotName, wantName)
	}
}
