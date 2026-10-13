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

package testing

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/server/statusz/api/v1alpha1"
	"k8s.io/apiserver/pkg/server/statusz/api/v1beta1"
)

// VerifyStructuredResponse verifies that the response body matches the expected static fields
// and checks for the presence/absence of warnings.
func VerifyStructuredResponse(t *testing.T, acceptHeader string, body []byte, warnings []string, want interface{}, wantDeprecationHeader bool) {
	t.Helper()

	unmarshal := unmarshalFunc(t, acceptHeader)
	wantTypeMeta, wantName, wantPaths := wantFields(t, want)
	gotTypeMeta, gotName, gotPaths := gotFields(t, unmarshal, body, wantTypeMeta.APIVersion)

	if gotName != wantName {
		t.Errorf("ObjectMeta.Name mismatch: want %q, got %q", wantName, gotName)
	}
	if gotTypeMeta != wantTypeMeta {
		t.Errorf("TypeMeta mismatch: want %+v, got %+v", wantTypeMeta, gotTypeMeta)
	}
	if diff := cmp.Diff(wantPaths, gotPaths); diff != "" {
		t.Errorf("Paths mismatch (-want,+got):\n%s", diff)
	}

	foundWarning := false
	for _, w := range warnings {
		if strings.Contains(w, "deprecated") {
			foundWarning = true
			break
		}
	}

	if wantDeprecationHeader {
		if !foundWarning {
			t.Errorf("want warning 'deprecated', got none or different: %v", warnings)
		}
	} else {
		if foundWarning {
			t.Errorf("want no warning 'deprecated', got one: %v", warnings)
		}
	}
}

func unmarshalFunc(t *testing.T, acceptHeader string) func([]byte, interface{}) error {
	switch {
	case strings.Contains(acceptHeader, "application/json"):
		return json.Unmarshal
	case strings.Contains(acceptHeader, "application/yaml"):
		return yaml.Unmarshal
	case strings.Contains(acceptHeader, "application/cbor"):
		return cbor.Unmarshal
	default:
		t.Fatalf("unexpected Accept header: %q", acceptHeader)
	}
	return nil
}

func wantFields(t *testing.T, want interface{}) (metav1.TypeMeta, string, []string) {
	t.Helper()
	switch w := want.(type) {
	case *v1alpha1.Statusz:
		return w.TypeMeta, w.Name, w.Paths
	case *v1beta1.Statusz:
		return w.TypeMeta, w.Name, w.Paths
	default:
		t.Fatalf("unexpected type: %T", want)
	}
	return metav1.TypeMeta{}, "", nil
}

func gotFields(t *testing.T, unmarshal func([]byte, interface{}) error, body []byte, apiVersion string) (metav1.TypeMeta, string, []string) {
	t.Helper()
	switch apiVersion {
	case "config.k8s.io/v1alpha1":
		var got v1alpha1.Statusz
		if err := unmarshal(body, &got); err != nil {
			t.Fatalf("error unmarshalling %s: %v", apiVersion, err)
		}
		return got.TypeMeta, got.Name, got.Paths
	case "config.k8s.io/v1beta1":
		var got v1beta1.Statusz
		if err := unmarshal(body, &got); err != nil {
			t.Fatalf("error unmarshalling %s: %v", apiVersion, err)
		}
		return got.TypeMeta, got.Name, got.Paths
	default:
		t.Fatalf("unexpected APIVersion: %q", apiVersion)
	}
	return metav1.TypeMeta{}, "", nil
}
