/*
Copyright 2024 The Kubernetes Authors.

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

package apply

import (
	"fmt"
	"io"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/printers"
	dynamicfakeclient "k8s.io/client-go/dynamic/fake"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

// errPrinter is a ResourcePrinter whose PrintObj always returns a fixed error.
type errPrinter struct{ err error }

func (e *errPrinter) PrintObj(_ runtime.Object, _ io.Writer) error { return e.err }

// TestPrunerPropagatesPrintObjError verifies that a printer error inside
// pruner.prune is returned to the caller rather than silently discarded.
// Regression test for the silent-discard bug at prune.go#L137.
func TestPrunerPropagatesPrintObjError(t *testing.T) {
	// Build a ConfigMap that looks like it was created by kubectl apply
	// (has LastAppliedConfigAnnotation) so that pruner will try to print it.
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name":      "to-prune",
				"namespace": "test",
				"uid":       "uid-prune",
				"annotations": map[string]interface{}{
					corev1.LastAppliedConfigAnnotation: "{}",
				},
			},
		},
	}

	// NewSimpleDynamicClient pre-seeds the object; no separate Create needed.
	dynamicClient := dynamicfakeclient.NewSimpleDynamicClient(scheme.Scheme, obj)

	// Mapper that resolves ConfigMap to the namespaced REST mapping.
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{{Group: "", Version: "v1"}})
	mapper.Add(corev1.SchemeGroupVersion.WithKind("ConfigMap"), meta.RESTScopeNamespace)

	wantErr := fmt.Errorf("simulated printer failure")
	p := &pruner{
		mapper:        mapper,
		dynamicClient: dynamicClient,

		// No visited UIDs → the object will not be skipped.
		visitedUids:       sets.New[types.UID](),
		visitedNamespaces: sets.New[string](),

		// DryRunClient so pruner.delete is never called; we only test the print path.
		dryRunStrategy: cmdutil.DryRunClient,

		toPrinter: func(_ string) (printers.ResourcePrinter, error) {
			return &errPrinter{err: wantErr}, nil
		},
		out: io.Discard,
	}

	mapping, err := mapper.RESTMapping(corev1.SchemeGroupVersion.WithKind("ConfigMap").GroupKind(), "v1")
	if err != nil {
		t.Fatalf("RESTMapping: %v", err)
	}

	err = p.prune("test", mapping)
	if err == nil {
		t.Fatal("expected prune to return an error, got nil")
	}
	if err.Error() != wantErr.Error() {
		t.Errorf("expected error %q, got %q", wantErr.Error(), err.Error())
	}
}
