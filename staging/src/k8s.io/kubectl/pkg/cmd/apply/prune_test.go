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
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/printers"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

// errPrinter is a ResourcePrinter whose PrintObj always returns an error.
// It is used to verify that callers correctly propagate PrintObj errors.
type errPrinter struct{ err error }

func (e *errPrinter) PrintObj(_ runtime.Object, _ io.Writer) error { return e.err }

// TestPrunePropagatePrintObjError is a regression test for the bug where
// printer.PrintObj(obj, p.out) had its return value ignored inside the
// prune() loop, so any write failure (e.g. a broken pipe) was invisible
// and the command exited 0.
//
// After the fix, prune() must propagate the error returned by PrintObj.
func TestPrunePropagatePrintObjError(t *testing.T) {
	// Build a fake dynamic client pre-populated with one pod that carries the
	// last-applied-configuration annotation so the pruner will attempt to
	// print it as a "pruned" resource.
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "prune-me",
			Namespace: "default",
			UID:       types.UID("abc-123"),
			Annotations: map[string]string{
				corev1.LastAppliedConfigAnnotation: `{}`,
			},
		},
	}

	gvr := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme.Scheme, pod)

	printErr := fmt.Errorf("simulated write error: broken pipe")

	p := pruner{
		dynamicClient: fakeClient,
		// Empty visitedUids means the pod is not in the visited set → it will
		// fall through to the print path.
		visitedUids:       sets.New[types.UID](),
		visitedNamespaces: sets.New[string](),
		// DryRunClient skips the actual DELETE call; we only care about the
		// print path triggered afterward.
		dryRunStrategy: cmdutil.DryRunClient,
		toPrinter: func(_ string) (printers.ResourcePrinter, error) {
			return &errPrinter{err: printErr}, nil
		},
		out: io.Discard,
	}

	mapping := &meta.RESTMapping{
		Resource: gvr,
		GroupVersionKind: schema.GroupVersionKind{
			Group:   "",
			Version: "v1",
			Kind:    "Pod",
		},
		// meta.RESTScopeNamespace is the exported package-level var for namespace scope.
		Scope: meta.RESTScopeNamespace,
	}

	err := p.prune("default", mapping)
	if err == nil {
		t.Fatal("expected prune() to return an error when PrintObj fails, but got nil")
	}
	if !strings.Contains(err.Error(), printErr.Error()) {
		t.Errorf("expected error to contain %q, got: %v", printErr.Error(), err)
	}
}
