package meta

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type fakeResource struct {
	metav1.TypeMeta
	metav1.ObjectMeta
}

func (o *fakeResource) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}
func (o *fakeResource) DeepCopyObject() runtime.Object {
	t := *o
	return &t
}

// TestAddRemoveFinalizer tests the add/remove and hasFinalizer methods.
func TestAddRemoveFinalizer(t *testing.T) {
	resource := &fakeResource{}

	// add one works
	prev := resource.DeepCopyObject().(*fakeResource)
	curr := AddFinalizer(resource, "the-finalizer").(*fakeResource)
	if FinalizersEqual(curr, prev) {
		t.Fatalf("unexpected result")
	}
	if !HasFinalizer(curr, "the-finalizer") {
		t.Fatalf("unexpected result")
	}
	if HasFinalizer(curr, "the-other-finalizer") {
		t.Fatalf("unexpected result")
	}
	// confirm we didn't mutate input
	if !reflect.DeepEqual(prev, resource) {
		t.Fatalf("unexpected result")
	}

	// re-add doesn't mutate
	prev = curr.DeepCopyObject().(*fakeResource)
	curr = AddFinalizer(curr, "the-finalizer").(*fakeResource)
	if !FinalizersEqual(curr, prev) {
		t.Fatalf("unexpected result")
	}

	// add second still works
	prev = curr.DeepCopyObject().(*fakeResource)
	curr = AddFinalizer(curr, "the-other-finalizer").(*fakeResource)
	if FinalizersEqual(curr, prev) {
		t.Fatalf("unexpected result")
	}
	if !HasFinalizer(curr, "the-finalizer") {
		t.Fatalf("unexpected result")
	}
	if !HasFinalizer(curr, "the-other-finalizer") {
		t.Fatalf("unexpected result")
	}

	// remove one works
	prev = curr.DeepCopyObject().(*fakeResource)
	curr = RemoveFinalizer(curr, "the-other-finalizer").(*fakeResource)
	if FinalizersEqual(curr, prev) {
		t.Fatalf("unexpected result")
	}
	if !HasFinalizer(curr, "the-finalizer") {
		t.Fatalf("unexpected result")
	}
	if HasFinalizer(curr, "the-other-finalizer") {
		t.Fatalf("unexpected result")
	}
	// confirm we didn't mutate input
	if reflect.DeepEqual(prev, curr) { // if we mutated input, these would be the same
		t.Fatalf("unexpected result")
	}

	// re-remove doesn't mutate
	prev = curr.DeepCopyObject().(*fakeResource)
	curr = RemoveFinalizer(curr, "the-other-finalizer").(*fakeResource)
	if !FinalizersEqual(curr, prev) {
		t.Fatalf("unexpected result")
	}
	if !HasFinalizer(curr, "the-finalizer") {
		t.Fatalf("unexpected result")
	}
}
