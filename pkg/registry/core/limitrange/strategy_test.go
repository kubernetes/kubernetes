package limitrange

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestStrategyValidate(t *testing.T) {
	lr := &api.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default", // Add this!
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{},
		},
	}

	errs := Strategy.Validate(context.Background(), lr)
	if len(errs) != 0 {
		t.Fatalf("unexpected errors from Validate: %v", errs)
	}
}

func TestStrategyValidateUpdate(t *testing.T) {
	oldLR := &api.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default", // Add this!
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{},
		},
	}

	newLR := &api.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default", // Add this!
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{},
		},
	}

	errs := Strategy.ValidateUpdate(context.Background(), newLR, oldLR)
	if len(errs) != 0 {
		t.Fatalf("unexpected errors from ValidateUpdate: %v", errs)
	}
}

func TestStrategyInvalid(t *testing.T) {
	lr := &api.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "",        // Empty name should fail
			Namespace: "default", // Add this!
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{},
		},
	}

	errs := Strategy.Validate(context.Background(), lr)
	if len(errs) == 0 {
		t.Fatalf("expected validation errors but got none")
	}
}
