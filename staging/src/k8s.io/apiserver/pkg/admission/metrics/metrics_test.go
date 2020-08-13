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

package metrics

import (
	"context"
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

var (
	kind     = schema.GroupVersionKind{Group: "kgroup", Version: "kversion", Kind: "kind"}
	resource = schema.GroupVersionResource{Group: "rgroup", Version: "rversion", Resource: "resource"}
	attr     = admission.NewAttributesRecord(nil, nil, kind, "ns", "name", resource, "subresource", admission.Create, &metav1.CreateOptions{}, false, nil)
)

func TestObserveAdmissionStep(t *testing.T) {
	Metrics.reset()
	handler := WithStepMetrics(&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create), true, true})
	if err := handler.(admission.MutationInterface).Admit(context.TODO(), attr, nil); err != nil {
		t.Errorf("Unexpected error in admit: %v", err)
	}
	if err := handler.(admission.ValidationInterface).Validate(context.TODO(), attr, nil); err != nil {
		t.Errorf("Unexpected error in validate: %v", err)
	}
	wantLabels := map[string]string{
		"operation": string(admission.Create),
		"type":      "admit",
		"rejected":  "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_step_admission_duration_seconds", wantLabels, 1)
	expectFindMetric(t, "apiserver_admission_step_admission_duration_seconds_summary", wantLabels)

	wantLabels["type"] = "validate"
	expectHistogramCountTotal(t, "apiserver_admission_step_admission_duration_seconds", wantLabels, 1)
	expectFindMetric(t, "apiserver_admission_step_admission_duration_seconds_summary", wantLabels)
}

func TestObserveAdmissionController(t *testing.T) {
	Metrics.reset()
	handler := WithControllerMetrics(&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create), true, true}, "a")
	if err := handler.(admission.MutationInterface).Admit(context.TODO(), attr, nil); err != nil {
		t.Errorf("Unexpected error in admit: %v", err)
	}
	if err := handler.(admission.ValidationInterface).Validate(context.TODO(), attr, nil); err != nil {
		t.Errorf("Unexpected error in validate: %v", err)
	}
	wantLabels := map[string]string{
		"name":      "a",
		"operation": string(admission.Create),
		"type":      "admit",
		"rejected":  "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_controller_admission_duration_seconds", wantLabels, 1)

	wantLabels["type"] = "validate"
	expectHistogramCountTotal(t, "apiserver_admission_controller_admission_duration_seconds", wantLabels, 1)
}

func TestObserveWebhook(t *testing.T) {
	Metrics.reset()
	Metrics.ObserveWebhook(2*time.Second, false, attr, stepAdmit, "x")
	wantLabels := map[string]string{
		"name":      "x",
		"operation": string(admission.Create),
		"type":      "admit",
		"rejected":  "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_webhook_admission_duration_seconds", wantLabels, 1)
}

func TestObserveWebhookRejection(t *testing.T) {
	Metrics.reset()
	Metrics.ObserveWebhookRejection("x", stepAdmit, string(admission.Create), WebhookRejectionNoError, 500)
	Metrics.ObserveWebhookRejection("x", stepAdmit, string(admission.Create), WebhookRejectionNoError, 654)
	Metrics.ObserveWebhookRejection("x", stepValidate, string(admission.Update), WebhookRejectionCallingWebhookError, 0)
	wantLabels := map[string]string{
		"name":           "x",
		"operation":      string(admission.Create),
		"type":           "admit",
		"error_type":     "no_error",
		"rejection_code": "500",
	}
	wantLabels600 := map[string]string{
		"name":           "x",
		"operation":      string(admission.Create),
		"type":           "admit",
		"error_type":     "no_error",
		"rejection_code": "600",
	}
	wantLabelsCallingWebhookError := map[string]string{
		"name":           "x",
		"operation":      string(admission.Update),
		"type":           "validate",
		"error_type":     "calling_webhook_error",
		"rejection_code": "0",
	}
	expectCounterValue(t, "apiserver_admission_webhook_rejection_count", wantLabels, 1)
	expectCounterValue(t, "apiserver_admission_webhook_rejection_count", wantLabels600, 1)
	expectCounterValue(t, "apiserver_admission_webhook_rejection_count", wantLabelsCallingWebhookError, 1)
}

func TestWithMetrics(t *testing.T) {
	Metrics.reset()

	type Test struct {
		name            string
		ns              string
		operation       admission.Operation
		options         runtime.Object
		handler         admission.Interface
		admit, validate bool
	}
	for _, test := range []Test{
		{
			"both-interfaces-admit-and-validate",
			"some-ns",
			admission.Create,
			&metav1.CreateOptions{},
			&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true, true},
			true, true,
		},
		{
			"both-interfaces-dont-admit",
			"some-ns",
			admission.Create,
			&metav1.CreateOptions{},
			&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), false, true},
			false, true,
		},
		{
			"both-interfaces-admit-dont-validate",
			"some-ns",
			admission.Create,
			&metav1.CreateOptions{},
			&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true, false},
			true, false,
		},
		{
			"validate-interfaces-validate",
			"some-ns",
			admission.Create,
			&metav1.CreateOptions{},
			&validatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true},
			true, true,
		},
		{
			"validate-interfaces-dont-validate",
			"some-ns",
			admission.Create,
			&metav1.CreateOptions{},
			&validatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), false},
			true, false,
		},
		{
			"mutating-interfaces-admit",
			"some-ns",
			admission.Create,
			&metav1.CreateOptions{},
			&mutatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true},
			true, true,
		},
		{
			"mutating-interfaces-dont-admit",
			"some-ns",
			admission.Create,
			&metav1.CreateOptions{},
			&mutatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), false},
			false, true,
		},
	} {
		Metrics.reset()

		h := WithMetrics(test.handler, Metrics.ObserveAdmissionController, test.name)

		// test mutation
		err := h.(admission.MutationInterface).Admit(context.TODO(), admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, test.ns, "", schema.GroupVersionResource{}, "", test.operation, test.options, false, nil), nil)
		if test.admit && err != nil {
			t.Errorf("expected admit to succeed, but failed: %v", err)
			continue
		} else if !test.admit && err == nil {
			t.Errorf("expected admit to fail, but it succeeded")
			continue
		}

		filter := map[string]string{"type": "admit", "rejected": "false"}
		if !test.admit {
			filter["rejected"] = "true"
		}
		if _, mutating := test.handler.(admission.MutationInterface); mutating {
			expectHistogramCountTotal(t, "apiserver_admission_controller_admission_duration_seconds", filter, 1)
		} else {
			expectHistogramCountTotal(t, "apiserver_admission_controller_admission_duration_seconds", filter, 0)
		}

		if err != nil {
			// skip validation step if mutation failed
			continue
		}

		// test validation
		err = h.(admission.ValidationInterface).Validate(context.TODO(), admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, test.ns, "", schema.GroupVersionResource{}, "", test.operation, test.options, false, nil), nil)
		if test.validate && err != nil {
			t.Errorf("expected admit to succeed, but failed: %v", err)
			continue
		} else if !test.validate && err == nil {
			t.Errorf("expected validation to fail, but it succeeded")
			continue
		}

		filter = map[string]string{"type": "validate", "rejected": "false"}
		if !test.validate {
			filter["rejected"] = "true"
		}
		if _, validating := test.handler.(admission.ValidationInterface); validating {
			expectHistogramCountTotal(t, "apiserver_admission_controller_admission_duration_seconds", filter, 1)
		} else {
			expectHistogramCountTotal(t, "apiserver_admission_controller_admission_duration_seconds", filter, 0)
		}
	}
}

type mutatingAndValidatingFakeHandler struct {
	*admission.Handler
	admit    bool
	validate bool
}

func (h *mutatingAndValidatingFakeHandler) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if h.admit {
		return nil
	}
	return fmt.Errorf("don't admit")
}

func (h *mutatingAndValidatingFakeHandler) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if h.validate {
		return nil
	}
	return fmt.Errorf("don't validate")
}

type validatingFakeHandler struct {
	*admission.Handler
	validate bool
}

func (h *validatingFakeHandler) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if h.validate {
		return nil
	}
	return fmt.Errorf("don't validate")
}

type mutatingFakeHandler struct {
	*admission.Handler
	admit bool
}

func (h *mutatingFakeHandler) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if h.admit {
		return nil
	}
	return fmt.Errorf("don't admit")
}
