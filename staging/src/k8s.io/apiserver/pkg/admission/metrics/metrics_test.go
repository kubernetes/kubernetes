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
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

var (
	kind     = schema.GroupVersionKind{Group: "kgroup", Version: "kversion", Kind: "kind"}
	resource = schema.GroupVersionResource{Group: "rgroup", Version: "rversion", Resource: "resource"}
	attr     = admission.NewAttributesRecord(nil, nil, kind, "ns", "name", resource, "subresource", admission.Create, false, nil)
)

func TestObserveAdmissionStep(t *testing.T) {
	Metrics.reset()
	handler := WithStepMetrics(&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create), true, true})
	handler.(admission.MutationInterface).Admit(attr)
	handler.(admission.ValidationInterface).Validate(attr)
	wantLabels := map[string]string{
		"operation": string(admission.Create),
		"type":      "admit",
		"rejected":  "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_step_admission_latencies_seconds", wantLabels, 1)
	expectFindMetric(t, "apiserver_admission_step_admission_latencies_seconds_summary", wantLabels)

	wantLabels["type"] = "validate"
	expectHistogramCountTotal(t, "apiserver_admission_step_admission_latencies_seconds", wantLabels, 1)
	expectFindMetric(t, "apiserver_admission_step_admission_latencies_seconds_summary", wantLabels)
}

func TestObserveAdmissionController(t *testing.T) {
	Metrics.reset()
	handler := WithControllerMetrics(&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create), true, true}, "a")
	handler.(admission.MutationInterface).Admit(attr)
	handler.(admission.ValidationInterface).Validate(attr)
	wantLabels := map[string]string{
		"name":      "a",
		"operation": string(admission.Create),
		"type":      "admit",
		"rejected":  "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", wantLabels, 1)

	wantLabels["type"] = "validate"
	expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", wantLabels, 1)
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
	expectHistogramCountTotal(t, "apiserver_admission_webhook_admission_latencies_seconds", wantLabels, 1)
}

func TestWithMetrics(t *testing.T) {
	Metrics.reset()

	type Test struct {
		name            string
		ns              string
		operation       admission.Operation
		handler         admission.Interface
		admit, validate bool
	}
	for _, test := range []Test{
		{
			"both-interfaces-admit-and-validate",
			"some-ns",
			admission.Create,
			&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true, true},
			true, true,
		},
		{
			"both-interfaces-dont-admit",
			"some-ns",
			admission.Create,
			&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), false, true},
			false, true,
		},
		{
			"both-interfaces-admit-dont-validate",
			"some-ns",
			admission.Create,
			&mutatingAndValidatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true, false},
			true, false,
		},
		{
			"validate-interfaces-validate",
			"some-ns",
			admission.Create,
			&validatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true},
			true, true,
		},
		{
			"validate-interfaces-dont-validate",
			"some-ns",
			admission.Create,
			&validatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), false},
			true, false,
		},
		{
			"mutating-interfaces-admit",
			"some-ns",
			admission.Create,
			&mutatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true},
			true, true,
		},
		{
			"mutating-interfaces-dont-admit",
			"some-ns",
			admission.Create,
			&mutatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), false},
			false, true,
		},
	} {
		Metrics.reset()

		h := WithMetrics(test.handler, Metrics.ObserveAdmissionController, test.name)

		// test mutation
		err := h.(admission.MutationInterface).Admit(admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, test.ns, "", schema.GroupVersionResource{}, "", test.operation, false, nil))
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
			expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", filter, 1)
		} else {
			expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", filter, 0)
		}

		if err != nil {
			// skip validation step if mutation failed
			continue
		}

		// test validation
		err = h.(admission.ValidationInterface).Validate(admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, test.ns, "", schema.GroupVersionResource{}, "", test.operation, false, nil))
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
			expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", filter, 1)
		} else {
			expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", filter, 0)
		}
	}
}

type mutatingAndValidatingFakeHandler struct {
	*admission.Handler
	admit    bool
	validate bool
}

func (h *mutatingAndValidatingFakeHandler) Admit(a admission.Attributes) (err error) {
	if h.admit {
		return nil
	}
	return fmt.Errorf("don't admit")
}

func (h *mutatingAndValidatingFakeHandler) Validate(a admission.Attributes) (err error) {
	if h.validate {
		return nil
	}
	return fmt.Errorf("don't validate")
}

type validatingFakeHandler struct {
	*admission.Handler
	validate bool
}

func (h *validatingFakeHandler) Validate(a admission.Attributes) (err error) {
	if h.validate {
		return nil
	}
	return fmt.Errorf("don't validate")
}

type mutatingFakeHandler struct {
	*admission.Handler
	admit bool
}

func (h *mutatingFakeHandler) Admit(a admission.Attributes) (err error) {
	if h.admit {
		return nil
	}
	return fmt.Errorf("don't admit")
}
