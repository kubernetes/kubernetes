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
	attr     = admission.NewAttributesRecord(nil, nil, kind, "ns", "name", resource, "subresource", admission.Create, nil)
)

func TestObserveAdmissionStep(t *testing.T) {
	Metrics.reset()
	handler := WithStepMetrics(&mutatingFakeHandler{admission.NewHandler(admission.Create), true})
	handler.Admit(attr)
	wantLabels := map[string]string{
		"operation":   string(admission.Create),
		"group":       resource.Group,
		"version":     resource.Version,
		"resource":    resource.Resource,
		"subresource": "subresource",
		"type":        "admit",
		"rejected":    "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_step_admission_latencies_seconds", wantLabels, 1)
	expectFindMetric(t, "apiserver_admission_step_admission_latencies_seconds_summary", wantLabels)
}

func TestObserveAdmissionController(t *testing.T) {
	Metrics.reset()
	handler := WithControllerMetrics(&mutatingFakeHandler{admission.NewHandler(admission.Create), true}, "a")
	handler.Admit(attr)
	wantLabels := map[string]string{
		"name":        "a",
		"operation":   string(admission.Create),
		"group":       resource.Group,
		"version":     resource.Version,
		"resource":    resource.Resource,
		"subresource": "subresource",
		"type":        "admit",
		"rejected":    "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", wantLabels, 1)
}

func TestObserveWebhook(t *testing.T) {
	Metrics.reset()
	Metrics.ObserveWebhook(2*time.Second, false, attr, stepAdmit, "x")
	wantLabels := map[string]string{
		"name":        "x",
		"operation":   string(admission.Create),
		"group":       resource.Group,
		"version":     resource.Version,
		"resource":    resource.Resource,
		"subresource": "subresource",
		"type":        "admit",
		"rejected":    "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_webhook_admission_latencies_seconds", wantLabels, 1)
}

func TestWithMetrics(t *testing.T) {
	Metrics.reset()

	type Test struct {
		name      string
		ns        string
		operation admission.Operation
		handler   admission.Interface
		admit     bool
	}
	for _, test := range []Test{
		{
			"mutating-interfaces-admit",
			"some-ns",
			admission.Create,
			&mutatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), true},
			true,
		},
		{
			"mutating-interfaces-dont-admit",
			"some-ns",
			admission.Create,
			&mutatingFakeHandler{admission.NewHandler(admission.Create, admission.Update), false},
			false,
		},
	} {
		Metrics.reset()

		h := WithMetrics(test.handler, Metrics.ObserveAdmissionController, test.name)

		// test mutation
		err := h.Admit(admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{}, test.ns, "", schema.GroupVersionResource{}, "", test.operation, nil))
		if test.admit && err != nil {
			t.Errorf("expected admit to succeed, but failed: %v", err)
			continue
		} else if !test.admit && err == nil {
			t.Errorf("expected admit to fail, but it succeeded")
			continue
		}

		filter := map[string]string{"rejected": "false"}
		if !test.admit {
			filter["rejected"] = "true"
		}
		expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", filter, 1)
	}
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
