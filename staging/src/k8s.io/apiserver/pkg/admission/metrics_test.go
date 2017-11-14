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

package admission

import (
	"testing"
	"time"

	"k8s.io/api/admissionregistration/v1alpha1"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	kind     = schema.GroupVersionKind{Group: "kgroup", Version: "kversion", Kind: "kind"}
	resource = schema.GroupVersionResource{Group: "rgroup", Version: "rversion", Resource: "resource"}
	attr     = NewAttributesRecord(nil, nil, kind, "ns", "name", resource, "subresource", Create, nil)
)

func TestObserveAdmissionStep(t *testing.T) {
	Metrics.reset()
	Metrics.ObserveAdmissionStep(2*time.Second, false, attr, "admit")
	wantLabels := map[string]string{
		"operation":   string(Create),
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
	handler := makeValidatingNamedHandler("a", true, Create)
	Metrics.ObserveAdmissionController(2*time.Second, false, handler, attr, "validate")
	wantLabels := map[string]string{
		"name":        "a",
		"operation":   string(Create),
		"group":       resource.Group,
		"version":     resource.Version,
		"resource":    resource.Resource,
		"subresource": "subresource",
		"type":        "validate",
		"rejected":    "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_controller_admission_latencies_seconds", wantLabels, 1)
	expectFindMetric(t, "apiserver_admission_controller_admission_latencies_seconds_summary", wantLabels)
}

func TestObserveWebhook(t *testing.T) {
	Metrics.reset()
	hook := &v1alpha1.Webhook{Name: "x"}
	Metrics.ObserveWebhook(2*time.Second, false, hook, attr)
	wantLabels := map[string]string{
		"name":        "x",
		"operation":   string(Create),
		"group":       resource.Group,
		"version":     resource.Version,
		"resource":    resource.Resource,
		"subresource": "subresource",
		"type":        "admit",
		"rejected":    "false",
	}
	expectHistogramCountTotal(t, "apiserver_admission_webhook_admission_latencies_seconds", wantLabels, 1)
	expectFindMetric(t, "apiserver_admission_webhook_admission_latencies_seconds_summary", wantLabels)
}
