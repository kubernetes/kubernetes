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
	Metrics.ObserveAdmissionStep(2*time.Second, false, attr, "mutating")
	wantLabels := metricLabels{
		operation:   string(Create),
		group:       resource.Group,
		version:     resource.Version,
		resource:    resource.Resource,
		subresource: "subresource",
		tpe:         "mutating",
		isSystemNs:  false,
	}
	expectCountMetric(t, "apiserver_admission_step_total", wantLabels, 1)
}

func TestObserveAdmissionController(t *testing.T) {
	Metrics.reset()
	handler := makeHandler("a", true, Create)
	Metrics.ObserveAdmissionController(2*time.Second, false, handler, attr)
	wantLabels := metricLabels{
		name:        "a",
		operation:   string(Create),
		group:       resource.Group,
		version:     resource.Version,
		resource:    resource.Resource,
		subresource: "subresource",
		tpe:         "mutating",
		isSystemNs:  false,
	}
	expectCountMetric(t, "apiserver_admission_controller_total", wantLabels, 1)
}

func TestObserveExternalWebhook(t *testing.T) {
	Metrics.reset()
	hook := &v1alpha1.ExternalAdmissionHook{Name: "x"}
	Metrics.ObserveExternalWebhook(2*time.Second, false, hook, attr)
	wantLabels := metricLabels{
		name:        "x",
		operation:   string(Create),
		group:       resource.Group,
		version:     resource.Version,
		resource:    resource.Resource,
		subresource: "subresource",
		tpe:         "validating",
		isSystemNs:  false,
	}
	expectCountMetric(t, "apiserver_admission_external_webhook_total", wantLabels, 1)
}
