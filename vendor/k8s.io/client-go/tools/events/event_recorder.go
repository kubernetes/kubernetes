/*
Copyright 2019 The Kubernetes Authors.

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

package events

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/reference"

	"k8s.io/api/events/v1beta1"
	"k8s.io/client-go/tools/record/util"
	"k8s.io/klog"
)

type recorderImpl struct {
	scheme              *runtime.Scheme
	reportingController string
	reportingInstance   string
	*watch.Broadcaster
	clock clock.Clock
}

func (recorder *recorderImpl) Eventf(regarding runtime.Object, related runtime.Object, eventtype, reason, action, note string, args ...interface{}) {
	timestamp := metav1.MicroTime{time.Now()}
	message := fmt.Sprintf(note, args...)
	refRegarding, err := reference.GetReference(recorder.scheme, regarding)
	if err != nil {
		klog.Errorf("Could not construct reference to: '%#v' due to: '%v'. Will not report event: '%v' '%v' '%v'", regarding, err, eventtype, reason, message)
		return
	}
	refRelated, err := reference.GetReference(recorder.scheme, related)
	if err != nil {
		klog.V(9).Infof("Could not construct reference to: '%#v' due to: '%v'.", related, err)
	}
	if !util.ValidateEventType(eventtype) {
		klog.Errorf("Unsupported event type: '%v'", eventtype)
		return
	}
	event := recorder.makeEvent(refRegarding, refRelated, timestamp, eventtype, reason, message, recorder.reportingController, recorder.reportingInstance, action)
	go func() {
		defer utilruntime.HandleCrash()
		recorder.Action(watch.Added, event)
	}()
}

func (recorder *recorderImpl) makeEvent(refRegarding *v1.ObjectReference, refRelated *v1.ObjectReference, timestamp metav1.MicroTime, eventtype, reason, message string, reportingController string, reportingInstance string, action string) *v1beta1.Event {
	t := metav1.Time{Time: recorder.clock.Now()}
	namespace := refRegarding.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceSystem
	}
	return &v1beta1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", refRegarding.Name, t.UnixNano()),
			Namespace: namespace,
		},
		EventTime:           timestamp,
		Series:              nil,
		ReportingController: reportingController,
		ReportingInstance:   reportingInstance,
		Action:              action,
		Reason:              reason,
		Regarding:           *refRegarding,
		Related:             refRelated,
		Note:                message,
		Type:                eventtype,
		// TODO: remove this when we change conversion to convert eventSource
		// to reportingController
		DeprecatedSource: v1.EventSource{Component: reportingController},
	}
}
