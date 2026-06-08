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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
)

// FakeRecorder is used as a fake during tests. It is thread safe. It is usable
// when created manually and not by NewFakeRecorder, however all events may be
// thrown away in this case.
type FakeRecorder struct {
	Events  chan string
	Verbose bool
}

var _ EventRecorderLogger = &FakeRecorder{}
var _ AnnotatedEventRecorder = &FakeRecorder{}

// Eventf emits an event
func (f *FakeRecorder) Eventf(regarding runtime.Object, related runtime.Object, eventtype, reason, action, note string, args ...interface{}) {
	f.writeEvent(regarding, related, nil, eventtype, reason, action, note, args...)
}

// AnnotatedEventf emits an event like Eventf, but with annotations attached
func (f *FakeRecorder) AnnotatedEventf(regarding runtime.Object, related runtime.Object, annotations map[string]string, eventtype, reason, action, note string, args ...interface{}) {
	f.writeEvent(regarding, related, annotations, eventtype, reason, action, note, args...)
}

func (f *FakeRecorder) WithLogger(logger klog.Logger) EventRecorderLogger {
	return f
}

// writeEvent constructs a string from the event parameters and sends
// it to the Events channel
func (f *FakeRecorder) writeEvent(regarding runtime.Object, related runtime.Object, annotations map[string]string, eventtype, reason, action, note string, args ...interface{}) {
	if f.Events != nil {
		msg := fmt.Sprintf(eventtype+" "+reason+" "+note, args...)
		if f.Verbose {
			msg = eventtype + " " + reason + " " + action + " " + fmt.Sprintf(note, args...) +
				objectString(regarding) + objectString(related) + annotationsString(annotations)
		}
		f.Events <- msg
	}
}

// annotationsString returns a formatted string of the annotations map,
// or empty string if none
func annotationsString(annotations map[string]string) string {
	annotationString := ""
	if len(annotations) > 0 {
		annotationString = " " + fmt.Sprint(annotations)
	}
	return annotationString
}

// objectString returns a formatted string with the object's kind and
// apiVersion
func objectString(object runtime.Object) string {
	objectString := ""
	if object != nil {
		gvk := object.GetObjectKind().GroupVersionKind()
		if !gvk.Empty() {
			objectString = fmt.Sprintf(" {kind=%s,apiVersion=%s}",
				gvk.Kind,
				gvk.GroupVersion(),
			)
		}
	}
	return objectString
}

// NewFakeRecorder creates new fake event recorder with event channel with
// buffer of given size.
func NewFakeRecorder(bufferSize int) *FakeRecorder {
	return &FakeRecorder{
		Events:  make(chan string, bufferSize),
		Verbose: false,
	}
}
