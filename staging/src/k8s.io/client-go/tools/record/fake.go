/*
Copyright 2015 The Kubernetes Authors.

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

package record

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
)

// FakeRecorder is used as a fake during tests. It is thread safe. It is usable
// when created manually and not by NewFakeRecorder, however all events may be
// thrown away in this case.
type FakeRecorder struct {
	Events chan string

	IncludeObject      bool
	IncludeAnnotations bool
}

func objectString(object runtime.Object, includeObject bool) string {
	if !includeObject {
		return ""
	}
	return fmt.Sprintf(" involvedObject{kind=%s,apiVersion=%s}",
		object.GetObjectKind().GroupVersionKind().Kind,
		object.GetObjectKind().GroupVersionKind().GroupVersion(),
	)
}

func (f *FakeRecorder) Event(object runtime.Object, eventtype, reason, message string) {
	if f.Events != nil {
		f.Events <- fmt.Sprintf("%s %s %s%s", eventtype, reason, message, objectString(object, f.IncludeObject))
	}
}

func (f *FakeRecorder) Eventf(object runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
	if f.Events != nil {
		f.Events <- fmt.Sprintf(eventtype+" "+reason+" "+messageFmt, args...) + objectString(object, f.IncludeObject)
	}
}

func (f *FakeRecorder) AnnotatedEventf(object runtime.Object, annotations map[string]string, eventtype, reason, messageFmt string, args ...interface{}) {
	if f.IncludeAnnotations {
		args = append(args, fmt.Sprint(annotations))
		f.Eventf(object, eventtype, reason, messageFmt+" %s", args...)
	} else {
		f.Eventf(object, eventtype, reason, messageFmt, args...)
	}
}

// NewFakeRecorder creates new fake event recorder with event channel with
// buffer of given size.
func NewFakeRecorder(bufferSize int) *FakeRecorder {
	return &FakeRecorder{
		Events: make(chan string, bufferSize),
	}
}
