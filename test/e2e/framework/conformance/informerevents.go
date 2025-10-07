/*
Copyright 2025 The Kubernetes Authors.

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

package architecture

import (
	"fmt"
	"slices"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// eventRecorder implements [cache.ResourceEventHandler] by recording all events.
// It is thread-safe.
type eventRecorder struct {
	mutex  sync.Mutex
	events []event
}

// event describes one add/update/delete event.
// They can be distinguished based on which object(s) are set.
// Delete events may contain a tombstone instead of the actual
// deleted object.
type event struct {
	oldObj, newObj any
	isInitialList  bool
}

func (e event) Type() string {
	switch {
	case e.oldObj == nil:
		return "add"
	case e.newObj == nil:
		return "delete"
	case e.oldObj != nil && e.newObj != nil:
		return "update"
	default:
		return "null"
	}
}

// ID returns "[namespace/]name, uid" for the object
// described in the event. If for whatever reason
// that is different for the old and new event (an error!),
// it returns both strings separated by semicolon.
// If meta data access is not possible, the error string is returned.
//
// This is meant to be used with gomega.HaveEach and a matcher
// which checks for the expected ID.
func (e event) ID() string {
	oldID := id(e.oldObj)
	newID := id(e.newObj)
	if oldID == newID {
		return newID
	}
	if oldID != "" && newID != "" {
		return oldID + "; " + newID
	}
	if newID != "" {
		return newID
	}
	return oldID
}

func id(obj any) string {
	if obj == nil {
		return ""
	}
	if tomb, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tomb.Obj
	}
	metaData, err := meta.Accessor(obj)
	if err != nil {
		return err.Error()
	}
	return fmt.Sprintf("%s, %s", klog.KObj(metaData).String(), metaData.GetUID())
}

var _ cache.ResourceEventHandler = &eventRecorder{}

func (er *eventRecorder) OnAdd(obj any, isInitialList bool) {
	er.mutex.Lock()
	defer er.mutex.Unlock()

	er.events = append(er.events, event{
		newObj:        obj,
		isInitialList: isInitialList,
	})
}

func (er *eventRecorder) OnUpdate(oldObj, newObj any) {
	er.mutex.Lock()
	defer er.mutex.Unlock()

	er.events = append(er.events, event{
		oldObj: oldObj,
		newObj: newObj,
	})
}

func (er *eventRecorder) OnDelete(obj any) {
	er.mutex.Lock()
	defer er.mutex.Unlock()

	er.events = append(er.events, event{
		oldObj: obj,
	})
}

// list returns a shallow copy of the current list of events.
func (er *eventRecorder) list() eventList {
	er.mutex.Lock()
	defer er.mutex.Unlock()

	return eventList{Events: slices.Clone(er.events)}
}

// reset clears the current list of events.
// Should only be called during idle periods.
func (er *eventRecorder) reset() eventList {
	er.mutex.Lock()
	defer er.mutex.Unlock()

	events := eventList{Events: er.events}
	er.events = nil
	return events
}

// eventList adds pretty-printing to a slice of events.
type eventList struct {
	Events []event
}

// Types returns a comma-separated list of the type of each event.
// For the sake of simplicity the last entry also ends with a comma.
//
// This can be used in Gomega assertions like this:
//
//	gomega.Expect(events).To(gomega.HaveField("Types()", gomega.MatchRegexp("^add,(update,)*$"))
//	gomega.Expect(events).To(gomega.HaveField("Types()", gomega.MatchRegexp("^(update,)*$"))
func (el eventList) Types() string {
	var buffer strings.Builder

	for _, e := range el.Events {
		buffer.WriteString(e.Type())
		buffer.WriteRune(',')
	}

	return buffer.String()
}
