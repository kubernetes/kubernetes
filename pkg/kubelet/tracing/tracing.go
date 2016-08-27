/*
Copyright 2014 The Kubernetes Authors.

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

package tracing

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/clock"

	"github.com/golang/glog"
)

var (
	// Set to true for test before we add tracing flag
	enabled      = true
	logf         = glog.Infof
	tracingClock clock.RealClock
)

const (
	// The list of probes
	PodCreatefirstSeen = "PodCreateFirstSeen"
	PodCreateRunning   = "PodCreateRunning"

	// Attributes of tracing events
	eventtype = api.EventTypeNormal
	reason    = events.NodeTracing
)

// Message is the message of a tracing event.
type Message struct {
	PodUID    types.UID
	Probe     string
	Timestamp time.Time
}

// ToString converts a message to string.
func (m *Message) ToString() string {
	return fmt.Sprintf("pod: %s, probe: %s, timestamp: %d", m.PodUID, m.Probe, m.Timestamp.UnixNano())
}

// SetEnable enables tracing by activating tracing probes.
func SetEnable(en bool) {
	enabled = en
}

// NewMessage creates a new message and return a pointer. It uses current time as timestamp.
func NewMessage(podUID types.UID, probe string) *Message {
	return NewMessageWithTs(podUID, probe, tracingClock.Now())
}

// NewMessage creates a new message with a given timestamp and return a pointer.
func NewMessageWithTs(podUID types.UID, probe string, timestamp time.Time) *Message {
	return &Message{
		PodUID:    podUID,
		Probe:     probe,
		Timestamp: timestamp,
	}
}

// SetProbe logs a tracing event.
func SetProbe(object runtime.Object, message *Message) {
	if enabled {
		ref, err := api.GetReference(object)
		if err != nil {
			glog.Errorf("Could not construct reference to: '%#v' due to: '%v'. Will not report event: '%v' '%v' '%v'",
				object, err, eventtype, reason, message)
			return
		}
		InvolvedObject := *ref
		logf("Event(%#v): type: '%v' reason: '%v' %v", InvolvedObject, eventtype, reason, message.ToString())
	}
}
