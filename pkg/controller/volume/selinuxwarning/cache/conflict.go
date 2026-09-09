/*
Copyright 2024 The Kubernetes Authors.

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

package cache

import (
	"fmt"

	"k8s.io/client-go/tools/cache"
)

// A single conflict between two Pods using the same volume with different SELinux labels or policies.
// Event should be sent to both of them.
type Conflict struct {
	// Human-readable name of the conflicting property + value of "property" label of selinux_volume_conflict metric.
	PropertyName string
	// Reason for the event, to be set as the Event.Reason field.
	EventReason string

	// Pod to generate the event on
	Pod           cache.ObjectName
	PropertyValue string
	// only for logging / messaging
	OtherPod           cache.ObjectName
	OtherPropertyValue string
}

// Generate a message about this conflict.
func (c *Conflict) EventMessage() string {
	// Quote the values for better readability.
	value := "\"" + c.PropertyValue + "\""
	otherValue := "\"" + c.OtherPropertyValue + "\""
	if c.Pod.Namespace == c.OtherPod.Namespace {
		// In the same namespace, be very specific about the pod names.
		return fmt.Sprint(c.PropertyName, " ", value, " conflicts with pod ", c.OtherPod.Name, " that uses the same volume as this pod with ", c.PropertyName, " ", otherValue, ". If both pods land on the same node, only one of them may access the volume.")
	}
	// Pods are in different namespaces, do not reveal the other namespace or pod name.
	return fmt.Sprint(c.PropertyName, value, " conflicts with another pod that uses the same volume as this pod with a different ", c.PropertyName, ". If both pods land on the same node, only one of them may access the volume.")
}
