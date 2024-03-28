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

package admission

import (
	"errors"
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
)

const (
	ErrorReasonAllocation = "ResourceAllocationError"
)

type ResourceAllocationError struct {
	Reason       string
	Resource     string
	NUMAAffinity string
	Err          error
}

func (e ResourceAllocationError) Error() string {
	return fmt.Sprintf("cannot allocate resource %s NUMA affinity %v: %v", e.Resource, e.NUMAAffinity, e.Err)
}

func (e ResourceAllocationError) Type() string {
	return ErrorReasonAllocation
}

func (e ResourceAllocationError) Unwrap() error {
	return e.Err
}

func MakeResourceAllocationError(reason, resource string, needed int, affinity fmt.Stringer, err error) ResourceAllocationError {
	return ResourceAllocationError{
		Reason:       reason,
		Resource:     fmt.Sprintf("%s=%d", resource, needed),
		NUMAAffinity: affinityToString(affinity),
		Err:          err,
	}
}

func MakeMultiResourceAllocationError(reason string, resources v1.ResourceList, affinity fmt.Stringer, err error) ResourceAllocationError {
	return ResourceAllocationError{
		Reason:       reason,
		Resource:     formatRequestedResources(resources),
		NUMAAffinity: affinityToString(affinity),
		Err:          err,
	}
}

func ResourceAllocationFailureEvent(recorder record.EventRecorder, pod *v1.Pod, cntName string, err error) error {
	var ra ResourceAllocationError
	if !errors.As(err, &ra) {
		return err
	}
	recorder.Eventf(pod, v1.EventTypeWarning, ra.Reason, "container %q: %v", cntName, ra)
	return ra.Err
}

func affinityToString(affinity fmt.Stringer) string {
	if affinity == nil {
		return "N/A"
	}
	return affinity.String()
}

func formatRequestedResources(res v1.ResourceList) string {
	items := make([]string, 0, len(res))
	for resName, resQty := range res {
		items = append(items, string(resName)+"="+resQty.String())
	}
	return strings.Join(items, ", ")
}
