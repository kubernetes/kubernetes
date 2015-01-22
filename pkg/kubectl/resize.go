/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubectl

import (
	"fmt"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

// ResizePrecondition describes a condition that must be true for the resize to take place
// If CurrentSize == -1, it is ignored.
// If CurrentResourceVersion is the empty string, it is ignored.
// Otherwise they must equal the values in the replication controller for it to be valid.
type ResizePrecondition struct {
	Size            int
	ResourceVersion string
}

type PreconditionError struct {
	Precondition  string
	ExpectedValue string
	ActualValue   string
}

func (pe *PreconditionError) Error() string {
	return fmt.Sprintf("Expected %s to be %s, was %s", pe.Precondition, pe.ExpectedValue, pe.ActualValue)
}

// Validate ensures that the preconditions match.  Returns nil if they are valid, an error otherwise
func (precondition *ResizePrecondition) Validate(controller *api.ReplicationController) error {
	if precondition.Size != -1 && controller.Spec.Replicas != precondition.Size {
		return &PreconditionError{"replicas", strconv.Itoa(precondition.Size), strconv.Itoa(controller.Spec.Replicas)}
	}
	if precondition.ResourceVersion != "" && controller.ResourceVersion != precondition.ResourceVersion {
		return &PreconditionError{"resource version", precondition.ResourceVersion, controller.ResourceVersion}
	}
	return nil
}

type Resizer interface {
	Resize(namespace, name string, preconditions *ResizePrecondition, newSize uint) (string, error)
}

func ResizerFor(kind string, c *client.Client) (Resizer, bool) {
	switch kind {
	case "ReplicationController":
		return &ReplicationControllerResizer{c}, true
	}
	return nil, false
}

type ReplicationControllerResizer struct {
	client.Interface
}

func (resize *ReplicationControllerResizer) Resize(namespace, name string, preconditions *ResizePrecondition, newSize uint) (string, error) {
	rc := resize.ReplicationControllers(namespace)
	controller, err := rc.Get(name)
	if err != nil {
		return "", err
	}

	if preconditions != nil {
		if err := preconditions.Validate(controller); err != nil {
			return "", err
		}
	}

	controller.Spec.Replicas = int(newSize)
	// TODO: do retry on 409 errors here?
	if _, err := rc.Update(controller); err != nil {
		return "", err
	}
	// TODO: do a better job of printing objects here.
	return "resized", nil
}
