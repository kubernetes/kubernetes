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
	"io"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
)

// RollingUpdater provides methods for updating replicated pods in a predictable,
// fault-tolerant way.
type RollingUpdater struct {
	// Client interface for creating and updating controllers
	c client.Interface
	// Namespace for resources
	ns string
}

// NewRollingUpdater creates a RollingUpdater from a client
func NewRollingUpdater(namespace string, c client.Interface) *RollingUpdater {
	return &RollingUpdater{
		c,
		namespace,
	}
}

const (
	sourceIdAnnotation        = kubectlAnnotationPrefix + "update-source-id"
	desiredReplicasAnnotation = kubectlAnnotationPrefix + "desired-replicas"
)

// Update all pods for a ReplicationController (oldRc) by creating a new controller (newRc)
// with 0 replicas, and synchronously resizing oldRc,newRc by 1 until oldRc has 0 replicas
// and newRc has the original # of desired replicas. oldRc is then deleted.
// If an update from newRc to oldRc is already in progress, we attempt to drive it to completion.
// If an error occurs at any step of the update, the error will be returned.
//  'out' writer for progress output
//  'oldRc' existing controller to be replaced
//  'newRc' controller that will take ownership of updated pods (will be created if needed)
//  'updatePeriod' time to wait between individual pod updates
//  'interval' time to wait between polling controller status after update
//  'timeout' time to wait for controller updates before giving up
//
// TODO: make this handle performing a rollback of a partially completed rollout.
func (r *RollingUpdater) Update(out io.Writer, oldRc, newRc *api.ReplicationController, updatePeriod, interval, timeout time.Duration) error {
	oldName := oldRc.ObjectMeta.Name
	newName := newRc.ObjectMeta.Name

	if newRc.Spec.Replicas <= 0 {
		return fmt.Errorf("Invalid controller spec for %s; required: > 0 replicas, actual: %s\n", newName, newRc.Spec)
	}
	desired := newRc.Spec.Replicas
	sourceId := fmt.Sprintf("%s:%s", oldName, oldRc.ObjectMeta.UID)

	// look for existing newRc, incase this update was previously started but interrupted
	rc, existing, err := r.getExistingNewRc(sourceId, newName)
	if existing {
		fmt.Fprintf(out, "Continuing update with existing controller %s.\n", newName)
		if err != nil {
			return err
		}
		replicas := rc.ObjectMeta.Annotations[desiredReplicasAnnotation]
		desired, err = strconv.Atoi(replicas)
		if err != nil {
			return fmt.Errorf("Unable to parse annotation for %s: %s=%s",
				newName, desiredReplicasAnnotation, replicas)
		}
		newRc = rc
	} else {
		fmt.Fprintf(out, "Creating %s\n", newName)
		if newRc.ObjectMeta.Annotations == nil {
			newRc.ObjectMeta.Annotations = map[string]string{}
		}
		newRc.ObjectMeta.Annotations[desiredReplicasAnnotation] = fmt.Sprintf("%d", desired)
		newRc.ObjectMeta.Annotations[sourceIdAnnotation] = sourceId
		newRc.Spec.Replicas = 0
		newRc, err = r.c.ReplicationControllers(r.ns).Create(newRc)
		if err != nil {
			return err
		}
	}

	// +1, -1 on oldRc, newRc until newRc has desired number of replicas or oldRc has 0 replicas
	for newRc.Spec.Replicas < desired && oldRc.Spec.Replicas != 0 {
		newRc.Spec.Replicas += 1
		oldRc.Spec.Replicas -= 1
		fmt.Fprintf(out, "Updating %s replicas: %d, %s replicas: %d\n",
			oldName, oldRc.Spec.Replicas,
			newName, newRc.Spec.Replicas)

		newRc, err = r.updateAndWait(newRc, interval, timeout)
		if err != nil {
			return err
		}
		time.Sleep(updatePeriod)
		oldRc, err = r.updateAndWait(oldRc, interval, timeout)
		if err != nil {
			return err
		}
	}
	// delete remaining replicas on oldRc
	if oldRc.Spec.Replicas != 0 {
		fmt.Fprintf(out, "Stopping %s replicas: %d -> %d\n",
			oldName, oldRc.Spec.Replicas, 0)
		oldRc.Spec.Replicas = 0
		oldRc, err = r.updateAndWait(oldRc, interval, timeout)
		if err != nil {
			return err
		}
	}
	// add remaining replicas on newRc, cleanup annotations
	if newRc.Spec.Replicas != desired {
		fmt.Fprintf(out, "Resizing %s replicas: %d -> %d\n",
			newName, newRc.Spec.Replicas, desired)
		newRc.Spec.Replicas = desired
	}
	delete(newRc.ObjectMeta.Annotations, sourceIdAnnotation)
	delete(newRc.ObjectMeta.Annotations, desiredReplicasAnnotation)
	newRc, err = r.updateAndWait(newRc, interval, timeout)
	if err != nil {
		return err
	}
	// delete old rc
	fmt.Fprintf(out, "Update succeeded. Deleting %s\n", oldName)
	return r.c.ReplicationControllers(r.ns).Delete(oldName)
}

func (r *RollingUpdater) getExistingNewRc(sourceId, name string) (rc *api.ReplicationController, existing bool, err error) {
	if rc, err = r.c.ReplicationControllers(r.ns).Get(name); err == nil {
		existing = true
		annotations := rc.ObjectMeta.Annotations
		source := annotations[sourceIdAnnotation]
		_, ok := annotations[desiredReplicasAnnotation]
		if source != sourceId || !ok {
			err = fmt.Errorf("Missing/unexpected annotations for controller %s: %s", name, annotations)
		}
		return
	}
	err = nil
	return
}

func (r *RollingUpdater) updateAndWait(rc *api.ReplicationController, interval, timeout time.Duration) (*api.ReplicationController, error) {
	rc, err := r.c.ReplicationControllers(r.ns).Update(rc)
	if err != nil {
		return nil, err
	}
	if err := wait.Poll(interval, timeout,
		client.ControllerHasDesiredReplicas(r.c, rc)); err != nil {
		return nil, err
	}
	return r.c.ReplicationControllers(r.ns).Get(rc.ObjectMeta.Name)
}
