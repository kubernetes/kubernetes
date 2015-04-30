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
	"io/ioutil"
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
	c RollingUpdaterClient
	// Namespace for resources
	ns string
	// debugOut is a writer for debugging messages.
	debugOut io.Writer
}

// RollingUpdaterConfig is the configuration for a rolling deployment process.
type RollingUpdaterConfig struct {
	// Out is a writer for progress output.
	Out io.Writer
	// OldRC is an existing controller to be replaced.
	OldRc *api.ReplicationController
	// NewRc is a controller that will take ownership of updated pods (will be
	// created if needed).
	NewRc *api.ReplicationController
	// UpdatePeriod is the time to wait between individual pod updates.
	UpdatePeriod time.Duration
	// Interval is the time to wait between polling controller status after
	// update.
	Interval time.Duration
	// Timeout is the time to wait for controller updates before giving up.
	Timeout time.Duration
	// CleanupPolicy defines the cleanup action to take after the deployment is
	// complete.
	CleanupPolicy RollingUpdaterCleanupPolicy
}

// RollingUpdaterCleanupPolicy is a cleanup action to take after the
// deployment is complete.
type RollingUpdaterCleanupPolicy string

const (
	// DeleteRollingUpdateCleanupPolicy means delete the old controller.
	DeleteRollingUpdateCleanupPolicy RollingUpdaterCleanupPolicy = "Delete"
	// PreserveRollingUpdateCleanupPolicy means keep the old controller.
	PreserveRollingUpdateCleanupPolicy RollingUpdaterCleanupPolicy = "Preserve"
	// RenameRollingUpdateCleanupPolicy means delete the old controller, and rename
	// the new controller to the name of the old controller.
	RenameRollingUpdateCleanupPolicy RollingUpdaterCleanupPolicy = "Rename"
)

// NewRollingUpdater creates a RollingUpdater from a client
func NewRollingUpdater(namespace string, c RollingUpdaterClient, debugOut io.Writer) *RollingUpdater {
	out := debugOut
	if out == nil {
		out = ioutil.Discard
	}
	return &RollingUpdater{
		c:        c,
		ns:       namespace,
		debugOut: out,
	}
}

const (
	sourceIdAnnotation        = kubectlAnnotationPrefix + "update-source-id"
	DesiredReplicasAnnotation = kubectlAnnotationPrefix + "desired-replicas"
)

// Update all pods for a ReplicationController (oldRc) by creating a new
// controller (newRc) with 0 replicas, and synchronously resizing oldRc,newRc
// by 1 until oldRc has 0 replicas and newRc has the original # of desired
// replicas. Cleanup occurs based on a RollingUpdaterCleanupPolicy.
//
// If an update from newRc to oldRc is already in progress, we attempt to
// drive it to completion. If an error occurs at any step of the update, the
// error will be returned.
//
// TODO: make this handle performing a rollback of a partially completed
// rollout.
func (r *RollingUpdater) Update(config *RollingUpdaterConfig) error {
	out := config.Out
	oldRc := config.OldRc
	newRc := config.NewRc
	updatePeriod := config.UpdatePeriod
	interval := config.Interval
	timeout := config.Timeout

	newName := newRc.ObjectMeta.Name
	retry := &RetryParams{interval, timeout}
	waitForReplicas := &RetryParams{interval, timeout}
	if newRc.Spec.Replicas <= 0 {
		return fmt.Errorf("Invalid controller spec for %s; required: > 0 replicas, actual: %s\n", newName, newRc.Spec)
	}
	desired := newRc.Spec.Replicas

	sourceId := ""
	if oldRc != nil {
		sourceId = fmt.Sprintf("%s:%s", oldRc.ObjectMeta.Name, oldRc.ObjectMeta.UID)
	}

	// look for existing newRc, incase this update was previously started but interrupted
	rc, existing, err := r.getExistingNewRc(sourceId, newName)
	if existing {
		fmt.Fprintf(out, "Continuing update with existing controller %s.\n", newName)
		if err != nil {
			return err
		}
		replicas := rc.ObjectMeta.Annotations[DesiredReplicasAnnotation]
		desired, err = strconv.Atoi(replicas)
		if err != nil {
			return fmt.Errorf("Unable to parse annotation for %s: %s=%s",
				newName, DesiredReplicasAnnotation, replicas)
		}
		newRc = rc
	} else {
		fmt.Fprintf(out, "Creating %s\n", newName)
		if newRc.ObjectMeta.Annotations == nil {
			newRc.ObjectMeta.Annotations = map[string]string{}
		}
		newRc.ObjectMeta.Annotations[DesiredReplicasAnnotation] = fmt.Sprintf("%d", desired)
		newRc.ObjectMeta.Annotations[sourceIdAnnotation] = sourceId
		newRc.Spec.Replicas = 0
		newRc, err = r.c.CreateReplicationController(r.ns, newRc)
		if err != nil {
			return err
		}
	}

	// If there's no source RC, so just scale up the new immediately and return
	// early.
	if oldRc == nil {
		return r.resizeAndCleanUp(newRc, newName, desired, retry, waitForReplicas, out)
	}

	oldName := oldRc.ObjectMeta.Name
	// +1, -1 on oldRc, newRc until newRc has desired number of replicas or oldRc has 0 replicas
	for newRc.Spec.Replicas < desired && oldRc.Spec.Replicas != 0 {
		newRc.Spec.Replicas += 1
		oldRc.Spec.Replicas -= 1

		fmt.Fprintf(r.debugOut, "At beginning of loop: %s replicas: %d, %s replicas: %d\n",
			oldName, oldRc.Spec.Replicas,
			newName, newRc.Spec.Replicas)
		fmt.Fprintf(out, "Updating %s replicas: %d, %s replicas: %d\n",
			oldName, oldRc.Spec.Replicas,
			newName, newRc.Spec.Replicas)

		newRc, err = r.resizeAndWait(newRc, retry, waitForReplicas)
		if err != nil {
			return err
		}
		time.Sleep(updatePeriod)
		oldRc, err = r.resizeAndWait(oldRc, retry, waitForReplicas)
		if err != nil {
			return err
		}
		fmt.Fprintf(r.debugOut, "At end of loop: %s replicas: %d, %s replicas: %d\n",
			oldName, oldRc.Spec.Replicas,
			newName, newRc.Spec.Replicas)
	}
	// delete remaining replicas on oldRc
	if oldRc.Spec.Replicas != 0 {
		fmt.Fprintf(out, "Stopping %s replicas: %d -> %d\n",
			oldName, oldRc.Spec.Replicas, 0)
		oldRc.Spec.Replicas = 0
		oldRc, err = r.resizeAndWait(oldRc, retry, waitForReplicas)
		if err != nil {
			return err
		}
	}
	// add remaining replicas on newRc and clean up annotations
	err = r.resizeAndCleanUp(newRc, newName, desired, retry, waitForReplicas, out)
	if err != nil {
		return err
	}

	// Clean up the old RC based on policy.
	switch config.CleanupPolicy {
	case DeleteRollingUpdateCleanupPolicy:
		// delete old rc
		fmt.Fprintf(out, "Update succeeded. Deleting %s\n", oldName)
		return r.c.DeleteReplicationController(r.ns, oldName)
	case RenameRollingUpdateCleanupPolicy:
		// delete old rc
		fmt.Fprintf(out, "Update succeeded. Deleting old controller: %s\n", oldName)
		if err := r.c.DeleteReplicationController(r.ns, oldName); err != nil {
			return err
		}
		fmt.Fprintf(out, "Renaming %s to %s\n", newRc.Name, oldName)
		return r.rename(newRc, oldName)
	case PreserveRollingUpdateCleanupPolicy:
		fmt.Fprintf(out, "Update succeeded. Preserving %s\n", oldName)
		return nil
	default:
		return nil
	}
}

func (r *RollingUpdater) getExistingNewRc(sourceId, name string) (*api.ReplicationController, bool, error) {
	rc, err := r.c.GetReplicationController(r.ns, name)
	if err != nil {
		return nil, false, err
	}

	if _, ok := rc.ObjectMeta.Annotations[DesiredReplicasAnnotation]; !ok {
		return rc, true, fmt.Errorf("Missing %s annotation for controller %s", DesiredReplicasAnnotation, name)
	}

	// If there's no source associated with the RC, don't validate association.
	if len(sourceId) != 0 {
		if source, ok := rc.ObjectMeta.Annotations[sourceIdAnnotation]; !ok || source != sourceId {
			err = fmt.Errorf("Missing/unexpected %s annotation for controller %s: %s", sourceIdAnnotation, name, source)
		}
	}

	return rc, true, err
}

func (r *RollingUpdater) resizeAndWait(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
	resizer, err := ResizerFor("ReplicationController", r.c)
	if err != nil {
		return nil, err
	}
	if err := resizer.Resize(rc.Namespace, rc.Name, uint(rc.Spec.Replicas), &ResizePrecondition{-1, ""}, retry, wait); err != nil {
		return nil, err
	}
	return r.c.GetReplicationController(r.ns, rc.ObjectMeta.Name)
}

func (r *RollingUpdater) updateAndWait(rc *api.ReplicationController, interval, timeout time.Duration) (*api.ReplicationController, error) {
	rc, err := r.c.UpdateReplicationController(r.ns, rc)
	if err != nil {
		return nil, err
	}
	if err = wait.Poll(interval, timeout, r.c.ControllerHasDesiredReplicas(rc)); err != nil {
		return nil, err
	}
	return r.c.GetReplicationController(r.ns, rc.ObjectMeta.Name)
}

func (r *RollingUpdater) rename(rc *api.ReplicationController, newName string) error {
	oldName := rc.Name
	rc.Name = newName
	rc.ResourceVersion = ""

	_, err := r.c.CreateReplicationController(rc.Namespace, rc)
	if err != nil {
		return err
	}
	return r.c.DeleteReplicationController(rc.Namespace, oldName)
}

// resizeAndCleanUp resizes rc up to the desired replicas immediately, and
// then cleans up annotations.
func (r *RollingUpdater) resizeAndCleanUp(rc *api.ReplicationController, newName string,
	desired int, retry *RetryParams, wait *RetryParams, out io.Writer) error {
	var err error
	if rc.Spec.Replicas != desired {
		fmt.Fprintf(out, "Resizing %s replicas: %d -> %d\n",
			newName, rc.Spec.Replicas, desired)
		rc.Spec.Replicas = desired
		rc, err = r.resizeAndWait(rc, retry, wait)
		if err != nil {
			return err
		}
	}
	// Clean up annotations
	if rc, err = r.c.GetReplicationController(r.ns, newName); err != nil {
		return err
	}
	delete(rc.ObjectMeta.Annotations, sourceIdAnnotation)
	delete(rc.ObjectMeta.Annotations, DesiredReplicasAnnotation)
	_, err = r.updateAndWait(rc, interval, timeout)
	return err
}

// RollingUpdaterClient abstracts access to ReplicationControllers.
type RollingUpdaterClient interface {
	GetReplicationController(namespace, name string) (*api.ReplicationController, error)
	UpdateReplicationController(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error)
	CreateReplicationController(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error)
	DeleteReplicationController(namespace, name string) error
	ControllerHasDesiredReplicas(rc *api.ReplicationController) wait.ConditionFunc
}

func NewRollingUpdaterClient(c client.Interface) RollingUpdaterClient {
	return &realRollingUpdaterClient{c}
}

// realRollingUpdaterClient is a RollingUpdaterClient which uses a Kube client.
type realRollingUpdaterClient struct {
	client client.Interface
}

func (c *realRollingUpdaterClient) GetReplicationController(namespace, name string) (*api.ReplicationController, error) {
	return c.client.ReplicationControllers(namespace).Get(name)
}

func (c *realRollingUpdaterClient) UpdateReplicationController(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error) {
	return c.client.ReplicationControllers(namespace).Update(rc)
}

func (c *realRollingUpdaterClient) CreateReplicationController(namespace string, rc *api.ReplicationController) (*api.ReplicationController, error) {
	return c.client.ReplicationControllers(namespace).Create(rc)
}

func (c *realRollingUpdaterClient) DeleteReplicationController(namespace, name string) error {
	return c.client.ReplicationControllers(namespace).Delete(name)
}

func (c *realRollingUpdaterClient) ControllerHasDesiredReplicas(rc *api.ReplicationController) wait.ConditionFunc {
	return client.ControllerHasDesiredReplicas(c.client, rc)
}
