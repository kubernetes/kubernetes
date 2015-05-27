/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	goerrors "errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
)

// RollingUpdater provides methods for updating replicated pods in a predictable,
// fault-tolerant way.
type RollingUpdater struct {
	// Client interface for creating and updating controllers
	c RollingUpdaterClient
	// Namespace for resources
	ns string
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

func LoadExistingNextReplicationController(c *client.Client, namespace, newName string) (*api.ReplicationController, error) {
	if len(newName) == 0 {
		return nil, nil
	}
	newRc, err := c.ReplicationControllers(namespace).Get(newName)
	if err != nil && errors.IsNotFound(err) {
		return nil, nil
	}
	return newRc, err
}

func CreateNewControllerFromCurrentController(c *client.Client, namespace, oldName, newName, image, deploymentKey string) (*api.ReplicationController, error) {
	// load the old RC into the "new" RC
	newRc, err := c.ReplicationControllers(namespace).Get(oldName)
	if err != nil {
		return nil, err
	}

	if len(newRc.Spec.Template.Spec.Containers) > 1 {
		// TODO: support multi-container image update.
		return nil, goerrors.New("Image update is not supported for multi-container pods")
	}
	if len(newRc.Spec.Template.Spec.Containers) == 0 {
		return nil, goerrors.New(fmt.Sprintf("Pod has no containers! (%v)", newRc))
	}
	newRc.Spec.Template.Spec.Containers[0].Image = image

	newHash, err := api.HashObject(newRc, c.Codec)
	if err != nil {
		return nil, err
	}

	if len(newName) == 0 {
		newName = fmt.Sprintf("%s-%s", newRc.Name, newHash)
	}
	newRc.Name = newName

	newRc.Spec.Selector[deploymentKey] = newHash
	newRc.Spec.Template.Labels[deploymentKey] = newHash
	// Clear resource version after hashing so that identical updates get different hashes.
	newRc.ResourceVersion = ""
	return newRc, nil
}

// NewRollingUpdater creates a RollingUpdater from a client
func NewRollingUpdater(namespace string, c RollingUpdaterClient) *RollingUpdater {
	return &RollingUpdater{
		c,
		namespace,
	}
}

const (
	sourceIdAnnotation        = kubectlAnnotationPrefix + "update-source-id"
	desiredReplicasAnnotation = kubectlAnnotationPrefix + "desired-replicas"
	nextControllerAnnotation  = kubectlAnnotationPrefix + "next-controller-id"
)

func AbortRollingUpdate(c *RollingUpdaterConfig) {
	// Swap the controllers
	tmp := c.OldRc
	c.OldRc = c.NewRc
	c.NewRc = tmp

	if c.NewRc.Annotations == nil {
		c.NewRc.Annotations = map[string]string{}
	}
	c.NewRc.Annotations[sourceIdAnnotation] = fmt.Sprintf("%s:%s", c.OldRc.Name, c.OldRc.UID)
	desiredSize, found := c.OldRc.Annotations[desiredReplicasAnnotation]
	if found {
		fmt.Printf("Found desired replicas.")
		c.NewRc.Annotations[desiredReplicasAnnotation] = desiredSize
	}
	c.CleanupPolicy = DeleteRollingUpdateCleanupPolicy
}

func GetNextControllerAnnotation(rc *api.ReplicationController) (string, bool) {
	res, found := rc.Annotations[nextControllerAnnotation]
	return res, found
}

func SetNextControllerAnnotation(rc *api.ReplicationController, name string) {
	if rc.Annotations == nil {
		rc.Annotations = map[string]string{}
	}
	rc.Annotations[nextControllerAnnotation] = name
}

func UpdateExistingReplicationController(c client.Interface, oldRc *api.ReplicationController, namespace, newName, deploymentKey, deploymentValue string, out io.Writer) (*api.ReplicationController, error) {
	SetNextControllerAnnotation(oldRc, newName)
	if _, found := oldRc.Spec.Selector[deploymentKey]; !found {
		return AddDeploymentKeyToReplicationController(oldRc, c, deploymentKey, deploymentValue, namespace, out)
	} else {
		// If we didn't need to update the controller for the deployment key, we still need to write
		// the "next" controller.
		return c.ReplicationControllers(namespace).Update(oldRc)
	}
}

const MaxRetries = 3

func AddDeploymentKeyToReplicationController(oldRc *api.ReplicationController, client client.Interface, deploymentKey, deploymentValue, namespace string, out io.Writer) (*api.ReplicationController, error) {
	var err error
	// First, update the template label.  This ensures that any newly created pods will have the new label
	if oldRc, err = updateWithRetries(client.ReplicationControllers(namespace), oldRc, func(rc *api.ReplicationController) {
		if rc.Spec.Template.Labels == nil {
			rc.Spec.Template.Labels = map[string]string{}
		}
		rc.Spec.Template.Labels[deploymentKey] = deploymentValue
	}); err != nil {
		return nil, err
	}

	// Update all pods managed by the rc to have the new hash label, so they are correctly adopted
	// TODO: extract the code from the label command and re-use it here.
	podList, err := client.Pods(namespace).List(labels.SelectorFromSet(oldRc.Spec.Selector), fields.Everything())
	if err != nil {
		return nil, err
	}
	for ix := range podList.Items {
		pod := &podList.Items[ix]
		if pod.Labels == nil {
			pod.Labels = map[string]string{
				deploymentKey: deploymentValue,
			}
		} else {
			pod.Labels[deploymentKey] = deploymentValue
		}
		err = nil
		delay := 3
		for i := 0; i < MaxRetries; i++ {
			_, err = client.Pods(namespace).Update(pod)
			if err != nil {
				fmt.Fprintf(out, "Error updating pod (%v), retrying after %d seconds", err, delay)
				time.Sleep(time.Second * time.Duration(delay))
				delay *= delay
			} else {
				break
			}
		}
		if err != nil {
			return nil, err
		}
	}

	if oldRc.Spec.Selector == nil {
		oldRc.Spec.Selector = map[string]string{}
	}
	// Copy the old selector, so that we can scrub out any orphaned pods
	selectorCopy := map[string]string{}
	for k, v := range oldRc.Spec.Selector {
		selectorCopy[k] = v
	}
	oldRc.Spec.Selector[deploymentKey] = deploymentValue

	// Update the selector of the rc so it manages all the pods we updated above
	if oldRc, err = updateWithRetries(client.ReplicationControllers(namespace), oldRc, func(rc *api.ReplicationController) {
		rc.Spec.Selector[deploymentKey] = deploymentValue
	}); err != nil {
		return nil, err
	}

	// Clean up any orphaned pods that don't have the new label, this can happen if the rc manager
	// doesn't see the update to its pod template and creates a new pod with the old labels after
	// we've finished re-adopting existing pods to the rc.
	podList, err = client.Pods(namespace).List(labels.SelectorFromSet(selectorCopy), fields.Everything())
	for ix := range podList.Items {
		pod := &podList.Items[ix]
		if value, found := pod.Labels[deploymentKey]; !found || value != deploymentValue {
			if err := client.Pods(namespace).Delete(pod.Name, nil); err != nil {
				return nil, err
			}
		}
	}

	return oldRc, nil
}

type updateFunc func(controller *api.ReplicationController)

// updateWithRetries updates applies the given rc as an update.
func updateWithRetries(rcClient client.ReplicationControllerInterface, rc *api.ReplicationController, applyUpdate updateFunc) (*api.ReplicationController, error) {
	// Each update could take ~100ms, so give it 0.5 second
	var err error
	oldRc := rc
	err = wait.Poll(10*time.Millisecond, 500*time.Millisecond, func() (bool, error) {
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(rc)
		if rc, err = rcClient.Update(rc); err == nil {
			// rc contains the latest controller post update
			return true, nil
		}
		// Update the controller with the latest resource version, if the update failed we
		// can't trust rc so use oldRc.Name.
		if rc, err = rcClient.Get(oldRc.Name); err != nil {
			// The Get failed: Value in rc cannot be trusted.
			rc = oldRc
		}
		// The Get passed: rc contains the latest controller, expect a poll for the update.
		return false, nil
	})
	// If the error is non-nil the returned controller cannot be trusted, if it is nil, the returned
	// controller contains the applied update.
	return rc, err
}

func FindSourceController(r RollingUpdaterClient, namespace, name string) (*api.ReplicationController, error) {
	list, err := r.ListReplicationControllers(namespace, labels.Everything())
	if err != nil {
		return nil, err
	}
	for ix := range list.Items {
		rc := &list.Items[ix]
		if rc.Annotations != nil && strings.HasPrefix(rc.Annotations[sourceIdAnnotation], name) {
			return rc, nil
		}
	}
	return nil, fmt.Errorf("couldn't find a replication controller with source id == %s/%s", namespace, name)
}

// Update all pods for a ReplicationController (oldRc) by creating a new
// controller (newRc) with 0 replicas, and synchronously scaling oldRc,newRc
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

	oldName := oldRc.ObjectMeta.Name
	newName := newRc.ObjectMeta.Name
	retry := &RetryParams{interval, timeout}
	waitForReplicas := &RetryParams{interval, timeout}
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
		newRc, err = r.c.CreateReplicationController(r.ns, newRc)
		if err != nil {
			return err
		}
	}

	// +1, -1 on oldRc, newRc until newRc has desired number of replicas or oldRc has 0 replicas
	for newRc.Spec.Replicas < desired && oldRc.Spec.Replicas != 0 {
		newRc.Spec.Replicas += 1
		oldRc.Spec.Replicas -= 1
		fmt.Printf("At beginning of loop: %s replicas: %d, %s replicas: %d\n",
			oldName, oldRc.Spec.Replicas,
			newName, newRc.Spec.Replicas)
		fmt.Fprintf(out, "Updating %s replicas: %d, %s replicas: %d\n",
			oldName, oldRc.Spec.Replicas,
			newName, newRc.Spec.Replicas)

		newRc, err = r.scaleAndWait(newRc, retry, waitForReplicas)
		if err != nil {
			return err
		}
		time.Sleep(updatePeriod)
		oldRc, err = r.scaleAndWait(oldRc, retry, waitForReplicas)
		if err != nil {
			return err
		}
		fmt.Printf("At end of loop: %s replicas: %d, %s replicas: %d\n",
			oldName, oldRc.Spec.Replicas,
			newName, newRc.Spec.Replicas)
	}
	// delete remaining replicas on oldRc
	if oldRc.Spec.Replicas != 0 {
		fmt.Fprintf(out, "Stopping %s replicas: %d -> %d\n",
			oldName, oldRc.Spec.Replicas, 0)
		oldRc.Spec.Replicas = 0
		oldRc, err = r.scaleAndWait(oldRc, retry, waitForReplicas)
		// oldRc, err = r.scaleAndWait(oldRc, interval, timeout)
		if err != nil {
			return err
		}
	}
	// add remaining replicas on newRc
	if newRc.Spec.Replicas != desired {
		fmt.Fprintf(out, "Scaling %s replicas: %d -> %d\n",
			newName, newRc.Spec.Replicas, desired)
		newRc.Spec.Replicas = desired
		newRc, err = r.scaleAndWait(newRc, retry, waitForReplicas)
		if err != nil {
			return err
		}
	}
	// Clean up annotations
	if newRc, err = r.c.GetReplicationController(r.ns, newName); err != nil {
		return err
	}
	delete(newRc.ObjectMeta.Annotations, sourceIdAnnotation)
	delete(newRc.ObjectMeta.Annotations, desiredReplicasAnnotation)
	newRc, err = r.updateAndWait(newRc, interval, timeout)
	if err != nil {
		return err
	}

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
		return nil
	default:
		return nil
	}
}

func (r *RollingUpdater) getExistingNewRc(sourceId, name string) (rc *api.ReplicationController, existing bool, err error) {
	if rc, err = r.c.GetReplicationController(r.ns, name); err == nil {
		existing = true
		annotations := rc.ObjectMeta.Annotations
		source := annotations[sourceIdAnnotation]
		_, ok := annotations[desiredReplicasAnnotation]
		if source != sourceId || !ok {
			err = fmt.Errorf("Missing/unexpected annotations for controller %s, expected %s : %s", name, sourceId, annotations)
		}
		return
	}
	err = nil
	return
}

func (r *RollingUpdater) scaleAndWait(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
	scaler, err := ScalerFor("ReplicationController", r.c)
	if err != nil {
		return nil, err
	}
	if err := scaler.Scale(rc.Namespace, rc.Name, uint(rc.Spec.Replicas), &ScalePrecondition{-1, ""}, retry, wait); err != nil {
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
	return Rename(r.c, rc, newName)
}

func Rename(c RollingUpdaterClient, rc *api.ReplicationController, newName string) error {
	oldName := rc.Name
	rc.Name = newName
	rc.ResourceVersion = ""

	_, err := c.CreateReplicationController(rc.Namespace, rc)
	if err != nil {
		return err
	}
	err = c.DeleteReplicationController(rc.Namespace, oldName)
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	return nil
}

// RollingUpdaterClient abstracts access to ReplicationControllers.
type RollingUpdaterClient interface {
	ListReplicationControllers(namespace string, selector labels.Selector) (*api.ReplicationControllerList, error)
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

func (c *realRollingUpdaterClient) ListReplicationControllers(namespace string, selector labels.Selector) (*api.ReplicationControllerList, error) {
	return c.client.ReplicationControllers(namespace).List(selector)
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
