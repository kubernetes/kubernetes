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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	desiredNodeAnnotation = kubectlAnnotationPrefix + "desired-nodes"
)

// RollingUpdaterByNodeConfig is the configuration for a rolling deployment process.
type RollingUpdaterByNodeConfig struct {
	// Out is a writer for progress output.
	Out io.Writer
	// OldRC is an existing controller to be replaced.
	OldRc *api.ReplicationController
	// NewRc is a controller that will take ownership of updated pods (will be
	// created if needed).
	NewRc *api.ReplicationController
	// NodeLabel
	NodeLabel string
	// UpdatePeriod is the time to wait between individual pod updates.
	UpdatePeriod time.Duration
	// Interval is the time to wait between polling controller status after
	// update.
	Interval time.Duration
	// Timeout is the time to wait for controller updates before giving up.
	Timeout time.Duration
	// CleanupPolicy defines the cleanup action to take after the deployment is
	// complete.
	CleanupPolicy RollingUpdaterByNodeCleanupPolicy
}

// RollingUpdaterByNodeCleanupPolicy is a cleanup action to take after the
// deployment is complete.
type RollingUpdaterByNodeCleanupPolicy string

const (
	// DeleteRollingUpdateByNodeCleanupPolicy means delete the old controller.
	DeleteRollingUpdateByNodeCleanupPolicy RollingUpdaterByNodeCleanupPolicy = "Delete"
	// PreserveRollingUpdateByNodeCleanupPolicy means keep the old controller.
	PreserveRollingUpdateByNodeCleanupPolicy RollingUpdaterByNodeCleanupPolicy = "Preserve"
	// RenameRollingUpdateByNodeCleanupPolicy means delete the old controller, and rename
	// the new controller to the name of the old controller.
	RenameRollingUpdateByNodeCleanupPolicy RollingUpdaterByNodeCleanupPolicy = "Rename"
)

// RollingUpdaterByNode provides methods for updating replicated pods in a predictable,
// fault-tolerant way.
type RollingUpdaterByNode struct {
	// Client interface for creating and updating controllers
	c client.Interface
	// Namespace for resources
	ns string
	// scaleAndWait scales a controller and returns its updated state.
	scaleAndWait func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error)
	//getOrCreateTargetController gets and validates an existing controller or
	//makes a new one.
	getOrCreateTargetController func(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error)
	// cleanup performs post deployment cleanup tasks for newRc and oldRc.
	cleanup func(oldRc, newRc *api.ReplicationController, config *RollingUpdaterByNodeConfig) error
	// getReadyPods returns the amount of old and new ready pods.
	getReadyPods func(oldRc, newRc *api.ReplicationController) (int, int, error)
}

// NewRollingUpdaterByNode creates a RollingUpdaterByNode from a client.
func NewRollingUpdaterByNode(namespace string, client client.Interface) *RollingUpdaterByNode {
	updater := &RollingUpdaterByNode{
		c:  client,
		ns: namespace,
	}
	// Inject real implementations.
	updater.scaleAndWait = updater.scaleAndWaitWithScaler
	updater.getOrCreateTargetController = updater.getOrCreateTargetControllerWithClient
	updater.getReadyPods = updater.readyPods
	updater.cleanup = updater.cleanupWithClients
	return updater
}

// Update all pods for a ReplicationController (oldRc) by creating a new
// controller (newRc) with 0 replicas, and synchronously scaling oldRc and
// newRc until oldRc has 0 replicas and newRc has the original # of desired
// replicas. Cleanup occurs based on a RollingUpdaterByNodeCleanupPolicy.
//
// If an update from newRc to oldRc is already in progress, we attempt to
// drive it to completion. If an error occurs at any step of the update, the
// error will be returned.
//
// A scaling event (either up or down) is considered progress; if no progress
// is made within the config.Timeout, an error is returned.
func (r *RollingUpdaterByNode) Update(config *RollingUpdaterByNodeConfig) error {
	out := config.Out
	oldRc := config.OldRc
	scaleRetryParams := NewRetryParams(config.Interval, config.Timeout)

	///////////////
	//  Node Label checks
	///////////////////
	rollingUpdateLabel := config.NodeLabel
	if config.NodeLabel == "" {
		return goerrors.New(fmt.Sprintf("NodeLabel parameter can not be empty"))
	}
	// Check if the old RC has the rollingUpdateLabel
	oldRollingUpdateLabelValue, ok := oldRc.Spec.Template.Spec.NodeSelector[rollingUpdateLabel]
	if !ok {
		return goerrors.New(fmt.Sprintf("RC '%s'; have not NodeSelector: '%s'", oldRc.Name, rollingUpdateLabel))
	}
	// Check if the new RC has the rollingUpdateLabel
	newRollingUpdateLabelValue, ok := config.NewRc.Spec.Template.Spec.NodeSelector[rollingUpdateLabel]
	if !ok {
		return goerrors.New(fmt.Sprintf("RC '%s'; have not NodeSelector: '%s'", config.NewRc.Name, rollingUpdateLabel))
	}
	// Compare old label value with new label value
	// error if equal
	// We can not roolling update by node if equal
	// Check if the old RC has the rollingUpdateLabel
	if oldRollingUpdateLabelValue == newRollingUpdateLabelValue {
		return goerrors.New(fmt.Sprintf("Old RC '%s' and new RC '%s'; have same value for label: '%s'", oldRc.Name, config.NewRc.Name, rollingUpdateLabel))
	}

	////////////////////
	//   Newrc creation
	////////////////////
	// Find an existing controller (for continuing an interrupted update) or
	// create a new one if necessary.
	sourceId := fmt.Sprintf("%s:%s", oldRc.Name, oldRc.UID)
	newRc, existed, err := r.getOrCreateTargetController(config.NewRc, sourceId)
	if err != nil {
		return err
	}
	if existed {
		fmt.Fprintf(out, "Continuing update with existing controller %s.\n", newRc.Name)
	} else {
		fmt.Fprintf(out, "Created %s\n", newRc.Name)
	}
	// Set node annotation key
	nodeAnnotationKey := desiredNodeAnnotation + "-" + oldRc.Namespace + "-" + rollingUpdateLabel
	// Extract the desired replica count from the controller.
	desired, err := strconv.Atoi(newRc.Annotations[desiredReplicasAnnotation])
	if err != nil {
		return fmt.Errorf("Unable to parse annotation for %s: %s=%s",
			newRc.Name, desiredReplicasAnnotation, newRc.Annotations[desiredReplicasAnnotation])
	}
	// Extract the original replica count from the old controller, adding the
	// annotation if it doesn't yet exist.
	_, hasOriginalAnnotation := oldRc.Annotations[originalReplicasAnnotation]
	if !hasOriginalAnnotation {
		existing, err := r.c.ReplicationControllers(oldRc.Namespace).Get(oldRc.Name)
		if err != nil {
			return err
		}
		if existing.Annotations == nil {
			existing.Annotations = map[string]string{}
		}
		existing.Annotations[originalReplicasAnnotation] = strconv.Itoa(existing.Spec.Replicas)
		updated, err := r.c.ReplicationControllers(existing.Namespace).Update(existing)
		if err != nil {
			return err
		}
		oldRc = updated
	}

	_, hasOriginalAnnotation = oldRc.Annotations[originalReplicasAnnotation]
	if !hasOriginalAnnotation {
	}
	original, err := strconv.Atoi(oldRc.Annotations[originalReplicasAnnotation])
	if err != nil {
		return fmt.Errorf("Unable to parse annotation for %s: %s=%s\n",
			oldRc.Name, originalReplicasAnnotation, oldRc.Annotations[originalReplicasAnnotation])
	}
	//////////////////
	// Get node lists
	////////////////
	// Node label selector
	label := labels.SelectorFromSet(labels.Set(map[string]string{rollingUpdateLabel: oldRollingUpdateLabelValue}))
	listOptions := api.ListOptions{
		LabelSelector: label,
		FieldSelector: fields.Everything(),
	}
	// Get nodes
	nodeList, err := r.c.Nodes().List(listOptions)
	if err != nil {
		return err
	}
	if len(nodeList.Items) == 0 && !existed {
		return fmt.Errorf("No node with label '%s' found", rollingUpdateLabel)
	}

	///////////////////////
	//  Count old pods by node
	///////////////////////
	// Prepare labels and fields for old pods
	oldPodsLabel := labels.SelectorFromSet(oldRc.Spec.Selector)
	allOldPodsListOptions := api.ListOptions{
		LabelSelector: oldPodsLabel,
		FieldSelector: fields.Everything(),
	}

	// Check if annotations (for pod number) are set on each node
	annotationsSet := true
	for _, node := range nodeList.Items {
		if _, exists := node.Annotations[nodeAnnotationKey]; !exists {
			// need to set annotations
			annotationsSet = false
		}
	}
	if !annotationsSet {
		// Get the number of pod running the old version
		podList, err := r.c.Pods(oldRc.Namespace).List(allOldPodsListOptions)
		if err != nil {
			return err
		}
		// Get nb of old pods by nodes
		nbPodByNode := make(map[string]int)
		for _, pod := range podList.Items {
			if _, ok := nbPodByNode[pod.Spec.NodeName]; !ok {
				nbPodByNode[pod.Spec.NodeName] = 0
			}
			// Increments pod number
			nbPodByNode[pod.Spec.NodeName] += 1
		}

		// Set annotation on each node
		for _, node := range nodeList.Items {
			// Get last version of the current node
			node, err := r.c.Nodes().Get(node.Name)
			// The controller wasn't found, so create it.
			if node.Annotations == nil {
				node.Annotations = map[string]string{}
			}
			// Update annotations
			node.Annotations[nodeAnnotationKey] = strconv.Itoa(nbPodByNode[node.Name])
			node, err = r.c.Nodes().Update(node)
			if err != nil {
				return err
			}
		}
	} else {
		// If existed we have to add node with rolling update label
		// in the node list
		// Node label selector
		label := labels.SelectorFromSet(labels.Set(map[string]string{rollingUpdateLabel: "v0.0.0"}))
		listOptions := api.ListOptions{
			LabelSelector: label,
			FieldSelector: fields.Everything(),
		}
		// Get nodes
		oldNodeList, err := r.c.Nodes().List(listOptions)
		if err != nil {
			return err
		}
		nodeList.Items = append(oldNodeList.Items, nodeList.Items...)
	}

	fmt.Fprintf(out, "Scaling up %s from %d to %d, scaling down %s from %d to 0 (Node by Node)\n",
		newRc.Name, newRc.Spec.Replicas, desired, oldRc.Name, oldRc.Spec.Replicas)

	////////////////////
	// Rolling update
	////////////////////////////

	// Set duration in seconds before the object should be deleted
	podsDeleteOptions := api.NewDeleteOptions(int64(5))

	for _, node := range nodeList.Items {
		// Get last version of the current node
		node, err := r.c.Nodes().Get(node.Name)
		fmt.Fprintf(out, "Rolling update by node starting on node %s.\n", node.Name)
		// Set label nodelabel to v0.0.0 to the current node
		// TODO replace "v0.0.0" by random stuff ???
		node.ObjectMeta.Labels[rollingUpdateLabel] = "v0.0.0"
		node, err = r.c.Nodes().Update(node)
		if err != nil {
			return err
		}

		// Get Number of pods wanted on the current Node
		var oldPodNumber int
		_, ok = node.Annotations[nodeAnnotationKey]
		if !ok {
			oldPodNumber = 0
		} else {
			oldPodNumber, err = strconv.Atoi(node.Annotations[nodeAnnotationKey])
			if err != nil {
				return fmt.Errorf("Unable to parse annotation node for %s: %s=%s",
					node.Name, nodeAnnotationKey, node.Annotations[nodeAnnotationKey])
			}
		}

		// Get all pods from current nodes and current RC
		podsFields := fields.Set{"spec.nodeName": node.Name}
		oldPodsListOptions := api.ListOptions{
			LabelSelector: oldPodsLabel,
			FieldSelector: podsFields.AsSelector(),
		}
		podList, err := r.c.Pods(oldRc.Namespace).List(oldPodsListOptions)
		if err != nil {
			return err
		}
		/////////////////////////////
		// Delete pod and wait deletion
		/////////////////////////////
		// Delete old pods
		nbPods := 0

		for _, pod := range podList.Items {
			// Counting nb of pods running on this node
			nbPods += 1
			// Delete pod from the current node
			r.c.Pods(oldRc.Namespace).Delete(pod.Name, podsDeleteOptions)
			// Scale down one by one for each pod deleted
			if nbPods <= oldPodNumber {
				tmpOldRc, err := r.scaleDownByOne(oldRc, config)
				if err != nil {
					fmt.Fprintf(out, "Scaling Down by one RC %s, error\n", oldRc.Name, err)
				} else {
					oldRc = tmpOldRc
				}
			}
		}
		if len(podList.Items) > 0 {
			// Waiting for pods deletion
			// TODO add timeout
			watcher, _ := r.c.Pods(oldRc.Namespace).Watch(oldPodsListOptions)
			if watcher != nil {
				for nbPods > 0 {
					// Waiting for events
					<-watcher.ResultChan()
					// init counter
					nbPods = 0
					// Counting pod deleted still running on this node
					podList, _ = r.c.Pods(oldRc.Namespace).List(oldPodsListOptions)
					nbPods = len(podList.Items)
				}
			}
		}
		// Here all pods (from old RC) one the current node are deleted

		//////////////////////////////
		/// Label node with the label of the new RC
		/////////////////////
		// Get last version of the current node
		node, err = r.c.Nodes().Get(node.Name)
		// Set the new label to the current node
		node.ObjectMeta.Labels[rollingUpdateLabel] = newRollingUpdateLabelValue
		node, err = r.c.Nodes().Update(node)

		////////////////////
		//  Scale and wait
		///////////////////
		// Increase the number of replicas of the new Replication Controller
		if oldPodNumber > 0 {
			desiredOnNode := oldPodNumber
			fmt.Fprintf(config.Out, "Scaling %s up to %d\n", newRc.Name, newRc.Spec.Replicas)
			scaledRc, err := r.scaleUp(newRc, oldRc, original, desired, desiredOnNode, scaleRetryParams, config)
			if err != nil {
				return err
			}
			newRc = scaledRc
		}

		// Update finished on the current node
		fmt.Fprintf(out, "Rolling update by node finished on node %s.\n", node.Name)

		// Go to the next one
	}
	// Rolling Update by Node finished

	// Scale Up newReplicasNumber is too low
	if newRc.Spec.Replicas < desired {
		fmt.Fprintf(out, "Final scaling Up Replication Controller %s to %d.\n", newRc.Name, desired)
		scaledRc, err := r.scaleUp(newRc, oldRc, original, desired, desired, scaleRetryParams, config)
		if err != nil {
			return err
		}
		newRc = scaledRc
	}

	// Scale down to 0 Old Rc if not equal to 0
	if oldRc.Spec.Replicas > 0 {
		oldRc.Spec.Replicas = 0
		fmt.Fprintf(out, "Final scaling Down Replication Controller %s to 0.\n", oldRc.Name)
		_, err := r.scaleAndWait(oldRc, scaleRetryParams, scaleRetryParams)
		if err != nil {
			return err
		}
	}

	// Remove node annotations
	// Get all nodes
	nodeList, err = r.c.Nodes().List(api.ListOptions{})
	if err != nil {
		return err
	}
	for _, node := range nodeList.Items {
		if _, ok := node.Annotations[nodeAnnotationKey]; ok {
			// Remove annotation on the current node
			currentNode, err := r.c.Nodes().Get(node.Name)
			// Update annotations
			delete(currentNode.Annotations, nodeAnnotationKey)

			fmt.Fprintf(out, "Delete annotation %s on node %s\n", nodeAnnotationKey, node.Name)
			_, err = r.c.Nodes().Update(currentNode)
			if err != nil {
				return err
			}
		}
	}

	// Housekeeping and cleanup policy execution.
	return r.cleanup(oldRc, newRc, config)
}

// scaleUp scales up newRc to desired by whatever increment is possible given
// the configured surge threshold. scaleUp will safely no-op as necessary when
// it detects redundancy or other relevant conditions.
func (r *RollingUpdaterByNode) scaleUp(newRc, oldRc *api.ReplicationController, original, desired, desiredOnNode int, scaleRetryParams *RetryParams, config *RollingUpdaterByNodeConfig) (*api.ReplicationController, error) {
	// If we're already at the desired, do nothing.
	if newRc.Spec.Replicas == desired {
		return newRc, nil
	}

	// Scale up as far as we can based on the surge limit.
	increment := desiredOnNode
	// If the old is already scaled down, go ahead and scale all the way up.
	// We can't scale up without violating the surge limit, so do nothing.
	if increment <= 0 {
		return newRc, nil
	}
	// Increase the replica count, and deal with fenceposts.
	newRc.Spec.Replicas += increment
	if newRc.Spec.Replicas > desired {
		newRc.Spec.Replicas = desired
	}
	// Perform the scale-up.
	fmt.Fprintf(config.Out, "Scaling %s up to %d\n", newRc.Name, newRc.Spec.Replicas)
	scaledRc, err := r.scaleAndWait(newRc, scaleRetryParams, scaleRetryParams)
	if err != nil {
		return nil, err
	}
	newRc = scaledRc

	// We have to wait that all pods are RUNNING on the current node
	// and for each events check the number of NON RUNNING pods on the current node
	newPodsLabel := labels.SelectorFromSet(newRc.Spec.Selector)
	NewPodsListOptions := api.ListOptions{
		LabelSelector: newPodsLabel,
		FieldSelector: fields.Everything(),
	}
	watcher, _ := r.c.Pods(newRc.Namespace).Watch(NewPodsListOptions)
	if watcher != nil {
		runningPods := 0
		// Wait to get all pods ready
		// TODO add timeout
		for runningPods < newRc.Spec.Replicas {
			// Waiting for events
			<-watcher.ResultChan()
			// init counter
			runningPods = 0
			// Counting
			podList, _ := r.c.Pods(newRc.Namespace).List(NewPodsListOptions)
			for _, pod := range podList.Items {
				if api.IsPodReady(&pod) {
					runningPods++
				}
			}
		}
	}

	return newRc, nil
}

// scaleDown scales down oldRc to 0 at whatever decrement possible given the
// thresholds defined on the config. scaleDown will safely no-op as necessary
// when it detects redundancy or other relevant conditions.
func (r *RollingUpdaterByNode) scaleDownByOne(oldRc *api.ReplicationController, config *RollingUpdaterByNodeConfig) (*api.ReplicationController, error) {
	oldRc, _ = r.c.ReplicationControllers(oldRc.Namespace).Get(oldRc.Name)
	if oldRc.Spec.Replicas == 0 {
		return oldRc, nil
	}
	oldRc.Spec.Replicas--
	// Perform the scale-down.
	fmt.Fprintf(config.Out, "Scaling %s down to %d\n", oldRc.Name, oldRc.Spec.Replicas)
	// Scaledown without wait
	scaledRc, err := r.scaleAndWait(oldRc, nil, nil)
	if err != nil {
		return nil, err
	}
	return scaledRc, nil
}

// scalerScaleAndWait scales a controller using a Scaler and a real client.
func (r *RollingUpdaterByNode) scaleAndWaitWithScaler(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
	scaler, err := ScalerFor(api.Kind("ReplicationController"), r.c)
	if err != nil {
		return nil, fmt.Errorf("Couldn't make scaler: %s", err)
	}
	if err := scaler.Scale(rc.Namespace, rc.Name, uint(rc.Spec.Replicas), &ScalePrecondition{-1, ""}, retry, wait); err != nil {
		return nil, err
	}
	return r.c.ReplicationControllers(rc.Namespace).Get(rc.Name)
}

// readyPods returns the old and new ready counts for their pods.
// If a pod is observed as being ready, it's considered ready even
// if it later becomes notReady.
func (r *RollingUpdaterByNode) readyPods(oldRc, newRc *api.ReplicationController) (int, int, error) {
	controllers := []*api.ReplicationController{oldRc, newRc}
	oldReady := 0
	newReady := 0

	for i := range controllers {
		controller := controllers[i]
		selector := labels.Set(controller.Spec.Selector).AsSelector()
		options := api.ListOptions{LabelSelector: selector}
		pods, err := r.c.Pods(controller.Namespace).List(options)
		if err != nil {
			return 0, 0, err
		}
		for _, pod := range pods.Items {
			if api.IsPodReady(&pod) {
				switch controller.Name {
				case oldRc.Name:
					oldReady++
				case newRc.Name:
					newReady++
				}
			}
		}
	}
	return oldReady, newReady, nil
}

// getOrCreateTargetControllerWithClient looks for an existing controller with
// sourceId. If found, the existing controller is returned with true
// indicating that the controller already exists. If the controller isn't
// found, a new one is created and returned along with false indicating the
// controller was created.
//
// Existing controllers are validated to ensure their sourceIdAnnotation
// matches sourceId; if there's a mismatch, an error is returned.
func (r *RollingUpdaterByNode) getOrCreateTargetControllerWithClient(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error) {
	existingRc, err := r.existingController(controller)
	if err != nil {
		if !errors.IsNotFound(err) {
			// There was an error trying to find the controller; don't assume we
			// should create it.
			return nil, false, err
		}
		if controller.Spec.Replicas <= 0 {
			return nil, false, fmt.Errorf("Invalid controller spec for %s; required: > 0 replicas, actual: %d\n", controller.Name, controller.Spec.Replicas)
		}
		// The controller wasn't found, so create it.
		if controller.Annotations == nil {
			controller.Annotations = map[string]string{}
		}
		controller.Annotations[desiredReplicasAnnotation] = fmt.Sprintf("%d", controller.Spec.Replicas)
		controller.Annotations[sourceIdAnnotation] = sourceId
		controller.Spec.Replicas = 0
		newRc, err := r.c.ReplicationControllers(r.ns).Create(controller)
		return newRc, false, err
	}
	// Validate and use the existing controller.
	annotations := existingRc.Annotations
	source := annotations[sourceIdAnnotation]
	_, ok := annotations[desiredReplicasAnnotation]
	if source != sourceId || !ok {
		return nil, false, fmt.Errorf("Missing/unexpected annotations for controller %s, expected %s : %s", controller.Name, sourceId, annotations)
	}
	return existingRc, true, nil
}

// existingController verifies if the controller already exists
func (r *RollingUpdaterByNode) existingController(controller *api.ReplicationController) (*api.ReplicationController, error) {
	// without rc name but generate name, there's no existing rc
	if len(controller.Name) == 0 && len(controller.GenerateName) > 0 {
		return nil, errors.NewNotFound(api.Resource("replicationcontrollers"), controller.Name)
	}
	// controller name is required to get rc back
	return r.c.ReplicationControllers(controller.Namespace).Get(controller.Name)
}

// cleanupWithClients performs cleanup tasks after the rolling update. Update
// process related annotations are removed from oldRc and newRc. The
// CleanupPolicy on config is executed.
func (r *RollingUpdaterByNode) cleanupWithClients(oldRc, newRc *api.ReplicationController, config *RollingUpdaterByNodeConfig) error {
	// Clean up annotations
	var err error
	newRc, err = r.c.ReplicationControllers(r.ns).Get(newRc.Name)
	if err != nil {
		return err
	}
	delete(newRc.Annotations, sourceIdAnnotation)
	delete(newRc.Annotations, desiredReplicasAnnotation)

	newRc, err = r.c.ReplicationControllers(r.ns).Update(newRc)
	if err != nil {
		return err
	}
	if err = wait.Poll(config.Interval, config.Timeout, client.ControllerHasDesiredReplicas(r.c, newRc)); err != nil {
		return err
	}
	newRc, err = r.c.ReplicationControllers(r.ns).Get(newRc.Name)
	if err != nil {
		return err
	}

	switch config.CleanupPolicy {
	case DeleteRollingUpdateByNodeCleanupPolicy:
		// delete old rc
		fmt.Fprintf(config.Out, "Update succeeded. Deleting %s\n", oldRc.Name)
		return r.c.ReplicationControllers(r.ns).Delete(oldRc.Name)
	case RenameRollingUpdateByNodeCleanupPolicy:
		// delete old rc
		fmt.Fprintf(config.Out, "Update succeeded. Deleting old controller: %s\n", oldRc.Name)
		if err := r.c.ReplicationControllers(r.ns).Delete(oldRc.Name); err != nil {
			return err
		}
		fmt.Fprintf(config.Out, "Renaming %s to %s\n", newRc.Name, oldRc.Name)
		return Rename(r.c, newRc, oldRc.Name)
	case PreserveRollingUpdateByNodeCleanupPolicy:
		return nil
	default:
		return nil
	}
}

func AbortRollingUpdateByNode(c *RollingUpdaterByNodeConfig) error {
	// Swap the controllers
	tmp := c.OldRc
	c.OldRc = c.NewRc
	c.NewRc = tmp

	if c.NewRc.Annotations == nil {
		c.NewRc.Annotations = map[string]string{}
	}
	c.NewRc.Annotations[sourceIdAnnotation] = fmt.Sprintf("%s:%s", c.OldRc.Name, c.OldRc.UID)

	// Use the original value since the replica count change from old to new
	// could be asymmetric. If we don't know the original count, we can't safely
	// roll back to a known good size.
	originalSize, foundOriginal := tmp.Annotations[originalReplicasAnnotation]
	if !foundOriginal {
		return fmt.Errorf("couldn't find original replica count of %q", tmp.Name)
	}
	fmt.Fprintf(c.Out, "Setting %q replicas to %s\n", c.NewRc.Name, originalSize)
	c.NewRc.Annotations[desiredReplicasAnnotation] = originalSize
	c.CleanupPolicy = DeleteRollingUpdateByNodeCleanupPolicy
	return nil
}
