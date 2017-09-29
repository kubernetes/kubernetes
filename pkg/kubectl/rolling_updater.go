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

package kubectl

import (
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/integer"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/kubectl/util"
)

const (
	sourceIdAnnotation         = kubectlAnnotationPrefix + "update-source-id"
	desiredReplicasAnnotation  = kubectlAnnotationPrefix + "desired-replicas"
	originalReplicasAnnotation = kubectlAnnotationPrefix + "original-replicas"
	nextControllerAnnotation   = kubectlAnnotationPrefix + "next-controller-id"
)

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
	// MinReadySeconds is the number of seconds to wait after the pods are ready
	MinReadySeconds int32
	// CleanupPolicy defines the cleanup action to take after the deployment is
	// complete.
	CleanupPolicy RollingUpdaterCleanupPolicy
	// MaxUnavailable is the maximum number of pods that can be unavailable during the update.
	// Value can be an absolute number (ex: 5) or a percentage of desired pods (ex: 10%).
	// Absolute number is calculated from percentage by rounding up.
	// This can not be 0 if MaxSurge is 0.
	// By default, a fixed value of 1 is used.
	// Example: when this is set to 30%, the old RC can be scaled down to 70% of desired pods
	// immediately when the rolling update starts. Once new pods are ready, old RC
	// can be scaled down further, followed by scaling up the new RC, ensuring
	// that the total number of pods available at all times during the update is at
	// least 70% of desired pods.
	MaxUnavailable intstr.IntOrString
	// MaxSurge is the maximum number of pods that can be scheduled above the desired number of pods.
	// Value can be an absolute number (ex: 5) or a percentage of desired pods (ex: 10%).
	// This can not be 0 if MaxUnavailable is 0.
	// Absolute number is calculated from percentage by rounding up.
	// By default, a value of 1 is used.
	// Example: when this is set to 30%, the new RC can be scaled up immediately
	// when the rolling update starts, such that the total number of old and new pods do not exceed
	// 130% of desired pods. Once old pods have been killed, new RC can be scaled up
	// further, ensuring that total number of pods running at any time during
	// the update is atmost 130% of desired pods.
	MaxSurge intstr.IntOrString
	// OnProgress is invoked if set during each scale cycle, to allow the caller to perform additional logic or
	// abort the scale. If an error is returned the cleanup method will not be invoked. The percentage value
	// is a synthetic "progress" calculation that represents the approximate percentage completion.
	OnProgress func(oldRc, newRc *api.ReplicationController, percentage int) error
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

// RollingUpdater provides methods for updating replicated pods in a predictable,
// fault-tolerant way.
type RollingUpdater struct {
	rcClient  coreclient.ReplicationControllersGetter
	podClient coreclient.PodsGetter
	// Namespace for resources
	ns string
	// scaleAndWait scales a controller and returns its updated state.
	scaleAndWait func(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error)
	//getOrCreateTargetController gets and validates an existing controller or
	//makes a new one.
	getOrCreateTargetController func(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error)
	// cleanup performs post deployment cleanup tasks for newRc and oldRc.
	cleanup func(oldRc, newRc *api.ReplicationController, config *RollingUpdaterConfig) error
	// getReadyPods returns the amount of old and new ready pods.
	getReadyPods func(oldRc, newRc *api.ReplicationController, minReadySeconds int32) (int32, int32, error)
	// nowFn returns the current time used to calculate the minReadySeconds
	nowFn func() metav1.Time
}

// NewRollingUpdater creates a RollingUpdater from a client.
func NewRollingUpdater(namespace string, rcClient coreclient.ReplicationControllersGetter, podClient coreclient.PodsGetter) *RollingUpdater {
	updater := &RollingUpdater{
		rcClient:  rcClient,
		podClient: podClient,
		ns:        namespace,
	}
	// Inject real implementations.
	updater.scaleAndWait = updater.scaleAndWaitWithScaler
	updater.getOrCreateTargetController = updater.getOrCreateTargetControllerWithClient
	updater.getReadyPods = updater.readyPods
	updater.cleanup = updater.cleanupWithClients
	updater.nowFn = func() metav1.Time { return metav1.Now() }
	return updater
}

// Update all pods for a ReplicationController (oldRc) by creating a new
// controller (newRc) with 0 replicas, and synchronously scaling oldRc and
// newRc until oldRc has 0 replicas and newRc has the original # of desired
// replicas. Cleanup occurs based on a RollingUpdaterCleanupPolicy.
//
// Each interval, the updater will attempt to make progress however it can
// without violating any availability constraints defined by the config. This
// means the amount scaled up or down each interval will vary based on the
// timeliness of readiness and the updater will always try to make progress,
// even slowly.
//
// If an update from newRc to oldRc is already in progress, we attempt to
// drive it to completion. If an error occurs at any step of the update, the
// error will be returned.
//
// A scaling event (either up or down) is considered progress; if no progress
// is made within the config.Timeout, an error is returned.
//
// TODO: make this handle performing a rollback of a partially completed
// rollout.
func (r *RollingUpdater) Update(config *RollingUpdaterConfig) error {
	out := config.Out
	oldRc := config.OldRc
	scaleRetryParams := NewRetryParams(config.Interval, config.Timeout)

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
	// Extract the desired replica count from the controller.
	desiredAnnotation, err := strconv.Atoi(newRc.Annotations[desiredReplicasAnnotation])
	if err != nil {
		return fmt.Errorf("Unable to parse annotation for %s: %s=%s",
			newRc.Name, desiredReplicasAnnotation, newRc.Annotations[desiredReplicasAnnotation])
	}
	desired := int32(desiredAnnotation)
	// Extract the original replica count from the old controller, adding the
	// annotation if it doesn't yet exist.
	_, hasOriginalAnnotation := oldRc.Annotations[originalReplicasAnnotation]
	if !hasOriginalAnnotation {
		existing, err := r.rcClient.ReplicationControllers(oldRc.Namespace).Get(oldRc.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		originReplicas := strconv.Itoa(int(existing.Spec.Replicas))
		applyUpdate := func(rc *api.ReplicationController) {
			if rc.Annotations == nil {
				rc.Annotations = map[string]string{}
			}
			rc.Annotations[originalReplicasAnnotation] = originReplicas
		}
		if oldRc, err = updateRcWithRetries(r.rcClient, existing.Namespace, existing, applyUpdate); err != nil {
			return err
		}
	}
	// maxSurge is the maximum scaling increment and maxUnavailable are the maximum pods
	// that can be unavailable during a rollout.
	maxSurge, maxUnavailable, err := deploymentutil.ResolveFenceposts(&config.MaxSurge, &config.MaxUnavailable, desired)
	if err != nil {
		return err
	}
	// Validate maximums.
	if desired > 0 && maxUnavailable == 0 && maxSurge == 0 {
		return fmt.Errorf("one of maxSurge or maxUnavailable must be specified")
	}
	// The minimum pods which must remain available throughout the update
	// calculated for internal convenience.
	minAvailable := int32(integer.IntMax(0, int(desired-maxUnavailable)))
	// If the desired new scale is 0, then the max unavailable is necessarily
	// the effective scale of the old RC regardless of the configuration
	// (equivalent to 100% maxUnavailable).
	if desired == 0 {
		maxUnavailable = oldRc.Spec.Replicas
		minAvailable = 0
	}

	fmt.Fprintf(out, "Scaling up %s from %d to %d, scaling down %s from %d to 0 (keep %d pods available, don't exceed %d pods)\n",
		newRc.Name, newRc.Spec.Replicas, desired, oldRc.Name, oldRc.Spec.Replicas, minAvailable, desired+maxSurge)

	// give a caller incremental notification and allow them to exit early
	goal := desired - newRc.Spec.Replicas
	if goal < 0 {
		goal = -goal
	}
	progress := func(complete bool) error {
		if config.OnProgress == nil {
			return nil
		}
		progress := desired - newRc.Spec.Replicas
		if progress < 0 {
			progress = -progress
		}
		percentage := 100
		if !complete && goal > 0 {
			percentage = int((goal - progress) * 100 / goal)
		}
		return config.OnProgress(oldRc, newRc, percentage)
	}

	// Scale newRc and oldRc until newRc has the desired number of replicas and
	// oldRc has 0 replicas.
	progressDeadline := time.Now().UnixNano() + config.Timeout.Nanoseconds()
	for newRc.Spec.Replicas != desired || oldRc.Spec.Replicas != 0 {
		// Store the existing replica counts for progress timeout tracking.
		newReplicas := newRc.Spec.Replicas
		oldReplicas := oldRc.Spec.Replicas

		// Scale up as much as possible.
		scaledRc, err := r.scaleUp(newRc, oldRc, desired, maxSurge, maxUnavailable, scaleRetryParams, config)
		if err != nil {
			return err
		}
		newRc = scaledRc

		// notify the caller if necessary
		if err := progress(false); err != nil {
			return err
		}

		// Wait between scaling operations for things to settle.
		time.Sleep(config.UpdatePeriod)

		// Scale down as much as possible.
		scaledRc, err = r.scaleDown(newRc, oldRc, desired, minAvailable, maxUnavailable, maxSurge, config)
		if err != nil {
			return err
		}
		oldRc = scaledRc

		// notify the caller if necessary
		if err := progress(false); err != nil {
			return err
		}

		// If we are making progress, continue to advance the progress deadline.
		// Otherwise, time out with an error.
		progressMade := (newRc.Spec.Replicas != newReplicas) || (oldRc.Spec.Replicas != oldReplicas)
		if progressMade {
			progressDeadline = time.Now().UnixNano() + config.Timeout.Nanoseconds()
		} else if time.Now().UnixNano() > progressDeadline {
			return fmt.Errorf("timed out waiting for any update progress to be made")
		}
	}

	// notify the caller if necessary
	if err := progress(true); err != nil {
		return err
	}

	// Housekeeping and cleanup policy execution.
	return r.cleanup(oldRc, newRc, config)
}

// scaleUp scales up newRc to desired by whatever increment is possible given
// the configured surge threshold. scaleUp will safely no-op as necessary when
// it detects redundancy or other relevant conditions.
func (r *RollingUpdater) scaleUp(newRc, oldRc *api.ReplicationController, desired, maxSurge, maxUnavailable int32, scaleRetryParams *RetryParams, config *RollingUpdaterConfig) (*api.ReplicationController, error) {
	// If we're already at the desired, do nothing.
	if newRc.Spec.Replicas == desired {
		return newRc, nil
	}

	// Scale up as far as we can based on the surge limit.
	increment := (desired + maxSurge) - (oldRc.Spec.Replicas + newRc.Spec.Replicas)
	// If the old is already scaled down, go ahead and scale all the way up.
	if oldRc.Spec.Replicas == 0 {
		increment = desired - newRc.Spec.Replicas
	}
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
	return scaledRc, nil
}

// scaleDown scales down oldRc to 0 at whatever decrement possible given the
// thresholds defined on the config. scaleDown will safely no-op as necessary
// when it detects redundancy or other relevant conditions.
func (r *RollingUpdater) scaleDown(newRc, oldRc *api.ReplicationController, desired, minAvailable, maxUnavailable, maxSurge int32, config *RollingUpdaterConfig) (*api.ReplicationController, error) {
	// Already scaled down; do nothing.
	if oldRc.Spec.Replicas == 0 {
		return oldRc, nil
	}
	// Get ready pods. We shouldn't block, otherwise in case both old and new
	// pods are unavailable then the rolling update process blocks.
	// Timeout-wise we are already covered by the progress check.
	_, newAvailable, err := r.getReadyPods(oldRc, newRc, config.MinReadySeconds)
	if err != nil {
		return nil, err
	}
	// The old controller is considered as part of the total because we want to
	// maintain minimum availability even with a volatile old controller.
	// Scale down as much as possible while maintaining minimum availability
	allPods := oldRc.Spec.Replicas + newRc.Spec.Replicas
	newUnavailable := newRc.Spec.Replicas - newAvailable
	decrement := allPods - minAvailable - newUnavailable
	// The decrement normally shouldn't drop below 0 because the available count
	// always starts below the old replica count, but the old replica count can
	// decrement due to externalities like pods death in the replica set. This
	// will be considered a transient condition; do nothing and try again later
	// with new readiness values.
	//
	// If the most we can scale is 0, it means we can't scale down without
	// violating the minimum. Do nothing and try again later when conditions may
	// have changed.
	if decrement <= 0 {
		return oldRc, nil
	}
	// Reduce the replica count, and deal with fenceposts.
	oldRc.Spec.Replicas -= decrement
	if oldRc.Spec.Replicas < 0 {
		oldRc.Spec.Replicas = 0
	}
	// If the new is already fully scaled and available up to the desired size, go
	// ahead and scale old all the way down.
	if newRc.Spec.Replicas == desired && newAvailable == desired {
		oldRc.Spec.Replicas = 0
	}
	// Perform the scale-down.
	fmt.Fprintf(config.Out, "Scaling %s down to %d\n", oldRc.Name, oldRc.Spec.Replicas)
	retryWait := &RetryParams{config.Interval, config.Timeout}
	scaledRc, err := r.scaleAndWait(oldRc, retryWait, retryWait)
	if err != nil {
		return nil, err
	}
	return scaledRc, nil
}

// scalerScaleAndWait scales a controller using a Scaler and a real client.
func (r *RollingUpdater) scaleAndWaitWithScaler(rc *api.ReplicationController, retry *RetryParams, wait *RetryParams) (*api.ReplicationController, error) {
	scaler := &ReplicationControllerScaler{r.rcClient}
	if err := scaler.Scale(rc.Namespace, rc.Name, uint(rc.Spec.Replicas), &ScalePrecondition{-1, ""}, retry, wait); err != nil {
		return nil, err
	}
	return r.rcClient.ReplicationControllers(rc.Namespace).Get(rc.Name, metav1.GetOptions{})
}

// readyPods returns the old and new ready counts for their pods.
// If a pod is observed as being ready, it's considered ready even
// if it later becomes notReady.
func (r *RollingUpdater) readyPods(oldRc, newRc *api.ReplicationController, minReadySeconds int32) (int32, int32, error) {
	controllers := []*api.ReplicationController{oldRc, newRc}
	oldReady := int32(0)
	newReady := int32(0)
	if r.nowFn == nil {
		r.nowFn = func() metav1.Time { return metav1.Now() }
	}

	for i := range controllers {
		controller := controllers[i]
		selector := labels.Set(controller.Spec.Selector).AsSelector()
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := r.podClient.Pods(controller.Namespace).List(options)
		if err != nil {
			return 0, 0, err
		}
		for _, pod := range pods.Items {
			v1Pod := &v1.Pod{}
			if err := apiv1.Convert_api_Pod_To_v1_Pod(&pod, v1Pod, nil); err != nil {
				return 0, 0, err
			}
			// Do not count deleted pods as ready
			if v1Pod.DeletionTimestamp != nil {
				continue
			}
			if !podutil.IsPodAvailable(v1Pod, minReadySeconds, r.nowFn()) {
				continue
			}
			switch controller.Name {
			case oldRc.Name:
				oldReady++
			case newRc.Name:
				newReady++
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
func (r *RollingUpdater) getOrCreateTargetControllerWithClient(controller *api.ReplicationController, sourceId string) (*api.ReplicationController, bool, error) {
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
		newRc, err := r.rcClient.ReplicationControllers(r.ns).Create(controller)
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
func (r *RollingUpdater) existingController(controller *api.ReplicationController) (*api.ReplicationController, error) {
	// without rc name but generate name, there's no existing rc
	if len(controller.Name) == 0 && len(controller.GenerateName) > 0 {
		return nil, errors.NewNotFound(api.Resource("replicationcontrollers"), controller.Name)
	}
	// controller name is required to get rc back
	return r.rcClient.ReplicationControllers(controller.Namespace).Get(controller.Name, metav1.GetOptions{})
}

// cleanupWithClients performs cleanup tasks after the rolling update. Update
// process related annotations are removed from oldRc and newRc. The
// CleanupPolicy on config is executed.
func (r *RollingUpdater) cleanupWithClients(oldRc, newRc *api.ReplicationController, config *RollingUpdaterConfig) error {
	// Clean up annotations
	var err error
	newRc, err = r.rcClient.ReplicationControllers(r.ns).Get(newRc.Name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	applyUpdate := func(rc *api.ReplicationController) {
		delete(rc.Annotations, sourceIdAnnotation)
		delete(rc.Annotations, desiredReplicasAnnotation)
	}
	if newRc, err = updateRcWithRetries(r.rcClient, r.ns, newRc, applyUpdate); err != nil {
		return err
	}

	if err = wait.Poll(config.Interval, config.Timeout, client.ControllerHasDesiredReplicas(r.rcClient, newRc)); err != nil {
		return err
	}
	newRc, err = r.rcClient.ReplicationControllers(r.ns).Get(newRc.Name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	switch config.CleanupPolicy {
	case DeleteRollingUpdateCleanupPolicy:
		// delete old rc
		fmt.Fprintf(config.Out, "Update succeeded. Deleting %s\n", oldRc.Name)
		return r.rcClient.ReplicationControllers(r.ns).Delete(oldRc.Name, nil)
	case RenameRollingUpdateCleanupPolicy:
		// delete old rc
		fmt.Fprintf(config.Out, "Update succeeded. Deleting old controller: %s\n", oldRc.Name)
		if err := r.rcClient.ReplicationControllers(r.ns).Delete(oldRc.Name, nil); err != nil {
			return err
		}
		fmt.Fprintf(config.Out, "Renaming %s to %s\n", newRc.Name, oldRc.Name)
		return Rename(r.rcClient, newRc, oldRc.Name)
	case PreserveRollingUpdateCleanupPolicy:
		return nil
	default:
		return nil
	}
}

func Rename(c coreclient.ReplicationControllersGetter, rc *api.ReplicationController, newName string) error {
	oldName := rc.Name
	rc.Name = newName
	rc.ResourceVersion = ""
	// First delete the oldName RC and orphan its pods.
	trueVar := true
	err := c.ReplicationControllers(rc.Namespace).Delete(oldName, &metav1.DeleteOptions{OrphanDependents: &trueVar})
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	err = wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
		_, err := c.ReplicationControllers(rc.Namespace).Get(oldName, metav1.GetOptions{})
		if err == nil {
			return false, nil
		} else if errors.IsNotFound(err) {
			return true, nil
		} else {
			return false, err
		}
	})
	if err != nil {
		return err
	}
	// Then create the same RC with the new name.
	_, err = c.ReplicationControllers(rc.Namespace).Create(rc)
	return err
}

func LoadExistingNextReplicationController(c coreclient.ReplicationControllersGetter, namespace, newName string) (*api.ReplicationController, error) {
	if len(newName) == 0 {
		return nil, nil
	}
	newRc, err := c.ReplicationControllers(namespace).Get(newName, metav1.GetOptions{})
	if err != nil && errors.IsNotFound(err) {
		return nil, nil
	}
	return newRc, err
}

type NewControllerConfig struct {
	Namespace        string
	OldName, NewName string
	Image            string
	Container        string
	DeploymentKey    string
	PullPolicy       api.PullPolicy
}

func CreateNewControllerFromCurrentController(rcClient coreclient.ReplicationControllersGetter, codec runtime.Codec, cfg *NewControllerConfig) (*api.ReplicationController, error) {
	containerIndex := 0
	// load the old RC into the "new" RC
	newRc, err := rcClient.ReplicationControllers(cfg.Namespace).Get(cfg.OldName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	if len(cfg.Container) != 0 {
		containerFound := false

		for i, c := range newRc.Spec.Template.Spec.Containers {
			if c.Name == cfg.Container {
				containerIndex = i
				containerFound = true
				break
			}
		}

		if !containerFound {
			return nil, fmt.Errorf("container %s not found in pod", cfg.Container)
		}
	}

	if len(newRc.Spec.Template.Spec.Containers) > 1 && len(cfg.Container) == 0 {
		return nil, fmt.Errorf("must specify container to update when updating a multi-container pod")
	}

	if len(newRc.Spec.Template.Spec.Containers) == 0 {
		return nil, fmt.Errorf("pod has no containers! (%v)", newRc)
	}
	newRc.Spec.Template.Spec.Containers[containerIndex].Image = cfg.Image
	if len(cfg.PullPolicy) != 0 {
		newRc.Spec.Template.Spec.Containers[containerIndex].ImagePullPolicy = cfg.PullPolicy
	}

	newHash, err := util.HashObject(newRc, codec)
	if err != nil {
		return nil, err
	}

	if len(cfg.NewName) == 0 {
		cfg.NewName = fmt.Sprintf("%s-%s", newRc.Name, newHash)
	}
	newRc.Name = cfg.NewName

	newRc.Spec.Selector[cfg.DeploymentKey] = newHash
	newRc.Spec.Template.Labels[cfg.DeploymentKey] = newHash
	// Clear resource version after hashing so that identical updates get different hashes.
	newRc.ResourceVersion = ""
	return newRc, nil
}

func AbortRollingUpdate(c *RollingUpdaterConfig) error {
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
	c.CleanupPolicy = DeleteRollingUpdateCleanupPolicy
	return nil
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

func UpdateExistingReplicationController(rcClient coreclient.ReplicationControllersGetter, podClient coreclient.PodsGetter, oldRc *api.ReplicationController, namespace, newName, deploymentKey, deploymentValue string, out io.Writer) (*api.ReplicationController, error) {
	if _, found := oldRc.Spec.Selector[deploymentKey]; !found {
		SetNextControllerAnnotation(oldRc, newName)
		return AddDeploymentKeyToReplicationController(oldRc, rcClient, podClient, deploymentKey, deploymentValue, namespace, out)
	}

	// If we didn't need to update the controller for the deployment key, we still need to write
	// the "next" controller.
	applyUpdate := func(rc *api.ReplicationController) {
		SetNextControllerAnnotation(rc, newName)
	}
	return updateRcWithRetries(rcClient, namespace, oldRc, applyUpdate)
}

func AddDeploymentKeyToReplicationController(oldRc *api.ReplicationController, rcClient coreclient.ReplicationControllersGetter, podClient coreclient.PodsGetter, deploymentKey, deploymentValue, namespace string, out io.Writer) (*api.ReplicationController, error) {
	var err error
	// First, update the template label.  This ensures that any newly created pods will have the new label
	applyUpdate := func(rc *api.ReplicationController) {
		if rc.Spec.Template.Labels == nil {
			rc.Spec.Template.Labels = map[string]string{}
		}
		rc.Spec.Template.Labels[deploymentKey] = deploymentValue
	}
	if oldRc, err = updateRcWithRetries(rcClient, namespace, oldRc, applyUpdate); err != nil {
		return nil, err
	}

	// Update all pods managed by the rc to have the new hash label, so they are correctly adopted
	// TODO: extract the code from the label command and re-use it here.
	selector := labels.SelectorFromSet(oldRc.Spec.Selector)
	options := metav1.ListOptions{LabelSelector: selector.String()}
	podList, err := podClient.Pods(namespace).List(options)
	if err != nil {
		return nil, err
	}
	for ix := range podList.Items {
		pod := &podList.Items[ix]
		applyUpdate := func(p *api.Pod) {
			if p.Labels == nil {
				p.Labels = map[string]string{
					deploymentKey: deploymentValue,
				}
			} else {
				p.Labels[deploymentKey] = deploymentValue
			}
		}
		if pod, err = updatePodWithRetries(podClient, namespace, pod, applyUpdate); err != nil {
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
	applyUpdate = func(rc *api.ReplicationController) {
		rc.Spec.Selector[deploymentKey] = deploymentValue
	}
	// Update the selector of the rc so it manages all the pods we updated above
	if oldRc, err = updateRcWithRetries(rcClient, namespace, oldRc, applyUpdate); err != nil {
		return nil, err
	}

	// Clean up any orphaned pods that don't have the new label, this can happen if the rc manager
	// doesn't see the update to its pod template and creates a new pod with the old labels after
	// we've finished re-adopting existing pods to the rc.
	selector = labels.SelectorFromSet(selectorCopy)
	options = metav1.ListOptions{LabelSelector: selector.String()}
	if podList, err = podClient.Pods(namespace).List(options); err != nil {
		return nil, err
	}
	for ix := range podList.Items {
		pod := &podList.Items[ix]
		if value, found := pod.Labels[deploymentKey]; !found || value != deploymentValue {
			if err := podClient.Pods(namespace).Delete(pod.Name, nil); err != nil {
				return nil, err
			}
		}
	}

	return oldRc, nil
}

type updateRcFunc func(controller *api.ReplicationController)

// updateRcWithRetries retries updating the given rc on conflict with the following steps:
// 1. Get latest resource
// 2. applyUpdate
// 3. Update the resource
func updateRcWithRetries(rcClient coreclient.ReplicationControllersGetter, namespace string, rc *api.ReplicationController, applyUpdate updateRcFunc) (*api.ReplicationController, error) {
	// Deep copy the rc in case we failed on Get during retry loop
	oldRc := rc.DeepCopy()
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() (e error) {
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(rc)
		if rc, e = rcClient.ReplicationControllers(namespace).Update(rc); e == nil {
			// rc contains the latest controller post update
			return
		}
		updateErr := e
		// Update the controller with the latest resource version, if the update failed we
		// can't trust rc so use oldRc.Name.
		if rc, e = rcClient.ReplicationControllers(namespace).Get(oldRc.Name, metav1.GetOptions{}); e != nil {
			// The Get failed: Value in rc cannot be trusted.
			rc = oldRc
		}
		// Only return the error from update
		return updateErr
	})
	// If the error is non-nil the returned controller cannot be trusted, if it is nil, the returned
	// controller contains the applied update.
	return rc, err
}

type updatePodFunc func(controller *api.Pod)

// updatePodWithRetries retries updating the given pod on conflict with the following steps:
// 1. Get latest resource
// 2. applyUpdate
// 3. Update the resource
func updatePodWithRetries(podClient coreclient.PodsGetter, namespace string, pod *api.Pod, applyUpdate updatePodFunc) (*api.Pod, error) {
	// Deep copy the pod in case we failed on Get during retry loop
	oldPod := pod.DeepCopy()
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() (e error) {
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(pod)
		if pod, e = podClient.Pods(namespace).Update(pod); e == nil {
			return
		}
		updateErr := e
		if pod, e = podClient.Pods(namespace).Get(oldPod.Name, metav1.GetOptions{}); e != nil {
			pod = oldPod
		}
		// Only return the error from update
		return updateErr
	})
	// If the error is non-nil the returned pod cannot be trusted, if it is nil, the returned
	// controller contains the applied update.
	return pod, err
}

func FindSourceController(r coreclient.ReplicationControllersGetter, namespace, name string) (*api.ReplicationController, error) {
	list, err := r.ReplicationControllers(namespace).List(metav1.ListOptions{})
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
