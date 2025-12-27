/*
Copyright 2022 The Kubernetes Authors.

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

package dra

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"

	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	draplugin "k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/resourceupdates"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// draManagerStateFileName is the file name where dra manager stores its state
const draManagerStateFileName = "dra_manager_state"

// defaultReconcilePeriod is the default reconciliation period to keep all claim info state in sync.
const defaultReconcilePeriod = 60 * time.Second

// The time that DRA drivers have to come back after being unregistered
// before the kubelet removes their ResourceSlices.
//
// This must be long enough to actually allow stopping a pod and
// starting the replacement (otherwise ResourceSlices get deleted
// unnecessarily) and not too long (otherwise the time window were
// pods might still get scheduled to the node after removal of a
// driver is too long).
//
// 30 seconds might be long enough for a simple container restart.
// If a DRA driver wants to be sure that slices don't get wiped,
// it should use rolling updates.
const defaultWipingDelay = 30 * time.Second

// ActivePodsFunc is a function that returns a list of pods to reconcile.
type ActivePodsFunc func() []*v1.Pod

// GetNodeFunc is a function that returns the node object using the kubelet's node lister.
type GetNodeFunc func() (*v1.Node, error)

// Manager is responsible for managing ResourceClaims.
// It ensures that they are prepared before starting pods
// and that they are unprepared before the last consuming
// pod is declared as terminated.
type Manager struct {
	// draPlugins manages the registered plugins.
	draPlugins *draplugin.DRAPluginManager

	// cache contains cached claim info
	cache *claimInfoCache

	// reconcilePeriod is the duration between calls to reconcileLoop.
	reconcilePeriod time.Duration

	// activePods is a method for listing active pods on the node
	// so all claim info state can be updated in the reconciliation loop.
	activePods ActivePodsFunc

	// sourcesReady provides the readiness of kubelet configuration sources such as apiserver update readiness.
	// We use it to determine when we can treat pods as inactive and react appropriately.
	sourcesReady config.SourcesReady

	// KubeClient reference
	kubeClient clientset.Interface

	// healthInfoCache contains cached health info
	healthInfoCache *healthInfoCache

	// update channel for resource updates
	update chan resourceupdates.Update
}

// NewManager creates a new DRA manager.
//
// Most errors returned by the manager show up in the context of a pod.
// They try to adhere to the following convention:
// - Don't include the pod.
// - Use terms that are familiar to users.
// - Don't include the namespace, it can be inferred from the context.
// - Avoid repeated "failed to ...: failed to ..." when wrapping errors.
// - Avoid wrapping when it does not provide relevant additional information to keep the user-visible error short.
func NewManager(logger klog.Logger, kubeClient clientset.Interface, stateFileDirectory string) (*Manager, error) {
	claimInfoCache, err := newClaimInfoCache(logger, stateFileDirectory, draManagerStateFileName)
	if err != nil {
		return nil, fmt.Errorf("create ResourceClaim cache: %w", err)
	}

	healthInfoCache, err := newHealthInfoCache(filepath.Join(stateFileDirectory, "dra_health_state"))
	if err != nil {
		return nil, fmt.Errorf("failed to create healthInfo cache: %w", err)
	}

	// TODO: for now the reconcile period is not configurable.
	// We should consider making it configurable in the future.
	reconcilePeriod := defaultReconcilePeriod

	manager := &Manager{
		cache:           claimInfoCache,
		kubeClient:      kubeClient,
		reconcilePeriod: reconcilePeriod,
		activePods:      nil,
		sourcesReady:    nil,
		healthInfoCache: healthInfoCache,
		update:          make(chan resourceupdates.Update, 100),
	}

	return manager, nil
}

func (m *Manager) NewMetricsCollector() metrics.StableCollector {
	return &claimInfoCollector{cache: m.cache}
}

// GetWatcherHandler must be called after Start, it indirectly depends
// on parameters which only get passed to Start, for example the context.
func (m *Manager) GetWatcherHandler() cache.PluginHandler {
	return m.draPlugins
}

// Start starts the reconcile loop of the manager.
func (m *Manager) Start(ctx context.Context, activePods ActivePodsFunc, getNode GetNodeFunc, sourcesReady config.SourcesReady) error {
	m.initDRAPluginManager(ctx, getNode, defaultWipingDelay)
	m.activePods = activePods
	m.sourcesReady = sourcesReady
	go wait.UntilWithContext(ctx, func(ctx context.Context) { m.reconcileLoop(ctx) }, m.reconcilePeriod)
	return nil
}

// initPluginManager can be used instead of Start to make the manager useable
// for calls to prepare/unprepare. It exists primarily for testing purposes.
func (m *Manager) initDRAPluginManager(ctx context.Context, getNode GetNodeFunc, wipingDelay time.Duration) {
	m.draPlugins = draplugin.NewDRAPluginManager(ctx, m.kubeClient, getNode, m, wipingDelay)
}

// reconcileLoop ensures that any stale state in the manager's claimInfoCache gets periodically reconciled.
func (m *Manager) reconcileLoop(ctx context.Context) {
	logger := klog.FromContext(ctx)
	// Only once all sources are ready do we attempt to reconcile.
	// This ensures that the call to m.activePods() below will succeed with
	// the actual active pods list.
	if m.sourcesReady == nil || !m.sourcesReady.AllReady() {
		return
	}

	// Get the full list of active pods.
	activePods := sets.New[string]()
	for _, p := range m.activePods() {
		activePods.Insert(string(p.UID))
	}

	// Get the list of inactive pods still referenced by any claimInfos.
	type podClaims struct {
		uid        types.UID
		namespace  string
		claimNames []string
	}
	inactivePodClaims := make(map[string]*podClaims)
	m.cache.RLock()
	for _, claimInfo := range m.cache.claimInfo {
		for podUID := range claimInfo.PodUIDs {
			if activePods.Has(podUID) {
				continue
			}
			if inactivePodClaims[podUID] == nil {
				inactivePodClaims[podUID] = &podClaims{
					uid:        types.UID(podUID),
					namespace:  claimInfo.Namespace,
					claimNames: []string{},
				}
			}
			inactivePodClaims[podUID].claimNames = append(inactivePodClaims[podUID].claimNames, claimInfo.ClaimName)
		}
	}
	m.cache.RUnlock()

	// Loop through all inactive pods and call UnprepareResources on them.
	for _, podClaims := range inactivePodClaims {
		if err := m.unprepareResources(ctx, podClaims.uid, podClaims.namespace, podClaims.claimNames); err != nil {
			logger.Info("Unpreparing pod resources in reconcile loop failed, will retry", "podUID", podClaims.uid, "err", err)
		}
	}
}

// PrepareResources attempts to prepare all of the required resources
// for the input container, issue NodePrepareResources rpc requests
// for each new resource requirement, process their responses and update the cached
// containerResources on success.
func (m *Manager) PrepareResources(ctx context.Context, pod *v1.Pod) error {
	startTime := time.Now()
	err := m.prepareResources(ctx, pod)
	kubeletmetrics.DRAOperationsDuration.WithLabelValues("PrepareResources", strconv.FormatBool(err == nil)).Observe(time.Since(startTime).Seconds())
	if err != nil {
		return fmt.Errorf("prepare dynamic resources: %w", err)
	}
	return nil
}

func (m *Manager) prepareResources(ctx context.Context, pod *v1.Pod) error {
	var err error
	logger := klog.FromContext(ctx)
	batches := make(map[*draplugin.DRAPlugin][]*drapb.Claim)
	resourceClaims := make(map[types.UID]*resourceapi.ResourceClaim)

	// Do a validation pass *without* changing the claim info cache.
	// If anything goes wrong, we don't proceed. This has the advantage
	// that the failing pod can be deleted without getting stuck.
	//
	// If we added the claim and pod to the cache, UnprepareResources
	// would have to asssume that NodePrepareResources was called and
	// try to call NodeUnprepareResources. This is particularly bad
	// when the driver never has been installed on the node and
	// remains unavailable.
	podResourceClaims := pod.Spec.ResourceClaims
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.DRAExtendedResource) {
		if pod.Status.ExtendedResourceClaimStatus != nil {
			extendedResourceClaim := v1.PodResourceClaim{
				ResourceClaimName: &pod.Status.ExtendedResourceClaimStatus.ResourceClaimName,
			}
			podResourceClaims = make([]v1.PodResourceClaim, 0, len(pod.Spec.ResourceClaims)+1)
			podResourceClaims = append(podResourceClaims, pod.Spec.ResourceClaims...)
			podResourceClaims = append(podResourceClaims, extendedResourceClaim)
		}
	}
	infos := make([]struct {
		resourceClaim *resourceapi.ResourceClaim
		podClaim      *v1.PodResourceClaim
		claimInfo     *ClaimInfo
		plugins       map[string]*draplugin.DRAPlugin
	}, len(podResourceClaims))
	for i := range podResourceClaims {
		podClaim := &podResourceClaims[i]
		infos[i].podClaim = podClaim
		logger.V(3).Info("Processing resource", "pod", klog.KObj(pod), "podClaim", podClaim.Name)
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, podClaim)
		if err != nil {
			return err
		}

		if claimName == nil {
			// Nothing to do.
			continue
		}
		// Query claim object from the API server
		resourceClaim, err := m.kubeClient.ResourceV1().ResourceClaims(pod.Namespace).Get(
			ctx,
			*claimName,
			metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("fetch ResourceClaim %s: %w", *claimName, err)
		}

		if mustCheckOwner {
			if err = resourceclaim.IsForPod(pod, resourceClaim); err != nil {
				// No wrapping, error is already informative.
				return err
			}
		}

		// Check if pod is in the ReservedFor for the claim
		if !resourceclaim.IsReservedForPod(pod, resourceClaim) {
			return fmt.Errorf("pod %s (%s) is not allowed to use ResourceClaim %s (%s)",
				pod.Name, pod.UID, *claimName, resourceClaim.UID)
		}

		// At this point we assume that we have to prepare the claim and thus need
		// the driver. If the driver is currently unavailable, it is better to fail
		// even if the claim is already prepared because something is wrong with
		// the node.
		infos[i].resourceClaim = resourceClaim
		claimInfo, err := newClaimInfoFromClaim(resourceClaim)
		if err != nil {
			return fmt.Errorf("ResourceClaim %s: %w", resourceClaim.Name, err)
		}
		infos[i].claimInfo = claimInfo
		infos[i].plugins = make(map[string]*draplugin.DRAPlugin, len(claimInfo.DriverState))
		for driverName := range claimInfo.DriverState {
			if plugin := infos[i].plugins[driverName]; plugin != nil {
				continue
			}
			plugin, err := m.draPlugins.GetPlugin(driverName)
			if err != nil {
				// No wrapping, error includes driver name already.
				return err
			}
			infos[i].plugins[driverName] = plugin
		}
	}

	// Now that we have everything that we need, we can update the claim info cache.
	// Almost nothing can go wrong anymore at this point.
	err = m.cache.withLock(func() error {
		for i := range podResourceClaims {
			resourceClaim := infos[i].resourceClaim
			podClaim := infos[i].podClaim
			if resourceClaim == nil {
				logger.V(5).Info("No need to prepare resources, no claim generated", "pod", klog.KObj(pod), "podClaim", podClaim.Name)
				continue
			}
			// Get a reference to the claim info for this claim from the cache.
			// If there isn't one yet, then add it to the cache.
			claimInfo, exists := m.cache.get(resourceClaim.Name, resourceClaim.Namespace)
			if !exists {
				claimInfo = infos[i].claimInfo
				m.cache.add(claimInfo)
				logger.V(6).Info("Created new claim info cache entry", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim), "claimInfoEntry", claimInfo)
			} else {
				if claimInfo.ClaimUID != resourceClaim.UID {
					return fmt.Errorf("old ResourceClaim with same name %s and different UID %s still exists (previous pod force-deleted?!)", resourceClaim.Name, claimInfo.ClaimUID)
				}
				logger.V(6).Info("Found existing claim info cache entry", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim), "claimInfoEntry", claimInfo)
			}

			// Add a reference to the current pod in the claim info.
			claimInfo.addPodReference(pod.UID)

			// Checkpoint to ensure all claims we plan to prepare are tracked.
			// If something goes wrong and the newly referenced pod gets
			// deleted without a successful prepare call, we will catch
			// that in the reconcile loop and take the appropriate action.
			if err := m.cache.syncToCheckpoint(); err != nil {
				return fmt.Errorf("checkpoint ResourceClaim cache: %w", err)
			}

			// If this claim is already prepared, continue preparing for any remaining claims.
			if claimInfo.isPrepared() {
				logger.V(5).Info("Resources already prepared", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim))
				continue
			}

			// This saved claim will be used to update ClaimInfo cache
			// after NodePrepareResources GRPC succeeds
			resourceClaims[claimInfo.ClaimUID] = resourceClaim

			// Loop through all drivers and prepare for calling NodePrepareResources.
			claim := &drapb.Claim{
				Namespace: claimInfo.Namespace,
				Uid:       string(claimInfo.ClaimUID),
				Name:      claimInfo.ClaimName,
			}
			for driverName := range claimInfo.DriverState {
				plugin := infos[i].plugins[driverName]
				batches[plugin] = append(batches[plugin], claim)
			}
		}

		return nil
	})
	if err != nil {
		// No error wrapping because there is no additional context needed.
		// What we get here are the errors from our own callback above.
		return err
	}

	// Call NodePrepareResources for all claims in each batch.
	// If there is any error, processing gets aborted.
	// We could try to continue, but that would make the code more complex.
	for plugin, claims := range batches {
		// Call NodePrepareResources RPC for all resource handles.
		response, err := plugin.NodePrepareResources(ctx, &drapb.NodePrepareResourcesRequest{Claims: claims})
		if err != nil {
			// General error unrelated to any particular claim.
			return fmt.Errorf("NodePrepareResources: %w", err)
		}
		for claimUID, result := range response.Claims {
			reqClaim := lookupClaimRequest(claims, claimUID)
			if reqClaim == nil {
				return fmt.Errorf("NodePrepareResources returned result for unknown claim UID %s", claimUID)
			}
			if result.GetError() != "" {
				return fmt.Errorf("NodePrepareResources failed for ResourceClaim %s: %s", reqClaim.Name, result.Error)
			}

			claim := resourceClaims[types.UID(claimUID)]

			// Add the prepared CDI devices to the claim info
			err := m.cache.withLock(func() error {
				info, exists := m.cache.get(claim.Name, claim.Namespace)
				if !exists {
					return fmt.Errorf("internal error: unable to get claim info for ResourceClaim %s", claim.Name)
				}
				for _, device := range result.GetDevices() {
					info.addDevice(plugin.DriverName(), state.Device{PoolName: device.PoolName,
						DeviceName: device.DeviceName, ShareID: (*types.UID)(device.ShareId),
						RequestNames: device.RequestNames, CDIDeviceIDs: device.CdiDeviceIds})
				}
				return nil
			})
			if err != nil {
				// No wrapping, this is the error above.
				return err
			}
		}

		unfinished := len(claims) - len(response.Claims)
		if unfinished != 0 {
			return fmt.Errorf("NodePrepareResources skipped %d ResourceClaims", unfinished)
		}
	}

	// Atomically perform some operations on the claimInfo cache.
	err = m.cache.withLock(func() error {
		// Mark all pod claims as prepared.
		for _, claim := range resourceClaims {
			info, exists := m.cache.get(claim.Name, claim.Namespace)
			if !exists {
				return fmt.Errorf("internal error: unable to get claim info for ResourceClaim %s", claim.Name)
			}
			info.setPrepared()
		}

		// Checkpoint to ensure all prepared claims are tracked with their list
		// of CDI devices attached.
		if err := m.cache.syncToCheckpoint(); err != nil {
			return fmt.Errorf("checkpoint ResourceClaim state: %w", err)
		}

		return nil
	})
	if err != nil {
		// No wrapping, this is the error above.
		return err
	}

	return nil
}

func lookupClaimRequest(claims []*drapb.Claim, claimUID string) *drapb.Claim {
	for _, claim := range claims {
		if claim.Uid == claimUID {
			return claim
		}
	}
	return nil
}

// GetResources gets a ContainerInfo object from the claimInfo cache.
// This information is used by the caller to update a container config.
func (m *Manager) GetResources(pod *v1.Pod, container *v1.Container) (*ContainerInfo, error) {
	cdiDevices := []kubecontainer.CDIDevice{}

	for i := range pod.Spec.ResourceClaims {
		podClaim := &pod.Spec.ResourceClaims[i]
		claimName, _, err := resourceclaim.Name(pod, podClaim)
		if err != nil {
			// No wrapping, error is already informative.
			return nil, err
		}
		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			continue
		}
		for _, claim := range container.Resources.Claims {
			if podClaim.Name != claim.Name {
				continue
			}

			err := m.cache.withRLock(func() error {
				claimInfo, exists := m.cache.get(*claimName, pod.Namespace)
				if !exists {
					return fmt.Errorf("internal error: unable to get claim info for ResourceClaim %s", *claimName)
				}

				// As of Kubernetes 1.31, CDI device IDs are not passed via annotations anymore.
				cdiDevices = append(cdiDevices, claimInfo.cdiDevicesAsList(claim.Request)...)

				return nil
			})
			if err != nil {
				// No wrapping, this is the error above.
				return nil, err
			}
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.DRAExtendedResource) && pod.Status.ExtendedResourceClaimStatus != nil {
		claimName := pod.Status.ExtendedResourceClaimStatus.ResourceClaimName
		// if the container has requests for extended resources backed by DRA,
		// they must have been allocated via the extendedResourceClaim created
		// by the kube-scheduler.
		err := m.cache.withRLock(func() error {
			claimInfo, exists := m.cache.get(claimName, pod.Namespace)
			if !exists {
				return fmt.Errorf("unable to get claim info for claim %s in namespace %s", claimName, pod.Namespace)
			}

			for rName, rValue := range container.Resources.Requests {
				if rValue.IsZero() {
					// We only care about the resources requested by the pod
					continue
				}
				if schedutil.IsDRAExtendedResourceName(rName) {
					for _, rm := range pod.Status.ExtendedResourceClaimStatus.RequestMappings {
						// allow multiple device requests per container per resource.
						if rm.ContainerName == container.Name && rm.ResourceName == rName.String() {
							// As of Kubernetes 1.31, CDI device IDs are not passed via annotations anymore.
							cdiDevices = append(cdiDevices, claimInfo.cdiDevicesAsList(rm.RequestName)...)
						}
					}
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return &ContainerInfo{CDIDevices: cdiDevices}, nil
}

// UnprepareResources calls a driver's NodeUnprepareResource API for each resource claim owned by a pod.
// This function is idempotent and may be called multiple times against the same pod.
// As such, calls to the underlying NodeUnprepareResource API are skipped for claims that have
// already been successfully unprepared.
func (m *Manager) UnprepareResources(ctx context.Context, pod *v1.Pod) error {
	startTime := time.Now()
	err := m.unprepareResourcesForPod(ctx, pod)
	kubeletmetrics.DRAOperationsDuration.WithLabelValues("UnprepareResources", strconv.FormatBool(err == nil)).Observe(time.Since(startTime).Seconds())
	if err != nil {
		return fmt.Errorf("unprepare dynamic resources: %w", err)
	}
	return nil
}

func (m *Manager) unprepareResourcesForPod(ctx context.Context, pod *v1.Pod) error {
	var claimNames []string
	for i := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			// No wrapping, the error is already informative.
			return err
		}
		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			continue
		}
		claimNames = append(claimNames, *claimName)
	}
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.DRAExtendedResource) {
		if pod.Status.ExtendedResourceClaimStatus != nil {
			claimNames = append(claimNames, pod.Status.ExtendedResourceClaimStatus.ResourceClaimName)
		}
	}

	return m.unprepareResources(ctx, pod.UID, pod.Namespace, claimNames)
}

func (m *Manager) unprepareResources(ctx context.Context, podUID types.UID, namespace string, claimNames []string) error {
	logger := klog.FromContext(ctx)
	batches := make(map[string][]*drapb.Claim)
	claimNamesMap := make(map[types.UID]string)
	for _, claimName := range claimNames {
		// Atomically perform some operations on the claimInfo cache.
		err := m.cache.withLock(func() error {
			// Get the claim info from the cache
			claimInfo, exists := m.cache.get(claimName, namespace)

			// Skip calling NodeUnprepareResource if claim info is not cached
			if !exists {
				return nil
			}

			// Skip calling NodeUnprepareResource if other pods are still referencing it
			if len(claimInfo.PodUIDs) > 1 {
				// We delay checkpointing of this change until
				// UnprepareResources returns successfully. It is OK to do
				// this because we will only return successfully from this call
				// if the checkpoint has succeeded. That means if the kubelet
				// is ever restarted before this checkpoint succeeds, we will
				// simply call into this (idempotent) function again.
				claimInfo.deletePodReference(podUID)
				return nil
			}

			// This claimInfo name will be used to update ClaimInfo cache
			// after NodeUnprepareResources GRPC succeeds
			claimNamesMap[claimInfo.ClaimUID] = claimInfo.ClaimName

			// Loop through all drivers and prepare for calling NodeUnprepareResources.
			claim := &drapb.Claim{
				Namespace: claimInfo.Namespace,
				Uid:       string(claimInfo.ClaimUID),
				Name:      claimInfo.ClaimName,
			}
			for driverName := range claimInfo.DriverState {
				batches[driverName] = append(batches[driverName], claim)
			}

			return nil
		})
		if err != nil {
			// No wrapping, this is the error above.
			return err
		}
	}

	// Call NodeUnprepareResources for all claims in each batch.
	// If there is any error, processing gets aborted.
	// We could try to continue, but that would make the code more complex.
	for driverName, claims := range batches {
		// Call NodeUnprepareResources RPC for all resource handles.
		plugin, err := m.draPlugins.GetPlugin(driverName)
		if plugin == nil {
			// No wrapping, error includes driver name already.
			return err
		}
		response, err := plugin.NodeUnprepareResources(ctx, &drapb.NodeUnprepareResourcesRequest{Claims: claims})
		if err != nil {
			// General error unrelated to any particular claim.
			return fmt.Errorf("NodeUnprepareResources: %w", err)
		}

		for claimUID, result := range response.Claims {
			reqClaim := lookupClaimRequest(claims, claimUID)
			if reqClaim == nil {
				return fmt.Errorf("NodeUnprepareResources returned result for unknown claim UID %s", claimUID)
			}
			if result.GetError() != "" {
				return fmt.Errorf("NodeUnprepareResources failed for ResourceClaim %s: %s", reqClaim.Name, result.Error)
			}
		}

		unfinished := len(claims) - len(response.Claims)
		if unfinished != 0 {
			return fmt.Errorf("NodeUnprepareResources skipped %d ResourceClaims", unfinished)
		}
	}

	// Atomically perform some operations on the claimInfo cache.
	err := m.cache.withLock(func() error {
		// TODO(#132978): Re-evaluate this logic to support post-mortem health updates.
		// As of the initial implementation, we immediately delete the claim info upon
		// unprepare. This means a late-arriving health update for a terminated pod
		// will be missed. A future enhancement could be to "tombstone" this entry for
		// a grace period instead of deleting it.

		// Delete all claimInfos from the cache that have just been unprepared.
		for _, claimName := range claimNamesMap {
			claimInfo, _ := m.cache.get(claimName, namespace)
			m.cache.delete(claimName, namespace)
			logger.V(6).Info("Deleted claim info cache entry", "claim", klog.KRef(namespace, claimName), "claimInfoEntry", claimInfo)
		}

		// Atomically sync the cache back to the checkpoint.
		if err := m.cache.syncToCheckpoint(); err != nil {
			return fmt.Errorf("checkpoint ResourceClaim state: %w", err)
		}
		return nil
	})
	if err != nil {
		// No wrapping, this is the error above.
		return err
	}

	return nil
}

// PodMightNeedToUnprepareResources returns true if the pod might need to
// unprepare resources
func (m *Manager) PodMightNeedToUnprepareResources(uid types.UID) bool {
	m.cache.Lock()
	defer m.cache.Unlock()
	return m.cache.hasPodReference(uid)
}

// GetContainerClaimInfos gets Container's ClaimInfo
func (m *Manager) GetContainerClaimInfos(pod *v1.Pod, container *v1.Container) ([]*ClaimInfo, error) {
	claimInfos := make([]*ClaimInfo, 0, len(pod.Spec.ResourceClaims))

	for i, podResourceClaim := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			// No wrapping, the error is already informative.
			return nil, err
		}

		if claimName == nil {
			// No ResourceClaim needed.
			continue
		}

		// Ownership doesn't get checked here, this should have been done before.

		for _, claim := range container.Resources.Claims {
			if podResourceClaim.Name != claim.Name {
				continue
			}

			err := m.cache.withRLock(func() error {
				claimInfo, exists := m.cache.get(*claimName, pod.Namespace)
				if !exists {
					return fmt.Errorf("unable to get information for ResourceClaim %s", *claimName)
				}
				claimInfos = append(claimInfos, claimInfo.DeepCopy())
				return nil
			})
			if err != nil {
				// No wrapping, this is the error above.
				return nil, err
			}
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.DRAExtendedResource) {
		// Handle the special claim for extended resources backed by DRA in the pod
		if pod.Status.ExtendedResourceClaimStatus != nil {
			var hasExtendedResourceClaim bool
			for _, n := range pod.Status.ExtendedResourceClaimStatus.RequestMappings {
				if n.ContainerName == container.Name {
					hasExtendedResourceClaim = true
					break
				}
			}
			if !hasExtendedResourceClaim {
				return claimInfos, nil
			}
			claimName := &pod.Status.ExtendedResourceClaimStatus.ResourceClaimName
			err := m.cache.withRLock(func() error {
				claimInfo, exists := m.cache.get(*claimName, pod.Namespace)
				if !exists {
					return fmt.Errorf("unable to get claim info for claim %s in namespace %s", *claimName, pod.Namespace)
				}
				claimInfos = append(claimInfos, claimInfo.DeepCopy())
				return nil
			})
			if err != nil {
				// No wrapping, this is the error above.
				return nil, err
			}
		}
	}
	return claimInfos, nil
}

// UpdateAllocatedResourcesStatus updates the health status of allocated DRA resources in the pod's container statuses.
func (m *Manager) UpdateAllocatedResourcesStatus(pod *v1.Pod, status *v1.PodStatus) {
	logger := klog.FromContext(context.Background())
	for i := range status.ContainerStatuses {
		containerStatus := &status.ContainerStatuses[i]

		// Find the corresponding container spec in the pod spec.
		var containerSpec *v1.Container
		for j := range pod.Spec.Containers {
			if pod.Spec.Containers[j].Name == containerStatus.Name {
				containerSpec = &pod.Spec.Containers[j]
				break
			}
		}

		// Skip if there's no matching container spec or it has no resource claims.
		if containerSpec == nil || len(containerSpec.Resources.Claims) == 0 {
			continue
		}

		resourceStatusMap := make(map[v1.ResourceName]*v1.ResourceStatus)
		if containerStatus.AllocatedResourcesStatus != nil {
			for j := range containerStatus.AllocatedResourcesStatus {
				resourceStatusMap[containerStatus.AllocatedResourcesStatus[j].Name] = &containerStatus.AllocatedResourcesStatus[j]
			}
		}

		// Iterate through the claims requested by this specific container.
		for _, claim := range containerSpec.Resources.Claims {
			// Find the actual name of the ResourceClaim object.
			var actualClaimName string
			for _, podClaimStatus := range pod.Status.ResourceClaimStatuses {
				if podClaimStatus.Name == claim.Name {
					if podClaimStatus.ResourceClaimName != nil {
						actualClaimName = *podClaimStatus.ResourceClaimName
					}
					break
				}
			}

			if actualClaimName == "" {
				logger.V(4).Info("Could not find generated name for resource claim in pod status", "pod", klog.KObj(pod), "container", containerSpec.Name, "claimName", claim.Name)
				continue
			}

			func() {
				m.cache.RLock()
				defer m.cache.RUnlock()

				// Use the actual claim name to look up claim info.
				claimInfo, exists := m.cache.get(actualClaimName, pod.Namespace)
				if !exists {
					logger.V(4).Info("Could not find claim info for resource claim", "pod", klog.KObj(pod), "container", containerSpec.Name, "claimName", actualClaimName)
					return
				}

				resourceName := v1.ResourceName(fmt.Sprintf("claim:%s", claim.Name))
				if claim.Request != "" {
					resourceName = v1.ResourceName(fmt.Sprintf("claim:%s/%s", claim.Name, claim.Request))
				}

				resStatus, ok := resourceStatusMap[resourceName]
				if !ok {
					newStatus := v1.ResourceStatus{
						Name:      resourceName,
						Resources: []v1.ResourceHealth{},
					}
					// Append and get a pointer to the new element.
					if containerStatus.AllocatedResourcesStatus == nil {
						containerStatus.AllocatedResourcesStatus = []v1.ResourceStatus{}
					}
					containerStatus.AllocatedResourcesStatus = append(containerStatus.AllocatedResourcesStatus, newStatus)
					resStatus = &containerStatus.AllocatedResourcesStatus[len(containerStatus.AllocatedResourcesStatus)-1]
					resourceStatusMap[resourceName] = resStatus
				}

				// Clear previous health entries before adding current ones.
				resStatus.Resources = []v1.ResourceHealth{}

				for driverName, driverState := range claimInfo.DriverState {
					for _, device := range driverState.Devices {
						healthStr := m.healthInfoCache.getHealthInfo(driverName, device.PoolName, device.DeviceName)

						var health v1.ResourceHealthStatus
						switch healthStr {
						case "Healthy":
							health = v1.ResourceHealthStatusHealthy
						case "Unhealthy":
							health = v1.ResourceHealthStatusUnhealthy
						default:
							health = v1.ResourceHealthStatusUnknown
						}

						resourceHealth := v1.ResourceHealth{Health: health}
						if len(device.CDIDeviceIDs) > 0 {
							resourceHealth.ResourceID = v1.ResourceID(device.CDIDeviceIDs[0])
						} else {
							resourceHealth.ResourceID = v1.ResourceID(fmt.Sprintf("%s/%s/%s", driverName, device.PoolName, device.DeviceName))
						}
						resStatus.Resources = append(resStatus.Resources, resourceHealth)
					}
				}
			}()
		}

		// Rebuild the slice from map to ensure correctness and remove empty entries.
		finalStatuses := make([]v1.ResourceStatus, 0, len(resourceStatusMap))
		for _, rs := range resourceStatusMap {
			if len(rs.Resources) > 0 {
				finalStatuses = append(finalStatuses, *rs)
			}
		}
		containerStatus.AllocatedResourcesStatus = finalStatuses
	}
}

// HandleWatchResourcesStream processes health updates from the DRA plugin.
func (m *Manager) HandleWatchResourcesStream(ctx context.Context, stream drahealthv1alpha1.DRAResourceHealth_NodeWatchResourcesClient, pluginName string) error {
	logger := klog.FromContext(ctx)

	defer func() {
		logger.V(4).Info("Clearing health cache for driver upon stream exit", "pluginName", pluginName)
		// Use a separate context for clearDriver if needed, though background should be fine.
		if err := m.healthInfoCache.clearDriver(pluginName); err != nil {
			logger.Error(err, "Failed to clear health info cache for driver", "pluginName", pluginName)
		}
	}()

	for {
		resp, err := stream.Recv()
		if err != nil {
			// Context canceled, normal shutdown.
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				logger.V(4).Info("Stopping health monitoring due to context cancellation", "pluginName", pluginName, "reason", err)
				return err
			}
			// Stream closed cleanly by the server, get normal EOF.
			if errors.Is(err, io.EOF) {
				logger.V(4).Info("Stream ended with EOF", "pluginName", pluginName)
				return nil
			}
			// Other errors are unexpected, log & return.
			logger.Error(err, "Error receiving from WatchResources stream", "pluginName", pluginName)
			return err
		}

		// Convert drahealthv1alpha1.DeviceHealth to state.DeviceHealth
		devices := make([]state.DeviceHealth, len(resp.GetDevices()))
		for i, d := range resp.GetDevices() {
			var health state.DeviceHealthStatus
			switch d.GetHealth() {
			case drahealthv1alpha1.HealthStatus_HEALTHY:
				health = state.DeviceHealthStatusHealthy
			case drahealthv1alpha1.HealthStatus_UNHEALTHY:
				health = state.DeviceHealthStatusUnhealthy
			default:
				health = state.DeviceHealthStatusUnknown
			}

			// Extract the health check timeout from the gRPC response
			// If not specified, zero, or negative, use the default timeout
			timeout := DefaultHealthTimeout
			timeoutSeconds := d.GetHealthCheckTimeoutSeconds()
			if timeoutSeconds > 0 {
				timeout = time.Duration(timeoutSeconds) * time.Second
			} else if timeoutSeconds < 0 {
				// Log warning for negative timeout values and use default
				logger.V(4).Info("Ignoring negative health check timeout, using default",
					"pluginName", pluginName,
					"poolName", d.GetDevice().GetPoolName(),
					"deviceName", d.GetDevice().GetDeviceName(),
					"providedTimeout", timeoutSeconds,
					"defaultTimeout", DefaultHealthTimeout)
			}

			devices[i] = state.DeviceHealth{
				PoolName:           d.GetDevice().GetPoolName(),
				DeviceName:         d.GetDevice().GetDeviceName(),
				Health:             health,
				LastUpdated:        time.Unix(d.GetLastUpdatedTime(), 0),
				HealthCheckTimeout: timeout,
			}
		}

		changedDevices, updateErr := m.healthInfoCache.updateHealthInfo(pluginName, devices)
		if updateErr != nil {
			logger.Error(updateErr, "Failed to update health info cache", "pluginName", pluginName)
		}
		if len(changedDevices) > 0 {
			logger.V(4).Info("Health info changed, checking affected pods", "pluginName", pluginName, "changedDevicesCount", len(changedDevices))

			podsToUpdate := sets.New[string]()

			m.cache.RLock()
			for _, dev := range changedDevices {
				for _, cInfo := range m.cache.claimInfo {
					if driverState, ok := cInfo.DriverState[pluginName]; ok {
						for _, allocatedDevice := range driverState.Devices {
							if allocatedDevice.PoolName == dev.PoolName && allocatedDevice.DeviceName == dev.DeviceName {
								podsToUpdate.Insert(cInfo.PodUIDs.UnsortedList()...)
								break
							}
						}
					}
				}
			}
			m.cache.RUnlock()

			if podsToUpdate.Len() > 0 {
				podUIDs := podsToUpdate.UnsortedList()
				logger.Info("Sending health update notification for pods", "pluginName", pluginName, "pods", podUIDs)
				select {
				case m.update <- resourceupdates.Update{PodUIDs: podUIDs}:
				default:
					logger.Error(nil, "DRA health update channel is full, discarding pod update notification", "pluginName", pluginName, "pods", podUIDs)
				}
			} else {
				logger.V(4).Info("Health info changed, but no active pods found using the affected devices", "pluginName", pluginName)
			}
		}

	}
}

// Updates returns the channel that provides resource updates.
func (m *Manager) Updates() <-chan resourceupdates.Update {
	// Return the internal channel that HandleWatchResourcesStream writes to.
	return m.update
}
