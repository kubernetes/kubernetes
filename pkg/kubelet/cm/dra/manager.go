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
	"fmt"
	"strconv"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	dra "k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/resourceupdates"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
)

// draManagerStateFileName is the file name where dra manager stores its state
const draManagerStateFileName = "dra_manager_state"

// defaultReconcilePeriod is the default reconciliation period to keep all claim info state in sync.
const defaultReconcilePeriod = 60 * time.Second

// ActivePodsFunc is a function that returns a list of pods to reconcile.
type ActivePodsFunc func() []*v1.Pod

// GetNodeFunc is a function that returns the node object using the kubelet's node lister.
type GetNodeFunc func() (*v1.Node, error)

// ManagerImpl is the structure in charge of managing DRA drivers.
type ManagerImpl struct {
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

	// getNode is a function that returns the node object using the kubelet's node lister.
	getNode GetNodeFunc

	// healthInfoCache contains cached health info
	healthInfoCache *healthInfoCache

	// lastHealthUpdate is the last time the health info cache was updated.
	lastHealthUpdate time.Time

	// healthInfoMutex protects the healthInfoCache and lastHealthUpdate.
	healthInfoMutex sync.Mutex

	update chan resourceupdates.Update
}

// NewManagerImpl creates a new manager.
func NewManagerImpl(kubeClient clientset.Interface, stateFileDirectory string, nodeName types.NodeName) (*ManagerImpl, error) {
	claimInfoCache, err := newClaimInfoCache(stateFileDirectory, draManagerStateFileName)
	if err != nil {
		return nil, fmt.Errorf("failed to create claimInfo cache: %w", err)
	}

	healthInfoCache, err := newHealthInfoCache(stateFileDirectory + "/" + "dra_health_state")
	if err != nil {
		return nil, fmt.Errorf("failed to create healthInfo cache: %w", err)
	}

	// TODO: for now the reconcile period is not configurable.
	// We should consider making it configurable in the future.
	reconcilePeriod := defaultReconcilePeriod

	manager := &ManagerImpl{
		cache:            claimInfoCache,
		kubeClient:       kubeClient,
		reconcilePeriod:  reconcilePeriod,
		activePods:       nil,
		sourcesReady:     nil,
		healthInfoCache:  healthInfoCache,
		lastHealthUpdate: time.Time{},
		healthInfoMutex:  sync.Mutex{},
		update:           make(chan resourceupdates.Update, 100),
	}

	return manager, nil
}

func (m *ManagerImpl) GetWatcherHandler() cache.PluginHandler {
	rootCtx := context.Background()
	return dra.NewRegistrationHandler(rootCtx, m.kubeClient, m.getNode, m)
}

// Start starts the reconcile loop of the manager.
func (m *ManagerImpl) Start(ctx context.Context, activePods ActivePodsFunc, getNode GetNodeFunc, sourcesReady config.SourcesReady) error {
	logger := klog.FromContext(ctx)
	m.activePods = activePods
	m.getNode = getNode
	m.sourcesReady = sourcesReady

	logger.Info("Starting DRA manager")

	go wait.UntilWithContext(ctx, func(ctx context.Context) { m.reconcileLoop(ctx) }, m.reconcilePeriod)
	return nil
}

// reconcileLoop ensures that any stale state in the manager's claimInfoCache gets periodically reconciled.
func (m *ManagerImpl) reconcileLoop(ctx context.Context) {
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
func (m *ManagerImpl) PrepareResources(ctx context.Context, pod *v1.Pod) error {
	startTime := time.Now()
	err := m.prepareResources(ctx, pod)
	metrics.DRAOperationsDuration.WithLabelValues("PrepareResources", strconv.FormatBool(err == nil)).Observe(time.Since(startTime).Seconds())
	return err
}

func (m *ManagerImpl) prepareResources(ctx context.Context, pod *v1.Pod) error {
	logger := klog.FromContext(ctx)
	batches := make(map[string][]*drapb.Claim)
	resourceClaims := make(map[types.UID]*resourceapi.ResourceClaim)
	for i := range pod.Spec.ResourceClaims {
		podClaim := &pod.Spec.ResourceClaims[i]
		logger.V(3).Info("Processing resource", "pod", klog.KObj(pod), "podClaim", podClaim.Name)
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, podClaim)
		if err != nil {
			return fmt.Errorf("prepare resource claim: %w", err)
		}

		if claimName == nil {
			// Nothing to do.
			logger.V(5).Info("No need to prepare resources, no claim generated", "pod", klog.KObj(pod), "podClaim", podClaim.Name)
			continue
		}
		// Query claim object from the API server
		resourceClaim, err := m.kubeClient.ResourceV1beta1().ResourceClaims(pod.Namespace).Get(
			ctx,
			*claimName,
			metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to fetch ResourceClaim %s referenced by pod %s: %w", *claimName, pod.Name, err)
		}

		if mustCheckOwner {
			if err = resourceclaim.IsForPod(pod, resourceClaim); err != nil {
				return err
			}
		}

		// Check if pod is in the ReservedFor for the claim
		if !resourceclaim.IsReservedForPod(pod, resourceClaim) {
			return fmt.Errorf("pod %s(%s) is not allowed to use resource claim %s(%s)",
				pod.Name, pod.UID, *claimName, resourceClaim.UID)
		}

		// Atomically perform some operations on the claimInfo cache.
		err = m.cache.withLock(func() error {
			// Get a reference to the claim info for this claim from the cache.
			// If there isn't one yet, then add it to the cache.
			claimInfo, exists := m.cache.get(resourceClaim.Name, resourceClaim.Namespace)
			if !exists {
				ci, err := newClaimInfoFromClaim(resourceClaim)
				if err != nil {
					return fmt.Errorf("claim %s: %w", klog.KObj(resourceClaim), err)
				}
				claimInfo = m.cache.add(ci)
				logger.V(6).Info("Created new claim info cache entry", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim), "claimInfoEntry", claimInfo)
			} else {
				logger.V(6).Info("Found existing claim info cache entry", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim), "claimInfoEntry", claimInfo)
			}

			// Add a reference to the current pod in the claim info.
			claimInfo.addPodReference(pod.UID)

			// Checkpoint to ensure all claims we plan to prepare are tracked.
			// If something goes wrong and the newly referenced pod gets
			// deleted without a successful prepare call, we will catch
			// that in the reconcile loop and take the appropriate action.
			if err := m.cache.syncToCheckpoint(); err != nil {
				return fmt.Errorf("failed to checkpoint claimInfo state: %w", err)
			}

			// If this claim is already prepared, there is no need to prepare it again.
			if claimInfo.isPrepared() {
				logger.V(5).Info("Resources already prepared", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim))
				return nil
			}

			// This saved claim will be used to update ClaimInfo cache
			// after NodePrepareResources GRPC succeeds
			resourceClaims[claimInfo.ClaimUID] = resourceClaim

			// Loop through all drivers and prepare for calling NodePrepareResources.
			claim := &drapb.Claim{
				Namespace: claimInfo.Namespace,
				UID:       string(claimInfo.ClaimUID),
				Name:      claimInfo.ClaimName,
			}
			for driverName := range claimInfo.DriverState {
				batches[driverName] = append(batches[driverName], claim)
			}

			return nil
		})
		if err != nil {
			return fmt.Errorf("locked cache operation: %w", err)
		}
	}

	// Call NodePrepareResources for all claims in each batch.
	// If there is any error, processing gets aborted.
	// We could try to continue, but that would make the code more complex.
	for driverName, claims := range batches {
		// Call NodePrepareResources RPC for all resource handles.
		client, err := dra.NewDRAPluginClient(driverName)
		if err != nil {
			return fmt.Errorf("failed to get gRPC client for driver %s: %w", driverName, err)
		}
		response, err := client.NodePrepareResources(ctx, &drapb.NodePrepareResourcesRequest{Claims: claims})
		if err != nil {
			// General error unrelated to any particular claim.
			return fmt.Errorf("NodePrepareResources failed: %w", err)
		}
		for claimUID, result := range response.Claims {
			reqClaim := lookupClaimRequest(claims, claimUID)
			if reqClaim == nil {
				return fmt.Errorf("NodePrepareResources returned result for unknown claim UID %s", claimUID)
			}
			if result.GetError() != "" {
				return fmt.Errorf("NodePrepareResources failed for claim %s/%s: %s", reqClaim.Namespace, reqClaim.Name, result.Error)
			}

			claim := resourceClaims[types.UID(claimUID)]

			// Add the prepared CDI devices to the claim info
			err := m.cache.withLock(func() error {
				info, exists := m.cache.get(claim.Name, claim.Namespace)
				if !exists {
					return fmt.Errorf("unable to get claim info for claim %s in namespace %s", claim.Name, claim.Namespace)
				}
				for _, device := range result.GetDevices() {
					info.addDevice(driverName, state.Device{PoolName: device.PoolName, DeviceName: device.DeviceName, RequestNames: device.RequestNames, CDIDeviceIDs: device.CDIDeviceIDs})
				}
				return nil
			})
			if err != nil {
				return fmt.Errorf("locked cache operation: %w", err)
			}
		}

		unfinished := len(claims) - len(response.Claims)
		if unfinished != 0 {
			return fmt.Errorf("NodePrepareResources left out %d claims", unfinished)
		}
	}

	// Atomically perform some operations on the claimInfo cache.
	err := m.cache.withLock(func() error {
		// Mark all pod claims as prepared.
		for _, claim := range resourceClaims {
			info, exists := m.cache.get(claim.Name, claim.Namespace)
			if !exists {
				return fmt.Errorf("unable to get claim info for claim %s in namespace %s", claim.Name, claim.Namespace)
			}
			info.setPrepared()
		}

		// Checkpoint to ensure all prepared claims are tracked with their list
		// of CDI devices attached.
		if err := m.cache.syncToCheckpoint(); err != nil {
			return fmt.Errorf("failed to checkpoint claimInfo state: %w", err)
		}

		return nil
	})
	if err != nil {
		return fmt.Errorf("locked cache operation: %w", err)
	}

	return nil
}

func lookupClaimRequest(claims []*drapb.Claim, claimUID string) *drapb.Claim {
	for _, claim := range claims {
		if claim.UID == claimUID {
			return claim
		}
	}
	return nil
}

// GetResources gets a ContainerInfo object from the claimInfo cache.
// This information is used by the caller to update a container config.
func (m *ManagerImpl) GetResources(pod *v1.Pod, container *v1.Container) (*ContainerInfo, error) {
	cdiDevices := []kubecontainer.CDIDevice{}

	for i := range pod.Spec.ResourceClaims {
		podClaim := &pod.Spec.ResourceClaims[i]
		claimName, _, err := resourceclaim.Name(pod, podClaim)
		if err != nil {
			return nil, fmt.Errorf("list resource claims: %w", err)
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
					return fmt.Errorf("unable to get claim info for claim %s in namespace %s", *claimName, pod.Namespace)
				}

				// As of Kubernetes 1.31, CDI device IDs are not passed via annotations anymore.
				cdiDevices = append(cdiDevices, claimInfo.cdiDevicesAsList(claim.Request)...)

				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("locked cache operation: %w", err)
			}
		}
	}
	return &ContainerInfo{CDIDevices: cdiDevices}, nil
}

// UnprepareResources calls a driver's NodeUnprepareResource API for each resource claim owned by a pod.
// This function is idempotent and may be called multiple times against the same pod.
// As such, calls to the underlying NodeUnprepareResource API are skipped for claims that have
// already been successfully unprepared.
func (m *ManagerImpl) UnprepareResources(ctx context.Context, pod *v1.Pod) error {
	var err error = nil
	defer func(startTime time.Time) {
		metrics.DRAOperationsDuration.WithLabelValues("UnprepareResources", strconv.FormatBool(err != nil)).Observe(time.Since(startTime).Seconds())
	}(time.Now())
	var claimNames []string
	for i := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			return fmt.Errorf("unprepare resource claim: %w", err)
		}
		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			continue
		}
		claimNames = append(claimNames, *claimName)
	}
	err = m.unprepareResources(ctx, pod.UID, pod.Namespace, claimNames)
	return err
}

func (m *ManagerImpl) unprepareResources(ctx context.Context, podUID types.UID, namespace string, claimNames []string) error {
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
				UID:       string(claimInfo.ClaimUID),
				Name:      claimInfo.ClaimName,
			}
			for driverName := range claimInfo.DriverState {
				batches[driverName] = append(batches[driverName], claim)
			}

			return nil
		})
		if err != nil {
			return fmt.Errorf("locked cache operation: %w", err)
		}
	}

	// Call NodeUnprepareResources for all claims in each batch.
	// If there is any error, processing gets aborted.
	// We could try to continue, but that would make the code more complex.
	for driverName, claims := range batches {
		// Call NodeUnprepareResources RPC for all resource handles.
		client, err := dra.NewDRAPluginClient(driverName)
		if err != nil {
			return fmt.Errorf("get gRPC client for DRA driver %s: %w", driverName, err)
		}
		response, err := client.NodeUnprepareResources(ctx, &drapb.NodeUnprepareResourcesRequest{Claims: claims})
		if err != nil {
			// General error unrelated to any particular claim.
			return fmt.Errorf("NodeUnprepareResources failed: %w", err)
		}

		for claimUID, result := range response.Claims {
			reqClaim := lookupClaimRequest(claims, claimUID)
			if reqClaim == nil {
				return fmt.Errorf("NodeUnprepareResources returned result for unknown claim UID %s", claimUID)
			}
			if result.GetError() != "" {
				return fmt.Errorf("NodeUnprepareResources failed for claim %s/%s: %s", reqClaim.Namespace, reqClaim.Name, result.Error)
			}
		}

		unfinished := len(claims) - len(response.Claims)
		if unfinished != 0 {
			return fmt.Errorf("NodeUnprepareResources left out %d claims", unfinished)
		}
	}

	// Atomically perform some operations on the claimInfo cache.
	err := m.cache.withLock(func() error {
		// Delete all claimInfos from the cache that have just been unprepared.
		for _, claimName := range claimNamesMap {
			claimInfo, _ := m.cache.get(claimName, namespace)
			m.cache.delete(claimName, namespace)
			logger.V(6).Info("Deleted claim info cache entry", "claim", klog.KRef(namespace, claimName), "claimInfoEntry", claimInfo)
		}

		// Atomically sync the cache back to the checkpoint.
		if err := m.cache.syncToCheckpoint(); err != nil {
			return fmt.Errorf("failed to checkpoint claimInfo state: %w", err)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("locked cache operation: %w", err)
	}

	return nil
}

// PodMightNeedToUnprepareResources returns true if the pod might need to
// unprepare resources
func (m *ManagerImpl) PodMightNeedToUnprepareResources(uid types.UID) bool {
	m.cache.Lock()
	defer m.cache.Unlock()
	return m.cache.hasPodReference(uid)
}

// GetContainerClaimInfos gets Container's ClaimInfo
func (m *ManagerImpl) GetContainerClaimInfos(pod *v1.Pod, container *v1.Container) ([]*ClaimInfo, error) {
	claimInfos := make([]*ClaimInfo, 0, len(pod.Spec.ResourceClaims))

	for i, podResourceClaim := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			return nil, fmt.Errorf("determine resource claim information: %w", err)
		}

		for _, claim := range container.Resources.Claims {
			if podResourceClaim.Name != claim.Name {
				continue
			}

			err := m.cache.withRLock(func() error {
				claimInfo, exists := m.cache.get(*claimName, pod.Namespace)
				if !exists {
					return fmt.Errorf("unable to get claim info for claim %s in namespace %s", *claimName, pod.Namespace)
				}
				claimInfos = append(claimInfos, claimInfo.DeepCopy())
				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("locked cache operation: %w", err)
			}
		}
	}
	return claimInfos, nil
}

// UpdateAllocatedResourcesStatus updates the health status of allocated DRA resources in the pod's container statuses.
func (m *ManagerImpl) UpdateAllocatedResourcesStatus(pod *v1.Pod, status *v1.PodStatus) {
	logger := klog.FromContext(context.Background())
	for _, container := range pod.Spec.Containers {
		// Get all the DRA claim details associated with this specific container.
		claimInfos, err := m.GetContainerClaimInfos(pod, &container)
		if err != nil {
			logger.Error(err, "Failed to get claim infos for container", "pod", klog.KObj(pod), "container", container.Name)
			continue
		}

		// Find the corresponding container status
		for i, containerStatus := range status.ContainerStatuses {
			if containerStatus.Name != container.Name {
				continue
			}

			// Initialize AllocatedResourcesStatus if nil
			if containerStatus.AllocatedResourcesStatus == nil {
				status.ContainerStatuses[i].AllocatedResourcesStatus = []v1.ResourceStatus{}
			}

			// Loop through each claim associated with the container
			for _, claimInfo := range claimInfos {
				// Iterate through the map holding the state specific to each driver
				for driverName, driverState := range claimInfo.DriverState {
					// Iterate through each specific device allocated by this driver
					for _, device := range driverState.Devices {
						m.healthInfoMutex.Lock()
						health := m.healthInfoCache.getHealthInfo(driverName, device.PoolName, device.DeviceName)
						m.healthInfoMutex.Unlock()

						// Create resource status
						resourceStatus := v1.ResourceStatus{
							Name: claimInfo.ClaimName,
							Health: v1.ResourceHealth{
								Health: string(health),
							},
						}
						if len(device.CDIDeviceIDs) > 0 {
							resourceStatus.ID = device.CDIDeviceIDs[0]
						}

						// Update or append to AllocatedResourcesStatus
						found := false
						for j, existing := range status.ContainerStatuses[i].AllocatedResourcesStatus {
							if existing.Name == resourceStatus.Name {
								status.ContainerStatuses[i].AllocatedResourcesStatus[j] = resourceStatus
								found = true
								break
							}
						}
						if !found {
							status.ContainerStatuses[i].AllocatedResourcesStatus = append(status.ContainerStatuses[i].AllocatedResourcesStatus, resourceStatus)
						}
					}
				}
			}
		}
	}
}

// watchResources starts a health monitoring stream for a DRA plugin.
func (m *ManagerImpl) watchResources(ctx context.Context, pluginName string, p *dra.Plugin) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Starting to watch resources for plugin", "pluginName", pluginName)

	stream, err := p.WatchResources(ctx)
	if err != nil {
		logger.Error(err, "Failed to establish WatchResources stream", "pluginName", pluginName)
		return err
	}
	return m.HandleWatchResourcesStream(ctx, stream, pluginName)
}

// HandleWatchResourcesStream processes health updates from the DRA plugin.
func (m *ManagerImpl) HandleWatchResourcesStream(ctx context.Context, stream drahealthv1alpha1.NodeHealth_WatchResourcesClient, pluginName string) error {
	logger := klog.FromContext(ctx)
	go func() {
		for {
			select {
			case <-ctx.Done():
				logger.Info("Stopping health monitoring", "pluginName", pluginName)
				return
			default:
				resp, err := stream.Recv()
				if err != nil {
					logger.Error(err, "Error receiving from WatchResources stream", "pluginName", pluginName)
					return
				}
				// Convert drahealthv1alpha1.DeviceHealth to state.DeviceHealth
				devices := make([]state.DeviceHealth, len(resp.GetDevices()))
				for i, d := range resp.GetDevices() {
					devices[i] = state.DeviceHealth{
						PoolName:    d.PoolName,
						DeviceName:  d.DeviceName,
						Health:      state.DeviceHealthString(d.Health),
						LastUpdated: time.Unix(d.LastUpdated, 0),
					}
				}

				changedDevices, changed, updateErr := m.healthInfoCache.updateHealthInfo(pluginName, devices)
				if updateErr != nil {
					logger.Error(updateErr, "Failed to update health info cache", "pluginName", pluginName)
				}
				if changed && len(changedDevices) > 0 {
					logger.V(5).Info("Health info changed, checking affected pods", "pluginName", pluginName, "changedDevicesCount", len(changedDevices))

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
						logger.V(4).Info("Sending health update notification for pods", "pluginName", pluginName, "pods", podUIDs)
						select {
						case m.update <- resourceupdates.Update{PodUIDs: podUIDs}:
						default:
							logger.Error(nil, "DRA health update channel is full, discarding pod update notification", "pluginName", pluginName, "pods", podUIDs)
						}
					} else {
						logger.V(5).Info("Health info changed, but no active pods found using the affected devices", "pluginName", pluginName)
					}
				} else if changed {
					logger.V(5).Info("Health info updated, but no specific device changes detected", "pluginName", pluginName)
				}

			}
		}
	}()
	return nil
}
