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
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1alpha4"
	dra "k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// draManagerStateFileName is the file name where dra manager stores its state
const draManagerStateFileName = "dra_manager_state"

// defaultReconcilePeriod is the default reconciliation period to keep all claim info state in sync.
const defaultReconcilePeriod = 60 * time.Second

// ActivePodsFunc is a function that returns a list of pods to reconcile.
type ActivePodsFunc func() []*v1.Pod

// ManagerImpl is the structure in charge of managing DRA resource Plugins.
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
}

// NewManagerImpl creates a new manager.
func NewManagerImpl(kubeClient clientset.Interface, stateFileDirectory string, nodeName types.NodeName) (*ManagerImpl, error) {
	klog.V(2).InfoS("Creating DRA manager")

	claimInfoCache, err := newClaimInfoCache(stateFileDirectory, draManagerStateFileName)
	if err != nil {
		return nil, fmt.Errorf("failed to create claimInfo cache: %+v", err)
	}

	// TODO: for now the reconcile period is not configurable.
	// We should consider making it configurable in the future.
	reconcilePeriod := defaultReconcilePeriod

	manager := &ManagerImpl{
		cache:           claimInfoCache,
		kubeClient:      kubeClient,
		reconcilePeriod: reconcilePeriod,
		activePods:      nil,
		sourcesReady:    nil,
	}

	return manager, nil
}

// Start starts the reconcile loop of the manager.
func (m *ManagerImpl) Start(activePods ActivePodsFunc, sourcesReady config.SourcesReady) error {
	m.activePods = activePods
	m.sourcesReady = sourcesReady
	go wait.Until(func() { m.reconcileLoop() }, m.reconcilePeriod, wait.NeverStop)
	return nil
}

// reconcileLoop ensures that any stale state in the manager's claimInfoCache gets periodically reconciled.
func (m *ManagerImpl) reconcileLoop() {
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
		if err := m.unprepareResources(podClaims.uid, podClaims.namespace, podClaims.claimNames); err != nil {
			klog.ErrorS(err, "Unpreparing pod resources in reconcile loop", "podUID", podClaims.uid)
		}
	}
}

// PrepareResources attempts to prepare all of the required resource
// plugin resources for the input container, issue NodePrepareResources rpc requests
// for each new resource requirement, process their responses and update the cached
// containerResources on success.
func (m *ManagerImpl) PrepareResources(pod *v1.Pod) error {
	batches := make(map[string][]*drapb.Claim)
	resourceClaims := make(map[types.UID]*resourceapi.ResourceClaim)
	for i := range pod.Spec.ResourceClaims {
		podClaim := &pod.Spec.ResourceClaims[i]
		klog.V(3).InfoS("Processing resource", "pod", klog.KObj(pod), "podClaim", podClaim.Name)
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, podClaim)
		if err != nil {
			return fmt.Errorf("prepare resource claim: %v", err)
		}

		if claimName == nil {
			// Nothing to do.
			klog.V(5).InfoS("No need to prepare resources, no claim generated", "pod", klog.KObj(pod), "podClaim", podClaim.Name)
			continue
		}
		// Query claim object from the API server
		resourceClaim, err := m.kubeClient.ResourceV1alpha3().ResourceClaims(pod.Namespace).Get(
			context.TODO(),
			*claimName,
			metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to fetch ResourceClaim %s referenced by pod %s: %+v", *claimName, pod.Name, err)
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
					return fmt.Errorf("claim %s: %v", klog.KObj(resourceClaim), err)
				}
				claimInfo = m.cache.add(ci)
				klog.V(6).InfoS("Created new claim info cache entry", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim), "claimInfoEntry", claimInfo)
			} else {
				klog.V(6).InfoS("Found existing claim info cache entry", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim), "claimInfoEntry", claimInfo)
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
				klog.V(5).InfoS("Resources already prepared", "pod", klog.KObj(pod), "podClaim", podClaim.Name, "claim", klog.KObj(resourceClaim))
				return nil
			}

			// This saved claim will be used to update ClaimInfo cache
			// after NodePrepareResources GRPC succeeds
			resourceClaims[claimInfo.ClaimUID] = resourceClaim

			// Loop through all plugins and prepare for calling NodePrepareResources.
			claim := &drapb.Claim{
				Namespace: claimInfo.Namespace,
				Uid:       string(claimInfo.ClaimUID),
				Name:      claimInfo.ClaimName,
			}
			for driverName := range claimInfo.Drivers {
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
			return fmt.Errorf("failed to get DRA Plugin client for plugin name %s: %v", driverName, err)
		}
		response, err := client.NodePrepareResources(context.Background(), &drapb.NodePrepareResourcesRequest{Claims: claims})
		if err != nil {
			// General error unrelated to any particular claim.
			return fmt.Errorf("NodePrepareResources failed: %v", err)
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
				for _, device := range result.GetCDIDevices() {
					info.setCDIDevices(driverName, device.RequestName, device.CdiIds)
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
		if claim.Uid == claimUID {
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
			return nil, fmt.Errorf("list resource claims: %v", err)
		}
		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			klog.V(5).InfoS("No CDI devices, no claim generated", "pod", klog.KObj(pod), "podClaimName", podClaim.Name)
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
				for _, driverData := range claimInfo.Drivers {
					for requestName, devices := range driverData.CDIDevices {
						claimRequestName := claim.Request
						if claimRequestName == "" || claimRequestName == requestName {
							for _, deviceID := range devices {
								cdiDevices = append(cdiDevices, kubecontainer.CDIDevice{Name: deviceID})
							}
						}
					}
				}

				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("locked cache operation: %w", err)
			}
		}
	}

	klog.V(5).InfoS("Determined CDI devices for pod", "pod", klog.KObj(pod), "cdiDevices", cdiDevices)
	return &ContainerInfo{CDIDevices: cdiDevices}, nil
}

// UnprepareResources calls a plugin's NodeUnprepareResource API for each resource claim owned by a pod.
// This function is idempotent and may be called multiple times against the same pod.
// As such, calls to the underlying NodeUnprepareResource API are skipped for claims that have
// already been successfully unprepared.
func (m *ManagerImpl) UnprepareResources(pod *v1.Pod) error {
	var claimNames []string
	for i := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			return fmt.Errorf("unprepare resource claim: %v", err)
		}
		// The claim name might be nil if no underlying resource claim
		// was generated for the referenced claim. There are valid use
		// cases when this might happen, so we simply skip it.
		if claimName == nil {
			continue
		}
		claimNames = append(claimNames, *claimName)
	}
	return m.unprepareResources(pod.UID, pod.Namespace, claimNames)
}

func (m *ManagerImpl) unprepareResources(podUID types.UID, namespace string, claimNames []string) error {
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

			// Loop through all plugins and prepare for calling NodeUnprepareResources.
			claim := &drapb.Claim{
				Namespace: claimInfo.Namespace,
				Uid:       string(claimInfo.ClaimUID),
				Name:      claimInfo.ClaimName,
			}
			for driverName := range claimInfo.Drivers {
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
			return fmt.Errorf("get gRPC client for DRA driver %s: %v", driverName, err)
		}
		response, err := client.NodeUnprepareResources(context.Background(), &drapb.NodeUnprepareResourcesRequest{Claims: claims})
		if err != nil {
			// General error unrelated to any particular claim.
			return fmt.Errorf("NodeUnprepareResources failed: %v", err)
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
			klog.V(6).InfoS("Deleted claim info cache entry", "claim", klog.KRef(namespace, claimName), "claimInfoEntry", claimInfo)
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
func (m *ManagerImpl) PodMightNeedToUnprepareResources(UID types.UID) bool {
	m.cache.Lock()
	defer m.cache.Unlock()
	return m.cache.hasPodReference(UID)
}

// GetContainerClaimInfos gets Container's ClaimInfo
func (m *ManagerImpl) GetContainerClaimInfos(pod *v1.Pod, container *v1.Container) ([]*ClaimInfo, error) {
	claimInfos := make([]*ClaimInfo, 0, len(pod.Spec.ResourceClaims))

	for i, podResourceClaim := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &pod.Spec.ResourceClaims[i])
		if err != nil {
			return nil, fmt.Errorf("determine resource claim information: %v", err)
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
