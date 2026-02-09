/*
Copyright 2016 The Kubernetes Authors.

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

package eviction

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/utils/clock"

	resourcehelper "k8s.io/component-helpers/resource"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	podCleanupTimeout  = 30 * time.Second
	podCleanupPollFreq = time.Second
)

const (
	// signalEphemeralContainerFsLimit is amount of storage available on filesystem requested by the container
	signalEphemeralContainerFsLimit string = "ephemeralcontainerfs.limit"
	// signalEphemeralPodFsLimit is amount of storage available on filesystem requested by the pod
	signalEphemeralPodFsLimit string = "ephemeralpodfs.limit"
	// signalEmptyDirFsLimit is amount of storage available on filesystem requested by an emptyDir
	signalEmptyDirFsLimit string = "emptydirfs.limit"
	// immediateEvictionGracePeriodSeconds is how long we give pods to shut down when we
	// need them to evict quickly due to resource pressure
	immediateEvictionGracePeriodSeconds = 1
)

// managerImpl implements Manager
type managerImpl struct {
	//  used to track time
	clock clock.WithTicker
	// config is how the manager is configured
	config Config
	// the function to invoke to kill a pod
	killPodFunc KillPodFunc
	// the interface that knows how to do image gc
	imageGC ImageGC
	// the interface that knows how to do container gc
	containerGC ContainerGC
	// protects access to internal state
	sync.RWMutex
	// node conditions are the set of conditions present
	nodeConditions []v1.NodeConditionType
	// captures when a node condition was last observed based on a threshold being met
	nodeConditionsLastObservedAt nodeConditionsObservedAt
	// nodeRef is a reference to the node
	nodeRef *v1.ObjectReference
	// used to record events about the node
	recorder record.EventRecorder
	// used to measure usage stats on system
	summaryProvider stats.SummaryProvider
	// records when a threshold was first observed
	thresholdsFirstObservedAt thresholdsObservedAt
	// records the set of thresholds that have been met (including graceperiod) but not yet resolved
	thresholdsMet []evictionapi.Threshold
	// signalToRankFunc maps a resource to ranking function for that resource.
	signalToRankFunc map[evictionapi.Signal]rankFunc
	// signalToNodeReclaimFuncs maps a resource to an ordered list of functions that know how to reclaim that resource.
	signalToNodeReclaimFuncs map[evictionapi.Signal]nodeReclaimFuncs
	// last observations from synchronize
	lastObservations signalObservations
	// dedicatedImageFs indicates if imagefs is on a separate device from the rootfs
	dedicatedImageFs *bool
	// splitContainerImageFs indicates if containerfs is on a separate device from imagefs
	splitContainerImageFs *bool
	// thresholdNotifiers is a list of memory threshold notifiers which each notify for a memory eviction threshold
	thresholdNotifiers []ThresholdNotifier
	// thresholdsLastUpdated is the last time the thresholdNotifiers were updated.
	thresholdsLastUpdated time.Time
	// whether can support local storage capacity isolation
	localStorageCapacityIsolation bool
}

// ensure it implements the required interface
var _ Manager = &managerImpl{}

// NewManager returns a configured Manager and an associated admission handler to enforce eviction configuration.
func NewManager(
	summaryProvider stats.SummaryProvider,
	config Config,
	killPodFunc KillPodFunc,
	imageGC ImageGC,
	containerGC ContainerGC,
	recorder record.EventRecorder,
	nodeRef *v1.ObjectReference,
	clock clock.WithTicker,
	localStorageCapacityIsolation bool,
) (Manager, lifecycle.PodAdmitHandler) {
	manager := &managerImpl{
		clock:                         clock,
		killPodFunc:                   killPodFunc,
		imageGC:                       imageGC,
		containerGC:                   containerGC,
		config:                        config,
		recorder:                      recorder,
		summaryProvider:               summaryProvider,
		nodeRef:                       nodeRef,
		nodeConditionsLastObservedAt:  nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:     thresholdsObservedAt{},
		dedicatedImageFs:              nil,
		splitContainerImageFs:         nil,
		thresholdNotifiers:            []ThresholdNotifier{},
		localStorageCapacityIsolation: localStorageCapacityIsolation,
	}
	return manager, manager
}

// Admit rejects a pod if its not safe to admit for node stability.
func (m *managerImpl) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	m.RLock()
	defer m.RUnlock()

	ctx := context.Background()
	logger := klog.FromContext(ctx)

	if len(m.nodeConditions) == 0 {
		return lifecycle.PodAdmitResult{Admit: true}
	}
	// Admit Critical pods even under resource pressure since they are required for system stability.
	// https://github.com/kubernetes/kubernetes/issues/40573 has more details.
	if kubelettypes.IsCriticalPod(attrs.Pod) {
		return lifecycle.PodAdmitResult{Admit: true}
	}

	// Conditions other than memory pressure reject all pods
	nodeOnlyHasMemoryPressureCondition := hasNodeCondition(m.nodeConditions, v1.NodeMemoryPressure) && len(m.nodeConditions) == 1
	if nodeOnlyHasMemoryPressureCondition {
		notBestEffort := v1.PodQOSBestEffort != v1qos.GetPodQOS(attrs.Pod)
		if notBestEffort {
			return lifecycle.PodAdmitResult{Admit: true}
		}

		// When node has memory pressure, check BestEffort Pod's toleration:
		// admit it if tolerates memory pressure taint, fail for other tolerations, e.g. DiskPressure.
		if corev1helpers.TolerationsTolerateTaint(logger, attrs.Pod.Spec.Tolerations, &v1.Taint{
			Key:    v1.TaintNodeMemoryPressure,
			Effect: v1.TaintEffectNoSchedule,
		}, utilfeature.DefaultFeatureGate.Enabled(features.TaintTolerationComparisonOperators)) {
			return lifecycle.PodAdmitResult{Admit: true}
		}
	}

	return lifecycle.PodAdmitResult{
		Admit:   false,
		Reason:  Reason,
		Message: fmt.Sprintf(nodeConditionMessageFmt, m.nodeConditions),
	}
}

// Start starts the control loop to observe and response to low compute resources.
func (m *managerImpl) Start(ctx context.Context, diskInfoProvider DiskInfoProvider, podFunc ActivePodsFunc, podCleanedUpFunc PodCleanedUpFunc, monitoringInterval time.Duration) {
	logger := klog.FromContext(ctx)
	thresholdHandler := func(message string) {
		logger.Info(message)
		_, _ = m.synchronize(ctx, diskInfoProvider, podFunc)
	}
	logger.Info("Eviction manager: starting control loop")
	if m.config.KernelMemcgNotification || runtime.GOOS == "windows" {
		for _, threshold := range m.config.Thresholds {
			if threshold.Signal == evictionapi.SignalMemoryAvailable || threshold.Signal == evictionapi.SignalAllocatableMemoryAvailable {
				notifier, err := NewMemoryThresholdNotifier(logger, threshold, m.config.PodCgroupRoot, &CgroupNotifierFactory{}, thresholdHandler)
				if err != nil {
					logger.Info("Eviction manager: failed to create memory threshold notifier", "err", err)
				} else {
					go notifier.Start(ctx)
					m.thresholdNotifiers = append(m.thresholdNotifiers, notifier)
				}
			}
		}
	}
	// start the eviction manager monitoring
	go func() {
		for {
			evictedPods, err := m.synchronize(ctx, diskInfoProvider, podFunc)
			if evictedPods != nil && err == nil {
				logger.Info("Eviction manager: pods evicted, waiting for pod to be cleaned up", "pods", klog.KObjSlice(evictedPods))
				m.waitForPodsCleanup(logger, podCleanedUpFunc, evictedPods)
			} else {
				if err != nil {
					logger.Error(err, "Eviction manager: failed to synchronize")
				}
				time.Sleep(monitoringInterval)
			}
		}
	}()
}

// IsUnderMemoryPressure returns true if the node is under memory pressure.
func (m *managerImpl) IsUnderMemoryPressure() bool {
	m.RLock()
	defer m.RUnlock()
	return hasNodeCondition(m.nodeConditions, v1.NodeMemoryPressure)
}

// IsUnderDiskPressure returns true if the node is under disk pressure.
func (m *managerImpl) IsUnderDiskPressure() bool {
	m.RLock()
	defer m.RUnlock()
	return hasNodeCondition(m.nodeConditions, v1.NodeDiskPressure)
}

// IsUnderPIDPressure returns true if the node is under PID pressure.
func (m *managerImpl) IsUnderPIDPressure() bool {
	m.RLock()
	defer m.RUnlock()
	return hasNodeCondition(m.nodeConditions, v1.NodePIDPressure)
}

// synchronize is the main control loop that enforces eviction thresholds.
// Returns the pod that was killed, or nil if no pod was killed.
func (m *managerImpl) synchronize(ctx context.Context, diskInfoProvider DiskInfoProvider, podFunc ActivePodsFunc) ([]*v1.Pod, error) {
	logger := klog.FromContext(ctx)
	// if we have nothing to do, just return
	thresholds := m.config.Thresholds
	if len(thresholds) == 0 && !m.localStorageCapacityIsolation {
		return nil, nil
	}

	logger.V(3).Info("Eviction manager: synchronize housekeeping")
	// build the ranking functions (if not yet known)
	// TODO: have a function in cadvisor that lets us know if global housekeeping has completed
	if m.dedicatedImageFs == nil {
		hasImageFs, imageFsErr := diskInfoProvider.HasDedicatedImageFs(ctx)
		if imageFsErr != nil {
			// TODO: This should be refactored to log an error and retry the HasDedicatedImageFs
			// If we have a transient error this will never be retried and we will not set eviction signals
			logger.Error(imageFsErr, "Eviction manager: failed to get HasDedicatedImageFs")
			return nil, fmt.Errorf("eviction manager: failed to get HasDedicatedImageFs: %w", imageFsErr)
		}
		m.dedicatedImageFs = &hasImageFs
		splitContainerImageFs, splitErr := diskInfoProvider.HasDedicatedContainerFs(ctx)
		if splitErr != nil {
			// A common error case is when there is no split filesystem
			// there is an error finding the split filesystem label and we want to ignore these errors
			logger.Error(splitErr, "eviction manager: failed to check if we have separate container filesystem. Ignoring.")
		}

		// If we are a split filesystem but the feature is turned off
		// we should return an error.
		// This is a bad state.
		if !utilfeature.DefaultFeatureGate.Enabled(features.KubeletSeparateDiskGC) && splitContainerImageFs {
			splitDiskError := fmt.Errorf("KubeletSeparateDiskGC is turned off but we still have a split filesystem")
			return nil, splitDiskError
		}
		thresholds, err := UpdateContainerFsThresholds(m.config.Thresholds, hasImageFs, splitContainerImageFs)
		m.config.Thresholds = thresholds
		if err != nil {
			logger.Error(err, "eviction manager: found conflicting containerfs eviction. Ignoring.")
		}
		m.splitContainerImageFs = &splitContainerImageFs
		m.signalToRankFunc = buildSignalToRankFunc(hasImageFs, splitContainerImageFs)
		m.signalToNodeReclaimFuncs = buildSignalToNodeReclaimFuncs(m.imageGC, m.containerGC, hasImageFs, splitContainerImageFs)
	}

	logger.V(3).Info("FileSystem detection", "DedicatedImageFs", m.dedicatedImageFs, "SplitImageFs", m.splitContainerImageFs)
	activePods := podFunc()
	updateStats := true
	summary, err := m.summaryProvider.Get(ctx, updateStats)
	if err != nil {
		logger.Error(err, "Eviction manager: failed to get summary stats")
		return nil, nil
	}

	if m.clock.Since(m.thresholdsLastUpdated) > notifierRefreshInterval {
		m.thresholdsLastUpdated = m.clock.Now()
		for _, notifier := range m.thresholdNotifiers {
			if err := notifier.UpdateThreshold(ctx, summary); err != nil {
				logger.Info("Eviction manager: failed to update notifier", "notifier", notifier.Description(), "err", err)
			}
		}
	}

	// make observations and get a function to derive pod usage stats relative to those observations.
	observations, statsFunc := makeSignalObservations(logger, summary)
	debugLogObservations(logger, "observations", observations)

	// determine the set of thresholds met independent of grace period
	thresholds = thresholdsMet(logger, thresholds, observations, false)
	debugLogThresholdsWithObservation(logger, "thresholds - ignoring grace period", thresholds, observations)

	// determine the set of thresholds previously met that have not yet satisfied the associated min-reclaim
	if len(m.thresholdsMet) > 0 {
		thresholdsNotYetResolved := thresholdsMet(logger, m.thresholdsMet, observations, true)
		thresholds = mergeThresholds(thresholds, thresholdsNotYetResolved)
	}
	debugLogThresholdsWithObservation(logger, "thresholds - reclaim not satisfied", thresholds, observations)

	// track when a threshold was first observed
	now := m.clock.Now()
	thresholdsFirstObservedAt := thresholdsFirstObservedAt(thresholds, m.thresholdsFirstObservedAt, now)

	// the set of node conditions that are triggered by currently observed thresholds
	nodeConditions := nodeConditions(thresholds)
	if len(nodeConditions) > 0 {
		logger.V(3).Info("Eviction manager: node conditions - observed", "nodeCondition", nodeConditions)
	}

	// track when a node condition was last observed
	nodeConditionsLastObservedAt := nodeConditionsLastObservedAt(nodeConditions, m.nodeConditionsLastObservedAt, now)

	// node conditions report true if it has been observed within the transition period window
	nodeConditions = nodeConditionsObservedSince(nodeConditionsLastObservedAt, m.config.PressureTransitionPeriod, now)
	if len(nodeConditions) > 0 {
		logger.V(3).Info("Eviction manager: node conditions - transition period not met", "nodeCondition", nodeConditions)
	}

	// determine the set of thresholds we need to drive eviction behavior (i.e. all grace periods are met)
	thresholds = thresholdsMetGracePeriod(logger, thresholdsFirstObservedAt, now)
	debugLogThresholdsWithObservation(logger, "thresholds - grace periods satisfied", thresholds, observations)

	// update internal state
	m.Lock()
	m.nodeConditions = nodeConditions
	m.thresholdsFirstObservedAt = thresholdsFirstObservedAt
	m.nodeConditionsLastObservedAt = nodeConditionsLastObservedAt
	m.thresholdsMet = thresholds

	// determine the set of thresholds whose stats have been updated since the last sync
	thresholds = thresholdsUpdatedStats(logger, thresholds, observations, m.lastObservations)
	debugLogThresholdsWithObservation(logger, "thresholds - updated stats", thresholds, observations)

	m.lastObservations = observations
	m.Unlock()

	// evict pods if there is a resource usage violation from local volume temporary storage
	// If eviction happens in localStorageEviction function, skip the rest of eviction action
	if m.localStorageCapacityIsolation {
		if evictedPods := m.localStorageEviction(logger, activePods, statsFunc); len(evictedPods) > 0 {
			return evictedPods, nil
		}
	}

	if len(thresholds) == 0 {
		logger.V(3).Info("Eviction manager: no resources are starved")
		return nil, nil
	}

	// rank the thresholds by eviction priority
	sort.Sort(byEvictionPriority(thresholds))
	thresholdToReclaim, resourceToReclaim, foundAny := getReclaimableThreshold(logger, thresholds)
	if !foundAny {
		return nil, nil
	}
	logger.Info("Eviction manager: attempting to reclaim", "resourceName", resourceToReclaim)

	// record an event about the resources we are now attempting to reclaim via eviction
	m.recorder.Eventf(m.nodeRef, v1.EventTypeWarning, "EvictionThresholdMet", "Attempting to reclaim %s", resourceToReclaim)

	// check if there are node-level resources we can reclaim to reduce pressure before evicting end-user pods.
	if m.reclaimNodeLevelResources(ctx, thresholdToReclaim.Signal, resourceToReclaim) {
		logger.Info("Eviction manager: able to reduce resource pressure without evicting pods.", "resourceName", resourceToReclaim)
		return nil, nil
	}

	logger.Info("Eviction manager: must evict pod(s) to reclaim", "resourceName", resourceToReclaim)

	// rank the pods for eviction
	rank, ok := m.signalToRankFunc[thresholdToReclaim.Signal]
	if !ok {
		logger.Error(nil, "Eviction manager: no ranking function for signal", "threshold", thresholdToReclaim.Signal)
		return nil, nil
	}

	// the only candidates viable for eviction are those pods that had anything running.
	if len(activePods) == 0 {
		logger.Error(nil, "Eviction manager: eviction thresholds have been met, but no pods are active to evict")
		return nil, nil
	}

	// rank the running pods for eviction for the specified resource
	rank(activePods, statsFunc)

	logger.Info("Eviction manager: pods ranked for eviction", "pods", klog.KObjSlice(activePods))

	//record age of metrics for met thresholds that we are using for evictions.
	for _, t := range thresholds {
		timeObserved := observations[t.Signal].time
		if !timeObserved.IsZero() {
			metrics.EvictionStatsAge.WithLabelValues(string(t.Signal)).Observe(metrics.SinceInSeconds(timeObserved.Time))
		}
	}

	// we kill at most a single pod during each eviction interval
	for i := range activePods {
		pod := activePods[i]
		gracePeriodOverride := int64(immediateEvictionGracePeriodSeconds)
		if !isHardEvictionThreshold(thresholdToReclaim) {
			gracePeriodOverride = m.config.MaxPodGracePeriodSeconds
			if pod.Spec.TerminationGracePeriodSeconds != nil {
				gracePeriodOverride = min(m.config.MaxPodGracePeriodSeconds, *pod.Spec.TerminationGracePeriodSeconds)
			}
		}

		message, annotations := evictionMessage(resourceToReclaim, pod, statsFunc, thresholds, observations)
		condition := &v1.PodCondition{
			Type:               v1.DisruptionTarget,
			ObservedGeneration: pod.Generation,
			Status:             v1.ConditionTrue,
			Reason:             v1.PodReasonTerminationByKubelet,
			Message:            message,
		}
		if m.evictPod(logger, pod, gracePeriodOverride, message, annotations, condition) {
			metrics.Evictions.WithLabelValues(string(thresholdToReclaim.Signal)).Inc()
			return []*v1.Pod{pod}, nil
		}
	}
	logger.Info("Eviction manager: unable to evict any pods from the node")
	return nil, nil
}

func (m *managerImpl) waitForPodsCleanup(logger klog.Logger, podCleanedUpFunc PodCleanedUpFunc, pods []*v1.Pod) {
	timeout := m.clock.NewTimer(podCleanupTimeout)
	defer timeout.Stop()
	ticker := m.clock.NewTicker(podCleanupPollFreq)
	defer ticker.Stop()
	for {
		select {
		case <-timeout.C():
			logger.Info("Eviction manager: timed out waiting for pods to be cleaned up", "pods", klog.KObjSlice(pods))
			return
		case <-ticker.C():
			for i, pod := range pods {
				if !podCleanedUpFunc(pod) {
					break
				}
				if i == len(pods)-1 {
					logger.Info("Eviction manager: pods successfully cleaned up", "pods", klog.KObjSlice(pods))
					return
				}
			}
		}
	}
}

// reclaimNodeLevelResources attempts to reclaim node level resources.  returns true if thresholds were satisfied and no pod eviction is required.
func (m *managerImpl) reclaimNodeLevelResources(ctx context.Context, signalToReclaim evictionapi.Signal, resourceToReclaim v1.ResourceName) bool {
	logger := klog.FromContext(ctx)
	nodeReclaimFuncs := m.signalToNodeReclaimFuncs[signalToReclaim]
	for _, nodeReclaimFunc := range nodeReclaimFuncs {
		// attempt to reclaim the pressured resource.
		if err := nodeReclaimFunc(ctx); err != nil {
			logger.Info("Eviction manager: unexpected error when attempting to reduce resource pressure", "resourceName", resourceToReclaim, "err", err)
		}

	}
	if len(nodeReclaimFuncs) > 0 {
		summary, err := m.summaryProvider.Get(ctx, true)
		if err != nil {
			logger.Error(err, "Eviction manager: failed to get summary stats after resource reclaim")
			return false
		}

		// make observations and get a function to derive pod usage stats relative to those observations.
		observations, _ := makeSignalObservations(logger, summary)
		debugLogObservations(logger, "observations after resource reclaim", observations)

		// evaluate all thresholds independently of their grace period to see if with
		// the new observations, we think we have met min reclaim goals
		thresholds := thresholdsMet(logger, m.config.Thresholds, observations, true)
		debugLogThresholdsWithObservation(logger, "thresholds after resource reclaim - ignoring grace period", thresholds, observations)

		if len(thresholds) == 0 {
			return true
		}
	}
	return false
}

// localStorageEviction checks the EmptyDir volume usage for each pod and determine whether it exceeds the specified limit and needs
// to be evicted. It also checks every container in the pod, if the container overlay usage exceeds the limit, the pod will be evicted too.
func (m *managerImpl) localStorageEviction(logger klog.Logger, pods []*v1.Pod, statsFunc statsFunc) []*v1.Pod {
	evicted := []*v1.Pod{}
	for _, pod := range pods {
		podStats, ok := statsFunc(pod)
		if !ok {
			continue
		}

		if m.emptyDirLimitEviction(logger, podStats, pod) {
			evicted = append(evicted, pod)
			continue
		}

		if m.podEphemeralStorageLimitEviction(logger, podStats, pod) {
			evicted = append(evicted, pod)
			continue
		}

		if m.containerEphemeralStorageLimitEviction(logger, podStats, pod) {
			evicted = append(evicted, pod)
		}
	}

	return evicted
}

func (m *managerImpl) emptyDirLimitEviction(logger klog.Logger, podStats statsapi.PodStats, pod *v1.Pod) bool {
	podVolumeUsed := make(map[string]*resource.Quantity)
	for _, volume := range podStats.VolumeStats {
		podVolumeUsed[volume.Name] = resource.NewQuantity(int64(*volume.UsedBytes), resource.BinarySI)
	}
	for i := range pod.Spec.Volumes {
		source := &pod.Spec.Volumes[i].VolumeSource
		if source.EmptyDir != nil {
			size := source.EmptyDir.SizeLimit
			used := podVolumeUsed[pod.Spec.Volumes[i].Name]
			if used != nil && size != nil && size.Sign() == 1 && used.Cmp(*size) > 0 {
				// the emptyDir usage exceeds the size limit, evict the pod
				if m.evictPod(logger, pod, immediateEvictionGracePeriodSeconds, fmt.Sprintf(emptyDirMessageFmt, pod.Spec.Volumes[i].Name, size.String()), nil, nil) {
					metrics.Evictions.WithLabelValues(signalEmptyDirFsLimit).Inc()
					return true
				}
				return false
			}
		}
	}

	return false
}

func (m *managerImpl) podEphemeralStorageLimitEviction(logger klog.Logger, podStats statsapi.PodStats, pod *v1.Pod) bool {
	podLimits := resourcehelper.PodLimits(pod, resourcehelper.PodResourcesOptions{})
	_, found := podLimits[v1.ResourceEphemeralStorage]
	if !found {
		return false
	}

	// pod stats api summarizes ephemeral storage usage (container, emptyDir, host[etc-hosts, logs])
	podEphemeralStorageTotalUsage := &resource.Quantity{}
	if podStats.EphemeralStorage != nil && podStats.EphemeralStorage.UsedBytes != nil {
		podEphemeralStorageTotalUsage = resource.NewQuantity(int64(*podStats.EphemeralStorage.UsedBytes), resource.BinarySI)
	}
	podEphemeralStorageLimit := podLimits[v1.ResourceEphemeralStorage]
	if podEphemeralStorageTotalUsage.Cmp(podEphemeralStorageLimit) > 0 {
		// the total usage of pod exceeds the total size limit of containers, evict the pod
		message := fmt.Sprintf(podEphemeralStorageMessageFmt, podEphemeralStorageLimit.String())
		if m.evictPod(logger, pod, immediateEvictionGracePeriodSeconds, message, nil, nil) {
			metrics.Evictions.WithLabelValues(signalEphemeralPodFsLimit).Inc()
			return true
		}
		return false
	}
	return false
}

func (m *managerImpl) containerEphemeralStorageLimitEviction(logger klog.Logger, podStats statsapi.PodStats, pod *v1.Pod) bool {
	thresholdsMap := make(map[string]*resource.Quantity)
	for _, container := range pod.Spec.Containers {
		ephemeralLimit := container.Resources.Limits.StorageEphemeral()
		if ephemeralLimit != nil && ephemeralLimit.Value() != 0 {
			thresholdsMap[container.Name] = ephemeralLimit
		}
	}

	for _, containerStat := range podStats.Containers {
		containerUsed := diskUsage(containerStat.Logs)
		if !*m.dedicatedImageFs {
			containerUsed.Add(*diskUsage(containerStat.Rootfs))
		}

		if ephemeralStorageThreshold, ok := thresholdsMap[containerStat.Name]; ok {
			if ephemeralStorageThreshold.Cmp(*containerUsed) < 0 {
				if m.evictPod(logger, pod, immediateEvictionGracePeriodSeconds, fmt.Sprintf(containerEphemeralStorageMessageFmt, containerStat.Name, ephemeralStorageThreshold.String()), nil, nil) {
					metrics.Evictions.WithLabelValues(signalEphemeralContainerFsLimit).Inc()
					return true
				}
				return false
			}
		}
	}
	return false
}

func (m *managerImpl) evictPod(logger klog.Logger, pod *v1.Pod, gracePeriodOverride int64, evictMsg string, annotations map[string]string, condition *v1.PodCondition) bool {
	// If the pod is marked as critical and static, and support for critical pod annotations is enabled,
	// do not evict such pods. Static pods are not re-admitted after evictions.
	// https://github.com/kubernetes/kubernetes/issues/40573 has more details.
	if kubelettypes.IsCriticalPod(pod) {
		logger.Error(nil, "Eviction manager: cannot evict a critical pod", "pod", klog.KObj(pod))
		return false
	}
	// record that we are evicting the pod
	m.recorder.AnnotatedEventf(pod, annotations, v1.EventTypeWarning, Reason, evictMsg)
	// this is a blocking call and should only return when the pod and its containers are killed.
	logger.V(3).Info("Evicting pod", "pod", klog.KObj(pod), "podUID", pod.UID, "message", evictMsg)
	err := m.killPodFunc(pod, true, &gracePeriodOverride, func(status *v1.PodStatus) {
		status.Phase = v1.PodFailed
		status.Reason = Reason
		status.Message = evictMsg
		if condition != nil {
			condition.ObservedGeneration = podutil.CalculatePodConditionObservedGeneration(status, pod.Generation, v1.DisruptionTarget)
			podutil.UpdatePodCondition(status, condition)
		}
	})
	if err != nil {
		logger.Error(err, "Eviction manager: pod failed to evict", "pod", klog.KObj(pod))
	} else {
		logger.Info("Eviction manager: pod is evicted successfully", "pod", klog.KObj(pod))
	}
	return true
}
