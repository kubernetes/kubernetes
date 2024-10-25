/*
Copyright 2017 The Kubernetes Authors.

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

    podutil "k8s.io/kubernetes/pkg/api/v1/pod"
    resourcehelper "k8s.io/kubernetes/pkg/api/v1/resource"
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
    // signalEphemeralContainerFsLimit is the storage limit for a container's ephemeral storage.
    signalEphemeralContainerFsLimit string = "ephemeralcontainerfs.limit"
    // signalEphemeralPodFsLimit is the storage limit for a pod's ephemeral storage.
    signalEphemeralPodFsLimit string = "ephemeralpodfs.limit"
    // signalEmptyDirFsLimit is the storage limit for emptyDir volumes.
    signalEmptyDirFsLimit string = "emptydirfs.limit"
    // immediateEvictionGracePeriodSeconds is the shutdown grace period during quick evictions.
    immediateEvictionGracePeriodSeconds = 1
)

// managerImpl is the eviction manager that implements the Manager interface.
type managerImpl struct {
    clock                        clock.WithTicker
    config                       Config
    killPodFunc                  KillPodFunc
    imageGC                      ImageGC
    containerGC                  ContainerGC
    sync.RWMutex
    nodeConditions               []v1.NodeConditionType
    nodeConditionsLastObservedAt nodeConditionsObservedAt
    nodeRef                      *v1.ObjectReference
    recorder                     record.EventRecorder
    summaryProvider              stats.SummaryProvider
    thresholdsFirstObservedAt    thresholdsObservedAt
    thresholdsMet                []evictionapi.Threshold
    signalToRankFunc             map[evictionapi.Signal]rankFunc
    signalToNodeReclaimFuncs     map[evictionapi.Signal]nodeReclaimFuncs
    lastObservations             signalObservations
    dedicatedImageFs             *bool
    splitContainerImageFs        *bool
    thresholdNotifiers           []ThresholdNotifier
    thresholdsLastUpdated        time.Time
    localStorageCapacityIsolation bool
}

// verify that managerImpl implements the Manager interface
var _ Manager = &managerImpl{}

// NewManager returns a configured Manager and associated admission handler.
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

// Admit denies pod admission when conditions make it unsafe to admit.
func (m *managerImpl) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
    m.RLock()
    defer m.RUnlock()
    if len(m.nodeConditions) == 0 {
        return lifecycle.PodAdmitResult{Admit: true}
    }
    if kubelettypes.IsCriticalPod(attrs.Pod) {
        return lifecycle.PodAdmitResult{Admit: true}
    }

    nodeOnlyHasMemoryPressureCondition := hasNodeCondition(m.nodeConditions, v1.NodeMemoryPressure) && len(m.nodeConditions) == 1
    if nodeOnlyHasMemoryPressureCondition {
        notBestEffort := v1.PodQOSBestEffort != v1qos.GetPodQOS(attrs.Pod)
        if notBestEffort {
            return lifecycle.PodAdmitResult{Admit: true}
        }

        if corev1helpers.TolerationsTolerateTaint(attrs.Pod.Spec.Tolerations, &v1.Taint{
            Key:    v1.TaintNodeMemoryPressure,
            Effect: v1.TaintEffectNoSchedule,
        }) {
            return lifecycle.PodAdmitResult{Admit: true}
        }
    }

    return lifecycle.PodAdmitResult{
        Admit:   false,
        Reason:  Reason,
        Message: fmt.Sprintf(nodeConditionMessageFmt, m.nodeConditions),
    }
}

// Start begins the main loop, observing and responding to resource pressure.
func (m *managerImpl) Start(diskInfoProvider DiskInfoProvider, podFunc ActivePodsFunc, podCleanedUpFunc PodCleanedUpFunc, monitoringInterval time.Duration) {
    thresholdHandler := func(message string) {
        klog.InfoS(message)
        m.synchronize(diskInfoProvider, podFunc)
    }
    klog.InfoS("Eviction manager: starting control loop")
    if m.config.KernelMemcgNotification || runtime.GOOS == "windows" {
        for _, threshold := range m.config.Thresholds {
            if threshold.Signal == evictionapi.SignalMemoryAvailable || threshold.Signal == evictionapi.SignalAllocatableMemoryAvailable {
                notifier, err := NewMemoryThresholdNotifier(threshold, m.config.PodCgroupRoot, &CgroupNotifierFactory{}, thresholdHandler)
                if err != nil {
                    klog.InfoS("Eviction manager: failed to create memory threshold notifier", "err", err)
                } else {
                    go notifier.Start()
                    m.thresholdNotifiers = append(m.thresholdNotifiers, notifier)
                }
            }
        }
    }
    go func() {
        for {
            evictedPods, err := m.synchronize(diskInfoProvider, podFunc)
            if evictedPods != nil && err == nil {
                klog.InfoS("Eviction manager: pods evicted, waiting for pod to be cleaned up", "pods", klog.KObjSlice(evictedPods))
                m.waitForPodsCleanup(podCleanedUpFunc, evictedPods)
            } else {
                if err != nil {
                    klog.ErrorS(err, "Eviction manager: failed to synchronize")
                }
                time.Sleep(monitoringInterval)
            }
        }
    }()
}

// IsUnderMemoryPressure returns true if node is under memory pressure.
func (m *managerImpl) IsUnderMemoryPressure() bool {
    m.RLock()
    defer m.RUnlock()
    return hasNodeCondition(m.nodeConditions, v1.NodeMemoryPressure)
}

// IsUnderDiskPressure returns true if node is under disk pressure.
func (m *managerImpl) IsUnderDiskPressure() bool {
    m.RLock()
    defer m.RUnlock()
    return hasNodeCondition(m.nodeConditions, v1.NodeDiskPressure)
}

// IsUnderPIDPressure returns true if node is under PID pressure.
func (m *managerImpl) IsUnderPIDPressure() bool {
    m.RLock()
    defer m.RUnlock()
    return hasNodeCondition(m.nodeConditions, v1.NodePIDPressure)
}

// synchronize enforces eviction thresholds by checking resource levels.
func (m *managerImpl) synchronize(diskInfoProvider DiskInfoProvider, podFunc ActivePodsFunc) ([]*v1.Pod, error) {
    ctx := context.Background()
    thresholds := m.config.Thresholds
    if len(thresholds) == 0 && !m.localStorageCapacityIsolation {
        return nil, nil
    }

    klog.V(3).InfoS("Eviction manager: synchronize housekeeping")
    if m.dedicatedImageFs == nil {
        hasImageFs, imageFsErr := diskInfoProvider.HasDedicatedImageFs(ctx)
        if imageFsErr != nil {
            klog.ErrorS(imageFsErr, "Eviction manager: failed to get HasDedicatedImageFs")
            return nil, fmt.Errorf("eviction manager: failed to get HasDedicatedImageFs: %w", imageFsErr)
        }
        m.dedicatedImageFs = &hasImageFs
        splitContainerImageFs, splitErr := diskInfoProvider.HasDedicatedContainerFs(ctx)
        if splitErr != nil {
            klog.ErrorS(splitErr, "Eviction manager: failed to get HasDedicatedContainerFs")
            return nil, fmt.Errorf("eviction manager: failed to get HasDedicatedContainerFs: %w", splitErr)
        }
        if !utilfeature.DefaultFeatureGate.Enabled(features.KubeletSeparateDiskGC) && splitContainerImageFs {
            splitDiskError := fmt.Errorf("KubeletSeparateDiskGC is turned off but we still have a split filesystem")
            return nil, splitDiskError
        }
        thresholds, err := UpdateContainerFsThresholds(m.config.Thresholds, hasImageFs, splitContainerImageFs)
        m.config.Thresholds = thresholds
        if err != nil {
            klog.ErrorS(err, "eviction manager: found conflicting containerfs eviction. Ignoring.")
        }
        m.splitContainerImageFs = &splitContainerImageFs
        m.signalToRankFunc = buildSignalToRankFunc(hasImageFs, splitContainerImageFs)
        m.signalToNodeReclaimFuncs = buildSignalToNodeReclaimFuncs(m.imageGC, m.containerGC, hasImageFs, splitContainerImageFs)
    }

    klog.V(3).InfoS("FileSystem detection", "DedicatedImageFs", m.dedicatedImageFs, "SplitImageFs", m.splitContainerImageFs)
    activePods := podFunc()
    updateStats := true
    summary, err := m.summaryProvider.Get(ctx, updateStats)
    if err != nil {
        klog.ErrorS(err, "Eviction manager: failed to get summary stats")
        return nil, nil
    }

    if m.clock.Since(m.thresholdsLastUpdated) > notifierRefreshInterval {
        m.thresholdsLastUpdated = m.clock.Now()
        for _, notifier := range m.thresholdNotifiers {
            if err := notifier.UpdateThreshold(summary); err != nil {
                klog.InfoS("Eviction manager: failed to update notifier", "notifier", notifier.Description(), "err", err)
            }
        }
    }

    // make observations and get a function to derive pod usage stats relative to those observations.
    observations, statsFunc := makeSignalObservations(summary)
    debugLogObservations("observations", observations)

    // determine the set of thresholds met independent of grace period
    thresholds = thresholdsMet(thresholds, observations, false)
    debugLogThresholdsWithObservation("thresholds - ignoring grace period", thresholds, observations)

    // determine the set of thresholds previously met that have not yet satisfied the associated min-reclaim
    if len(m.thresholdsMet) > 0 {
        thresholdsNotYetResolved := thresholdsMet(m.thresholdsMet, observations, true)
        thresholds = mergeThresholds(thresholds, thresholdsNotYetResolved)
    }
    debugLogThresholdsWithObservation("thresholds - reclaim not satisfied", thresholds, observations)

    // track when a threshold was first observed
    now := m.clock.Now()
    thresholdsFirstObservedAt := thresholdsFirstObservedAt(thresholds, m.thresholdsFirstObservedAt, now)

    // the set of node conditions that are triggered by currently observed thresholds
    nodeConditions := nodeConditions(thresholds)
    if len(nodeConditions) > 0 {
        klog.V(3).InfoS("Eviction manager: node conditions - observed", "nodeCondition", nodeConditions)
    }

    // track when a node condition was last observed
    nodeConditionsLastObservedAt := nodeConditionsLastObservedAt(nodeConditions, m.nodeConditionsLastObservedAt, now)

    // node conditions report true if it has been observed within the transition period window
    nodeConditions = nodeConditionsObservedSince(nodeConditionsLastObservedAt, m.config.PressureTransitionPeriod, now)
    if len(nodeConditions) > 0 {
        klog.V(3).InfoS("Eviction manager: node conditions - transition period not met", "nodeCondition", nodeConditions)
    }

    // determine the set of thresholds we need to drive eviction behavior (i.e. all grace periods are met)
    thresholds = thresholdsMetGracePeriod(thresholdsFirstObservedAt, now)
    debugLogThresholdsWithObservation("thresholds - grace periods satisfied", thresholds, observations)

    // update internal state
    m.Lock()
    m.nodeConditions = nodeConditions
    m.thresholdsFirstObservedAt = thresholdsFirstObservedAt
    m.nodeConditionsLastObservedAt = nodeConditionsLastObservedAt
    m.thresholdsMet = thresholds

    // determine the set of thresholds whose stats have been updated since the last sync
    thresholds = thresholdsUpdatedStats(thresholds, observations, m.lastObservations)
    debugLogThresholdsWithObservation("thresholds - updated stats", thresholds, observations)

    m.lastObservations = observations
    m.Unlock()

    // evict pods if there is a resource usage violation from local volume temporary storage
    // If eviction happens in localStorageEviction function, skip the rest of eviction action
    if m.localStorageCapacityIsolation {
        if evictedPods := m.localStorageEviction(activePods, statsFunc); len(evictedPods) > 0 {
            return evictedPods, nil
        }
    }

    if len(thresholds) == 0 {
        klog.V(3).InfoS("Eviction manager: no resources are starved")
        return nil, nil
    }

    // rank the thresholds by eviction priority
    sort.Sort(byEvictionPriority(thresholds))
    thresholdToReclaim, resourceToReclaim, foundAny := getReclaimableThreshold(thresholds)
    if !foundAny {
        return nil, nil
    }
    klog.InfoS("Eviction manager: attempting to reclaim", "resourceName", resourceToReclaim)

    // record an event about the resources we are now attempting to reclaim via eviction
    m.recorder.Eventf(m.nodeRef, v1.EventTypeWarning, "EvictionThresholdMet", "Attempting to reclaim %s", resourceToReclaim)

    // check if there are node-level resources we can reclaim to reduce pressure before evicting end-user pods.
    if m.reclaimNodeLevelResources(ctx, thresholdToReclaim.Signal, resourceToReclaim) {
        klog.InfoS("Eviction manager: able to reduce resource pressure without evicting pods.", "resourceName", resourceToReclaim)
        return nil, nil
    }

    klog.InfoS("Eviction manager: must evict pod(s) to reclaim", "resourceName", resourceToReclaim)

    // rank the pods for eviction
    rank, ok := m.signalToRankFunc[thresholdToReclaim.Signal]
    if !ok {
        klog.ErrorS(nil, "Eviction manager: no ranking function for signal", "threshold", thresholdToReclaim.Signal)
        return nil, nil
    }

    // the only candidates viable for eviction are those pods that had anything running.
    if len(activePods) == 0 {
        klog.ErrorS(nil, "Eviction manager: eviction thresholds have been met, but no pods are active to evict")
        return nil, nil
    }

    // rank the running pods for eviction for the specified resource
    rank(activePods, statsFunc)

    klog.InfoS("Eviction manager: pods ranked for eviction", "pods", klog.KObjSlice(activePods))

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
            if pod.Spec.TerminationGracePeriodSeconds != nil && !utilfeature.DefaultFeatureGate.Enabled(features.AllowOverwriteTerminationGracePeriodSeconds) {
                gracePeriodOverride = min(m.config.MaxPodGracePeriodSeconds, *pod.Spec.TerminationGracePeriodSeconds)
            }
        }

        message, annotations := evictionMessage(resourceToReclaim, pod, statsFunc, thresholds, observations)
        condition := &v1.PodCondition{
            Type:    v1.DisruptionTarget,
            Status:  v1.ConditionTrue,
            Reason:  v1.PodReasonTerminationByKubelet,
            Message: message,
        }
        if m.evictPod(pod, gracePeriodOverride, message, annotations, condition) {
            metrics.Evictions.WithLabelValues(string(thresholdToReclaim.Signal)).Inc()
            return []*v1.Pod{pod}, nil
        }
    }
    klog.InfoS("Eviction manager: unable to evict any pods from the node")
    return nil, nil
}
