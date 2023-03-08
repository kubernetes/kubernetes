/*
Copyright 2021 The Kubernetes Authors.

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

package nodeshutdown

import (
	"sort"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	"k8s.io/utils/clock"
)

// Manager interface provides methods for Kubelet to manage node shutdown.
type Manager interface {
	lifecycle.PodAdmitHandler

	Start() error
	ShutdownStatus() error
}

// Config represents Manager configuration
type Config struct {
	Logger                           klog.Logger
	ProbeManager                     prober.Manager
	Recorder                         record.EventRecorder
	NodeRef                          *v1.ObjectReference
	GetPodsFunc                      eviction.ActivePodsFunc
	KillPodFunc                      eviction.KillPodFunc
	SyncNodeStatusFunc               func()
	ShutdownGracePeriodRequested     time.Duration
	ShutdownGracePeriodCriticalPods  time.Duration
	ShutdownGracePeriodByPodPriority []kubeletconfig.ShutdownGracePeriodByPodPriority
	StateDirectory                   string
	Clock                            clock.Clock
}

// managerStub is a fake node shutdown managerImpl .
type managerStub struct{}

// Admit returns a fake Pod admission which always returns true
func (managerStub) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	return lifecycle.PodAdmitResult{Admit: true}
}

// Start is a no-op always returning nil for non linux platforms.
func (managerStub) Start() error {
	return nil
}

// ShutdownStatus is a no-op always returning nil for non linux platforms.
func (managerStub) ShutdownStatus() error {
	return nil
}

const (
	nodeShutdownReason  = "Terminated"
	nodeShutdownMessage = "Pod was terminated in response to imminent node shutdown."
)

type podShutdownGroup struct {
	kubeletconfig.ShutdownGracePeriodByPodPriority
	Pods []*v1.Pod
}

// podHandler is responsible for killing active pods by priority.
type podHandler struct {
	logger                           klog.Logger
	killPodFunc                      eviction.KillPodFunc
	shutdownGracePeriodByPodPriority []kubeletconfig.ShutdownGracePeriodByPodPriority
	clock                            clock.Clock
}

func newPodHandler(conf *Config) podHandler {
	shutdownGracePeriodByPodPriority := conf.ShutdownGracePeriodByPodPriority

	// Migration from the original configuration
	if !utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdownBasedOnPodPriority) ||
		len(shutdownGracePeriodByPodPriority) == 0 {
		shutdownGracePeriodByPodPriority = migrateConfig(conf.ShutdownGracePeriodRequested, conf.ShutdownGracePeriodCriticalPods)
	}

	// Sort by priority from low to high
	sort.Slice(shutdownGracePeriodByPodPriority, func(i, j int) bool {
		return shutdownGracePeriodByPodPriority[i].Priority < shutdownGracePeriodByPodPriority[j].Priority
	})

	if conf.Clock == nil {
		conf.Clock = clock.RealClock{}
	}

	return podHandler{
		logger:                           conf.Logger,
		killPodFunc:                      conf.KillPodFunc,
		shutdownGracePeriodByPodPriority: shutdownGracePeriodByPodPriority,
		clock:                            conf.Clock,
	}
}

// killPods terminates pods by priority.
func (p *podHandler) killPods(activePods []*v1.Pod) {
	groups := groupByPriority(p.shutdownGracePeriodByPodPriority, activePods)
	for _, group := range groups {
		// If there are no pods in a particular range,
		// then do not wait for pods in that priority range.
		if len(group.Pods) == 0 {
			continue
		}

		var wg sync.WaitGroup
		wg.Add(len(group.Pods))
		for _, pod := range group.Pods {
			go func(pod *v1.Pod, group podShutdownGroup) {
				defer wg.Done()

				gracePeriodOverride := group.ShutdownGracePeriodSeconds

				// If the pod's spec specifies a termination gracePeriod which is less than the gracePeriodOverride calculated, use the pod spec termination gracePeriod.
				if pod.Spec.TerminationGracePeriodSeconds != nil && *pod.Spec.TerminationGracePeriodSeconds <= gracePeriodOverride {
					gracePeriodOverride = *pod.Spec.TerminationGracePeriodSeconds
				}

				p.logger.V(1).Info("Shutdown manager killing pod with gracePeriod", "pod", klog.KObj(pod), "gracePeriod", gracePeriodOverride)

				if err := p.killPodFunc(pod, false, &gracePeriodOverride, func(status *v1.PodStatus) {
					// set the pod status to failed (unless it was already in a successful terminal phase)
					if status.Phase != v1.PodSucceeded {
						status.Phase = v1.PodFailed
					}
					status.Message = nodeShutdownMessage
					status.Reason = nodeShutdownReason
					if utilfeature.DefaultFeatureGate.Enabled(features.PodDisruptionConditions) {
						podutil.UpdatePodCondition(status, &v1.PodCondition{
							Type:    v1.DisruptionTarget,
							Status:  v1.ConditionTrue,
							Reason:  v1.PodReasonTerminationByKubelet,
							Message: nodeShutdownMessage,
						})
					}
				}); err != nil {
					p.logger.V(1).Info("Shutdown manager failed killing pod", "pod", klog.KObj(pod), "err", err)
				} else {
					p.logger.V(1).Info("Shutdown manager finished killing pod", "pod", klog.KObj(pod))
				}
			}(pod, group)
		}

		var (
			doneCh = make(chan struct{})
			timer  = p.clock.NewTimer(time.Duration(group.ShutdownGracePeriodSeconds) * time.Second)
		)
		go func() {
			defer close(doneCh)
			wg.Wait()
		}()

		select {
		case <-doneCh:
			timer.Stop()
		case <-timer.C():
			p.logger.V(1).Info("Shutdown manager pod killing time out", "gracePeriod", group.ShutdownGracePeriodSeconds, "priority", group.Priority)
		}
	}
}

func (p *podHandler) periodRequested() time.Duration {
	var sum int64
	for _, period := range p.shutdownGracePeriodByPodPriority {
		sum += period.ShutdownGracePeriodSeconds
	}
	return time.Duration(sum) * time.Second
}

func groupByPriority(shutdownGracePeriodByPodPriority []kubeletconfig.ShutdownGracePeriodByPodPriority, pods []*v1.Pod) []podShutdownGroup {
	groups := make([]podShutdownGroup, 0, len(shutdownGracePeriodByPodPriority))
	for _, period := range shutdownGracePeriodByPodPriority {
		groups = append(groups, podShutdownGroup{
			ShutdownGracePeriodByPodPriority: period,
		})
	}

	for _, pod := range pods {
		var priority int32
		if pod.Spec.Priority != nil {
			priority = *pod.Spec.Priority
		}

		// Find the group index according to the priority.
		index := sort.Search(len(groups), func(i int) bool {
			return groups[i].Priority >= priority
		})

		// 1. Those higher than the highest priority default to the highest priority
		// 2. Those lower than the lowest priority default to the lowest priority
		// 3. Those boundary priority default to the lower priority
		// if priority of pod is:
		//   groups[index-1].Priority <= pod priority < groups[index].Priority
		// in which case we want to pick lower one (i.e index-1)
		if index == len(groups) {
			index = len(groups) - 1
		} else if index < 0 {
			index = 0
		} else if index > 0 && groups[index].Priority > priority {
			index--
		}

		groups[index].Pods = append(groups[index].Pods, pod)
	}
	return groups
}

func migrateConfig(shutdownGracePeriodRequested, shutdownGracePeriodCriticalPods time.Duration) []kubeletconfig.ShutdownGracePeriodByPodPriority {
	if shutdownGracePeriodRequested == 0 {
		return nil
	}
	defaultPriority := shutdownGracePeriodRequested - shutdownGracePeriodCriticalPods
	if defaultPriority < 0 {
		return nil
	}
	criticalPriority := shutdownGracePeriodRequested - defaultPriority
	if criticalPriority < 0 {
		return nil
	}
	return []kubeletconfig.ShutdownGracePeriodByPodPriority{
		{
			Priority:                   scheduling.DefaultPriorityWhenNoDefaultClassExists,
			ShutdownGracePeriodSeconds: int64(defaultPriority / time.Second),
		},
		{
			Priority:                   scheduling.SystemCriticalPriority,
			ShutdownGracePeriodSeconds: int64(criticalPriority / time.Second),
		},
	}
}
