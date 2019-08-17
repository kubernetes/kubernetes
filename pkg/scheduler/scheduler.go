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

package scheduler

import (
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	policyinformers "k8s.io/client-go/informers/policy/v1beta1"
	storageinformersv1 "k8s.io/client-go/informers/storage/v1"
	storageinformersv1beta1 "k8s.io/client-go/informers/storage/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/events"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/pkg/scheduler/api/latest"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/core"
	"k8s.io/kubernetes/pkg/scheduler/factory"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

const (
	// BindTimeoutSeconds defines the default bind timeout
	BindTimeoutSeconds = 100
	// SchedulerError is the reason recorded for events when an error occurs during scheduling a pod.
	SchedulerError = "SchedulerError"
)

// Scheduler watches for new unscheduled pods. It attempts to find
// nodes that they fit on and writes bindings back to the api server.
type Scheduler struct {
	// It is expected that changes made via SchedulerCache will be observed
	// by NodeLister and Algorithm.
	SchedulerCache internalcache.Cache

	Algorithm core.ScheduleAlgorithm
	GetBinder func(pod *v1.Pod) factory.Binder
	// PodConditionUpdater is used only in case of scheduling errors. If we succeed
	// with scheduling, PodScheduled condition will be updated in apiserver in /bind
	// handler so that binding and setting PodCondition it is atomic.
	PodConditionUpdater factory.PodConditionUpdater
	// PodPreemptor is used to evict pods and update 'NominatedNode' field of
	// the preemptor pod.
	PodPreemptor factory.PodPreemptor
	// Framework runs scheduler plugins at configured extension points.
	Framework framework.Framework

	// NextPod should be a function that blocks until the next pod
	// is available. We don't use a channel for this, because scheduling
	// a pod may take some amount of time and we don't want pods to get
	// stale while they sit in a channel.
	NextPod func() *v1.Pod

	// WaitForCacheSync waits for scheduler cache to populate.
	// It returns true if it was successful, false if the controller should shutdown.
	WaitForCacheSync func() bool

	// Error is called if there is an error. It is passed the pod in
	// question, and the error
	Error func(*v1.Pod, error)

	// Recorder is the EventRecorder to use
	Recorder events.EventRecorder

	// Close this to shut down the scheduler.
	StopEverything <-chan struct{}

	// VolumeBinder handles PVC/PV binding for the pod.
	VolumeBinder *volumebinder.VolumeBinder

	// Disable pod preemption or not.
	DisablePreemption bool

	// SchedulingQueue holds pods to be scheduled
	SchedulingQueue internalqueue.SchedulingQueue
}

// Cache returns the cache in scheduler for test to check the data in scheduler.
func (sched *Scheduler) Cache() internalcache.Cache {
	return sched.SchedulerCache
}

type schedulerOptions struct {
	schedulerName                  string
	hardPodAffinitySymmetricWeight int32
	disablePreemption              bool
	percentageOfNodesToScore       int32
	bindTimeoutSeconds             int64
}

// Option configures a Scheduler
type Option func(*schedulerOptions)

// WithName sets schedulerName for Scheduler, the default schedulerName is default-scheduler
func WithName(schedulerName string) Option {
	return func(o *schedulerOptions) {
		o.schedulerName = schedulerName
	}
}

// WithHardPodAffinitySymmetricWeight sets hardPodAffinitySymmetricWeight for Scheduler, the default value is 1
func WithHardPodAffinitySymmetricWeight(hardPodAffinitySymmetricWeight int32) Option {
	return func(o *schedulerOptions) {
		o.hardPodAffinitySymmetricWeight = hardPodAffinitySymmetricWeight
	}
}

// WithPreemptionDisabled sets disablePreemption for Scheduler, the default value is false
func WithPreemptionDisabled(disablePreemption bool) Option {
	return func(o *schedulerOptions) {
		o.disablePreemption = disablePreemption
	}
}

// WithPercentageOfNodesToScore sets percentageOfNodesToScore for Scheduler, the default value is 50
func WithPercentageOfNodesToScore(percentageOfNodesToScore int32) Option {
	return func(o *schedulerOptions) {
		o.percentageOfNodesToScore = percentageOfNodesToScore
	}
}

// WithBindTimeoutSeconds sets bindTimeoutSeconds for Scheduler, the default value is 100
func WithBindTimeoutSeconds(bindTimeoutSeconds int64) Option {
	return func(o *schedulerOptions) {
		o.bindTimeoutSeconds = bindTimeoutSeconds
	}
}

var defaultSchedulerOptions = schedulerOptions{
	schedulerName:                  v1.DefaultSchedulerName,
	hardPodAffinitySymmetricWeight: v1.DefaultHardPodAffinitySymmetricWeight,
	disablePreemption:              false,
	percentageOfNodesToScore:       schedulerapi.DefaultPercentageOfNodesToScore,
	bindTimeoutSeconds:             BindTimeoutSeconds,
}

// New returns a Scheduler
func New(client clientset.Interface,
	nodeInformer coreinformers.NodeInformer,
	podInformer coreinformers.PodInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	replicationControllerInformer coreinformers.ReplicationControllerInformer,
	replicaSetInformer appsinformers.ReplicaSetInformer,
	statefulSetInformer appsinformers.StatefulSetInformer,
	serviceInformer coreinformers.ServiceInformer,
	pdbInformer policyinformers.PodDisruptionBudgetInformer,
	storageClassInformer storageinformersv1.StorageClassInformer,
	csiNodeInformer storageinformersv1beta1.CSINodeInformer,
	recorder events.EventRecorder,
	schedulerAlgorithmSource kubeschedulerconfig.SchedulerAlgorithmSource,
	stopCh <-chan struct{},
	registry framework.Registry,
	plugins *kubeschedulerconfig.Plugins,
	pluginConfig []kubeschedulerconfig.PluginConfig,
	opts ...func(o *schedulerOptions)) (*Scheduler, error) {

	options := defaultSchedulerOptions
	for _, opt := range opts {
		opt(&options)
	}
	// Set up the configurator which can create schedulers from configs.
	configurator := factory.NewConfigFactory(&factory.ConfigFactoryArgs{
		Client:                         client,
		NodeInformer:                   nodeInformer,
		PodInformer:                    podInformer,
		PvInformer:                     pvInformer,
		PvcInformer:                    pvcInformer,
		ReplicationControllerInformer:  replicationControllerInformer,
		ReplicaSetInformer:             replicaSetInformer,
		StatefulSetInformer:            statefulSetInformer,
		ServiceInformer:                serviceInformer,
		PdbInformer:                    pdbInformer,
		StorageClassInformer:           storageClassInformer,
		CSINodeInformer:                csiNodeInformer,
		HardPodAffinitySymmetricWeight: options.hardPodAffinitySymmetricWeight,
		DisablePreemption:              options.disablePreemption,
		PercentageOfNodesToScore:       options.percentageOfNodesToScore,
		BindTimeoutSeconds:             options.bindTimeoutSeconds,
		Registry:                       registry,
		Plugins:                        plugins,
		PluginConfig:                   pluginConfig,
	})
	var config *factory.Config
	source := schedulerAlgorithmSource
	switch {
	case source.Provider != nil:
		// Create the config from a named algorithm provider.
		sc, err := configurator.CreateFromProvider(*source.Provider)
		if err != nil {
			return nil, fmt.Errorf("couldn't create scheduler using provider %q: %v", *source.Provider, err)
		}
		config = sc
	case source.Policy != nil:
		// Create the config from a user specified policy source.
		policy := &schedulerapi.Policy{}
		switch {
		case source.Policy.File != nil:
			if err := initPolicyFromFile(source.Policy.File.Path, policy); err != nil {
				return nil, err
			}
		case source.Policy.ConfigMap != nil:
			if err := initPolicyFromConfigMap(client, source.Policy.ConfigMap, policy); err != nil {
				return nil, err
			}
		}
		sc, err := configurator.CreateFromConfig(*policy)
		if err != nil {
			return nil, fmt.Errorf("couldn't create scheduler from policy: %v", err)
		}
		config = sc
	default:
		return nil, fmt.Errorf("unsupported algorithm source: %v", source)
	}
	// Additional tweaks to the config produced by the configurator.
	config.Recorder = recorder
	config.DisablePreemption = options.disablePreemption
	config.StopEverything = stopCh

	// Create the scheduler.
	sched := NewFromConfig(config)

	AddAllEventHandlers(sched, options.schedulerName, nodeInformer, podInformer, pvInformer, pvcInformer, serviceInformer, storageClassInformer, csiNodeInformer)
	return sched, nil
}

// initPolicyFromFile initialize policy from file
func initPolicyFromFile(policyFile string, policy *schedulerapi.Policy) error {
	// Use a policy serialized in a file.
	_, err := os.Stat(policyFile)
	if err != nil {
		return fmt.Errorf("missing policy config file %s", policyFile)
	}
	data, err := ioutil.ReadFile(policyFile)
	if err != nil {
		return fmt.Errorf("couldn't read policy config: %v", err)
	}
	err = runtime.DecodeInto(latestschedulerapi.Codec, []byte(data), policy)
	if err != nil {
		return fmt.Errorf("invalid policy: %v", err)
	}
	return nil
}

// initPolicyFromConfigMap initialize policy from configMap
func initPolicyFromConfigMap(client clientset.Interface, policyRef *kubeschedulerconfig.SchedulerPolicyConfigMapSource, policy *schedulerapi.Policy) error {
	// Use a policy serialized in a config map value.
	policyConfigMap, err := client.CoreV1().ConfigMaps(policyRef.Namespace).Get(policyRef.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("couldn't get policy config map %s/%s: %v", policyRef.Namespace, policyRef.Name, err)
	}
	data, found := policyConfigMap.Data[kubeschedulerconfig.SchedulerPolicyConfigMapKey]
	if !found {
		return fmt.Errorf("missing policy config map value at key %q", kubeschedulerconfig.SchedulerPolicyConfigMapKey)
	}
	err = runtime.DecodeInto(latestschedulerapi.Codec, []byte(data), policy)
	if err != nil {
		return fmt.Errorf("invalid policy: %v", err)
	}
	return nil
}

// NewFromConfig returns a new scheduler using the provided Config.
func NewFromConfig(config *factory.Config) *Scheduler {
	metrics.Register()
	return &Scheduler{
		SchedulerCache:      config.SchedulerCache,
		Algorithm:           config.Algorithm,
		GetBinder:           config.GetBinder,
		PodConditionUpdater: config.PodConditionUpdater,
		PodPreemptor:        config.PodPreemptor,
		Framework:           config.Framework,
		NextPod:             config.NextPod,
		WaitForCacheSync:    config.WaitForCacheSync,
		Error:               config.Error,
		Recorder:            config.Recorder,
		StopEverything:      config.StopEverything,
		VolumeBinder:        config.VolumeBinder,
		DisablePreemption:   config.DisablePreemption,
		SchedulingQueue:     config.SchedulingQueue,
	}
}

// Run begins watching and scheduling. It waits for cache to be synced, then starts a goroutine and returns immediately.
func (sched *Scheduler) Run() {
	if !sched.WaitForCacheSync() {
		return
	}

	go wait.Until(sched.scheduleOne, 0, sched.StopEverything)
}

// recordFailedSchedulingEvent records an event for the pod that indicates the
// pod has failed to schedule.
// NOTE: This function modifies "pod". "pod" should be copied before being passed.
func (sched *Scheduler) recordSchedulingFailure(pod *v1.Pod, err error, reason string, message string) {
	sched.Error(pod, err)
	sched.Recorder.Eventf(pod, nil, v1.EventTypeWarning, "FailedScheduling", "Scheduling", message)
	if err := sched.PodConditionUpdater.Update(pod, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  reason,
		Message: err.Error(),
	}); err != nil {
		klog.Errorf("Error updating the condition of the pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}
}

// schedule implements the scheduling algorithm and returns the suggested result(host,
// evaluated nodes number,feasible nodes number).
func (sched *Scheduler) schedule(pod *v1.Pod, pluginContext *framework.PluginContext) (core.ScheduleResult, error) {
	result, err := sched.Algorithm.Schedule(pod, pluginContext)
	if err != nil {
		pod = pod.DeepCopy()
		sched.recordSchedulingFailure(pod, err, v1.PodReasonUnschedulable, err.Error())
		return core.ScheduleResult{}, err
	}
	return result, err
}

// preempt tries to create room for a pod that has failed to schedule, by preempting lower priority pods if possible.
// If it succeeds, it adds the name of the node where preemption has happened to the pod spec.
// It returns the node name and an error if any.
func (sched *Scheduler) preempt(fwk framework.Framework, preemptor *v1.Pod, scheduleErr error) (string, error) {
	preemptor, err := sched.PodPreemptor.GetUpdatedPod(preemptor)
	if err != nil {
		klog.Errorf("Error getting the updated preemptor pod object: %v", err)
		return "", err
	}

	node, victims, nominatedPodsToClear, err := sched.Algorithm.Preempt(preemptor, scheduleErr)
	if err != nil {
		klog.Errorf("Error preempting victims to make room for %v/%v: %v", preemptor.Namespace, preemptor.Name, err)
		return "", err
	}
	var nodeName = ""
	if node != nil {
		nodeName = node.Name
		// Update the scheduling queue with the nominated pod information. Without
		// this, there would be a race condition between the next scheduling cycle
		// and the time the scheduler receives a Pod Update for the nominated pod.
		sched.SchedulingQueue.UpdateNominatedPodForNode(preemptor, nodeName)

		// Make a call to update nominated node name of the pod on the API server.
		err = sched.PodPreemptor.SetNominatedNodeName(preemptor, nodeName)
		if err != nil {
			klog.Errorf("Error in preemption process. Cannot set 'NominatedPod' on pod %v/%v: %v", preemptor.Namespace, preemptor.Name, err)
			sched.SchedulingQueue.DeleteNominatedPodIfExists(preemptor)
			return "", err
		}

		for _, victim := range victims {
			if err := sched.PodPreemptor.DeletePod(victim); err != nil {
				klog.Errorf("Error preempting pod %v/%v: %v", victim.Namespace, victim.Name, err)
				return "", err
			}
			// If the victim is a WaitingPod, send a reject message to the PermitPlugin
			if waitingPod := fwk.GetWaitingPod(victim.UID); waitingPod != nil {
				waitingPod.Reject("preempted")
			}
			sched.Recorder.Eventf(victim, preemptor, v1.EventTypeNormal, "Preempted", "Preempting", "Preempted by %v/%v on node %v", preemptor.Namespace, preemptor.Name, nodeName)

		}
		metrics.PreemptionVictims.Set(float64(len(victims)))
	}
	// Clearing nominated pods should happen outside of "if node != nil". Node could
	// be nil when a pod with nominated node name is eligible to preempt again,
	// but preemption logic does not find any node for it. In that case Preempt()
	// function of generic_scheduler.go returns the pod itself for removal of
	// the 'NominatedPod' field.
	for _, p := range nominatedPodsToClear {
		rErr := sched.PodPreemptor.RemoveNominatedNodeName(p)
		if rErr != nil {
			klog.Errorf("Cannot remove 'NominatedPod' field of pod: %v", rErr)
			// We do not return as this error is not critical.
		}
	}
	return nodeName, err
}

// assumeVolumes will update the volume cache with the chosen bindings
//
// This function modifies assumed if volume binding is required.
func (sched *Scheduler) assumeVolumes(assumed *v1.Pod, host string) (allBound bool, err error) {
	allBound, err = sched.VolumeBinder.Binder.AssumePodVolumes(assumed, host)
	if err != nil {
		sched.recordSchedulingFailure(assumed, err, SchedulerError,
			fmt.Sprintf("AssumePodVolumes failed: %v", err))
	}
	return
}

// bindVolumes will make the API update with the assumed bindings and wait until
// the PV controller has completely finished the binding operation.
//
// If binding errors, times out or gets undone, then an error will be returned to
// retry scheduling.
func (sched *Scheduler) bindVolumes(assumed *v1.Pod) error {
	klog.V(5).Infof("Trying to bind volumes for pod \"%v/%v\"", assumed.Namespace, assumed.Name)
	err := sched.VolumeBinder.Binder.BindPodVolumes(assumed)
	if err != nil {
		klog.V(1).Infof("Failed to bind volumes for pod \"%v/%v\": %v", assumed.Namespace, assumed.Name, err)

		// Unassume the Pod and retry scheduling
		if forgetErr := sched.SchedulerCache.ForgetPod(assumed); forgetErr != nil {
			klog.Errorf("scheduler cache ForgetPod failed: %v", forgetErr)
		}

		sched.recordSchedulingFailure(assumed, err, "VolumeBindingFailed", err.Error())
		return err
	}

	klog.V(5).Infof("Success binding volumes for pod \"%v/%v\"", assumed.Namespace, assumed.Name)
	return nil
}

// assume signals to the cache that a pod is already in the cache, so that binding can be asynchronous.
// assume modifies `assumed`.
func (sched *Scheduler) assume(assumed *v1.Pod, host string) error {
	// Optimistically assume that the binding will succeed and send it to apiserver
	// in the background.
	// If the binding fails, scheduler will release resources allocated to assumed pod
	// immediately.
	assumed.Spec.NodeName = host

	if err := sched.SchedulerCache.AssumePod(assumed); err != nil {
		klog.Errorf("scheduler cache AssumePod failed: %v", err)

		// This is most probably result of a BUG in retrying logic.
		// We report an error here so that pod scheduling can be retried.
		// This relies on the fact that Error will check if the pod has been bound
		// to a node and if so will not add it back to the unscheduled pods queue
		// (otherwise this would cause an infinite loop).
		sched.recordSchedulingFailure(assumed, err, SchedulerError,
			fmt.Sprintf("AssumePod failed: %v", err))
		return err
	}
	// if "assumed" is a nominated pod, we should remove it from internal cache
	if sched.SchedulingQueue != nil {
		sched.SchedulingQueue.DeleteNominatedPodIfExists(assumed)
	}

	return nil
}

// bind binds a pod to a given node defined in a binding object.  We expect this to run asynchronously, so we
// handle binding metrics internally.
func (sched *Scheduler) bind(assumed *v1.Pod, targetNode string, pluginContext *framework.PluginContext) error {
	bindingStart := time.Now()
	bindStatus := sched.Framework.RunBindPlugins(pluginContext, assumed, targetNode)
	var err error
	if !bindStatus.IsSuccess() {
		if bindStatus.Code() == framework.Skip {
			// All bind plugins chose to skip binding of this pod, call original binding function.
			// If binding succeeds then PodScheduled condition will be updated in apiserver so that
			// it's atomic with setting host.
			err = sched.GetBinder(assumed).Bind(&v1.Binding{
				ObjectMeta: metav1.ObjectMeta{Namespace: assumed.Namespace, Name: assumed.Name, UID: assumed.UID},
				Target: v1.ObjectReference{
					Kind: "Node",
					Name: targetNode,
				},
			})
		} else {
			err = fmt.Errorf("Bind failure, code: %d: %v", bindStatus.Code(), bindStatus.Message())
		}
	}
	if finErr := sched.SchedulerCache.FinishBinding(assumed); finErr != nil {
		klog.Errorf("scheduler cache FinishBinding failed: %v", finErr)
	}
	if err != nil {
		klog.V(1).Infof("Failed to bind pod: %v/%v", assumed.Namespace, assumed.Name)
		if err := sched.SchedulerCache.ForgetPod(assumed); err != nil {
			klog.Errorf("scheduler cache ForgetPod failed: %v", err)
		}
		return err
	}

	metrics.BindingLatency.Observe(metrics.SinceInSeconds(bindingStart))
	metrics.DeprecatedBindingLatency.Observe(metrics.SinceInMicroseconds(bindingStart))
	metrics.SchedulingLatency.WithLabelValues(metrics.Binding).Observe(metrics.SinceInSeconds(bindingStart))
	metrics.DeprecatedSchedulingLatency.WithLabelValues(metrics.Binding).Observe(metrics.SinceInSeconds(bindingStart))
	sched.Recorder.Eventf(assumed, nil, v1.EventTypeNormal, "Scheduled", "Binding", "Successfully assigned %v/%v to %v", assumed.Namespace, assumed.Name, targetNode)
	return nil
}

// scheduleOne does the entire scheduling workflow for a single pod.  It is serialized on the scheduling algorithm's host fitting.
func (sched *Scheduler) scheduleOne() {
	fwk := sched.Framework

	pod := sched.NextPod()
	// pod could be nil when schedulerQueue is closed
	if pod == nil {
		return
	}
	if pod.DeletionTimestamp != nil {
		sched.Recorder.Eventf(pod, nil, v1.EventTypeWarning, "FailedScheduling", "Scheduling", "skip schedule deleting pod: %v/%v", pod.Namespace, pod.Name)
		klog.V(3).Infof("Skip schedule deleting pod: %v/%v", pod.Namespace, pod.Name)
		return
	}

	klog.V(3).Infof("Attempting to schedule pod: %v/%v", pod.Namespace, pod.Name)

	// Synchronously attempt to find a fit for the pod.
	start := time.Now()
	pluginContext := framework.NewPluginContext()
	scheduleResult, err := sched.schedule(pod, pluginContext)
	if err != nil {
		// schedule() may have failed because the pod would not fit on any host, so we try to
		// preempt, with the expectation that the next time the pod is tried for scheduling it
		// will fit due to the preemption. It is also possible that a different pod will schedule
		// into the resources that were preempted, but this is harmless.
		if fitError, ok := err.(*core.FitError); ok {
			if sched.DisablePreemption {
				klog.V(3).Infof("Pod priority feature is not enabled or preemption is disabled by scheduler configuration." +
					" No preemption is performed.")
			} else {
				preemptionStartTime := time.Now()
				sched.preempt(fwk, pod, fitError)
				metrics.PreemptionAttempts.Inc()
				metrics.SchedulingAlgorithmPremptionEvaluationDuration.Observe(metrics.SinceInSeconds(preemptionStartTime))
				metrics.DeprecatedSchedulingAlgorithmPremptionEvaluationDuration.Observe(metrics.SinceInMicroseconds(preemptionStartTime))
				metrics.SchedulingLatency.WithLabelValues(metrics.PreemptionEvaluation).Observe(metrics.SinceInSeconds(preemptionStartTime))
				metrics.DeprecatedSchedulingLatency.WithLabelValues(metrics.PreemptionEvaluation).Observe(metrics.SinceInSeconds(preemptionStartTime))
			}
			// Pod did not fit anywhere, so it is counted as a failure. If preemption
			// succeeds, the pod should get counted as a success the next time we try to
			// schedule it. (hopefully)
			metrics.PodScheduleFailures.Inc()
		} else {
			klog.Errorf("error selecting node for pod: %v", err)
			metrics.PodScheduleErrors.Inc()
		}
		return
	}
	metrics.SchedulingAlgorithmLatency.Observe(metrics.SinceInSeconds(start))
	metrics.DeprecatedSchedulingAlgorithmLatency.Observe(metrics.SinceInMicroseconds(start))
	// Tell the cache to assume that a pod now is running on a given node, even though it hasn't been bound yet.
	// This allows us to keep scheduling without waiting on binding to occur.
	assumedPod := pod.DeepCopy()

	// Assume volumes first before assuming the pod.
	//
	// If all volumes are completely bound, then allBound is true and binding will be skipped.
	//
	// Otherwise, binding of volumes is started after the pod is assumed, but before pod binding.
	//
	// This function modifies 'assumedPod' if volume binding is required.
	allBound, err := sched.assumeVolumes(assumedPod, scheduleResult.SuggestedHost)
	if err != nil {
		klog.Errorf("error assuming volumes: %v", err)
		metrics.PodScheduleErrors.Inc()
		return
	}

	// Run "reserve" plugins.
	if sts := fwk.RunReservePlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost); !sts.IsSuccess() {
		sched.recordSchedulingFailure(assumedPod, sts.AsError(), SchedulerError, sts.Message())
		metrics.PodScheduleErrors.Inc()
		return
	}

	// assume modifies `assumedPod` by setting NodeName=scheduleResult.SuggestedHost
	err = sched.assume(assumedPod, scheduleResult.SuggestedHost)
	if err != nil {
		klog.Errorf("error assuming pod: %v", err)
		metrics.PodScheduleErrors.Inc()
		// trigger un-reserve plugins to clean up state associated with the reserved Pod
		fwk.RunUnreservePlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
		return
	}
	// bind the pod to its host asynchronously (we can do this b/c of the assumption step above).
	go func() {
		// Bind volumes first before Pod
		if !allBound {
			err := sched.bindVolumes(assumedPod)
			if err != nil {
				klog.Errorf("error binding volumes: %v", err)
				metrics.PodScheduleErrors.Inc()
				// trigger un-reserve plugins to clean up state associated with the reserved Pod
				fwk.RunUnreservePlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
				return
			}
		}

		// Run "permit" plugins.
		permitStatus := fwk.RunPermitPlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
		if !permitStatus.IsSuccess() {
			var reason string
			if permitStatus.Code() == framework.Unschedulable {
				metrics.PodScheduleFailures.Inc()
				reason = v1.PodReasonUnschedulable
			} else {
				metrics.PodScheduleErrors.Inc()
				reason = SchedulerError
			}
			if forgetErr := sched.Cache().ForgetPod(assumedPod); forgetErr != nil {
				klog.Errorf("scheduler cache ForgetPod failed: %v", forgetErr)
			}
			// trigger un-reserve plugins to clean up state associated with the reserved Pod
			fwk.RunUnreservePlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
			sched.recordSchedulingFailure(assumedPod, permitStatus.AsError(), reason, permitStatus.Message())
			return
		}

		// Run "prebind" plugins.
		prebindStatus := fwk.RunPrebindPlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
		if !prebindStatus.IsSuccess() {
			var reason string
			if prebindStatus.Code() == framework.Unschedulable {
				metrics.PodScheduleFailures.Inc()
				reason = v1.PodReasonUnschedulable
			} else {
				metrics.PodScheduleErrors.Inc()
				reason = SchedulerError
			}
			if forgetErr := sched.Cache().ForgetPod(assumedPod); forgetErr != nil {
				klog.Errorf("scheduler cache ForgetPod failed: %v", forgetErr)
			}
			// trigger un-reserve plugins to clean up state associated with the reserved Pod
			fwk.RunUnreservePlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
			sched.recordSchedulingFailure(assumedPod, prebindStatus.AsError(), reason, prebindStatus.Message())
			return
		}

		err := sched.bind(assumedPod, scheduleResult.SuggestedHost, pluginContext)
		metrics.E2eSchedulingLatency.Observe(metrics.SinceInSeconds(start))
		metrics.DeprecatedE2eSchedulingLatency.Observe(metrics.SinceInMicroseconds(start))
		if err != nil {
			klog.Errorf("error binding pod: %v", err)
			metrics.PodScheduleErrors.Inc()
			// trigger un-reserve plugins to clean up state associated with the reserved Pod
			fwk.RunUnreservePlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
			sched.recordSchedulingFailure(assumedPod, err, SchedulerError, fmt.Sprintf("Binding rejected: %v", err))
		} else {
			// Calculating nodeResourceString can be heavy. Avoid it if klog verbosity is below 2.
			if klog.V(2) {
				node, _ := sched.Cache().GetNodeInfo(scheduleResult.SuggestedHost)
				klog.Infof("pod %v/%v is bound successfully on node %q, %d nodes evaluated, %d nodes were found feasible. Bound node resource: %q.", assumedPod.Namespace, assumedPod.Name, scheduleResult.SuggestedHost, scheduleResult.EvaluatedNodes, scheduleResult.FeasibleNodes, nodeResourceString(node))
			}

			metrics.PodScheduleSuccesses.Inc()

			// Run "postbind" plugins.
			fwk.RunPostbindPlugins(pluginContext, assumedPod, scheduleResult.SuggestedHost)
		}
	}()
}

// nodeResourceString returns a string representation of node resources.
func nodeResourceString(n *v1.Node) string {
	if n == nil {
		return "N/A"
	}
	return fmt.Sprintf("Capacity: %s; Allocatable: %s.", resourceString(&n.Status.Capacity), resourceString(&n.Status.Allocatable))
}

func resourceString(r *v1.ResourceList) string {
	return fmt.Sprintf("CPU<%s>|Memory<%s>|Pods<%s>|StorageEphemeral<%s>", r.Cpu().String(), r.Memory().String(), r.Pods().String(), r.StorageEphemeral().String())
}
