/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"net/http"
	"sync"
	"time"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	mutil "github.com/mesos/mesos-go/mesosutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/backoff"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	malgorithm "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/algorithm"
	schedapi "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/api"
	merrors "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/operations"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util"
	plugin "k8s.io/kubernetes/plugin/pkg/scheduler"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

const (
	pluginRecoveryDelay = 100 * time.Millisecond // delay after scheduler plugin crashes, before we resume scheduling

	FailedScheduling = "FailedScheduling"
	Scheduled        = "Scheduled"
)

type mesosSchedulerApiAdapter struct {
	sync.Mutex
	mesosScheduler *MesosScheduler
}

func (k *mesosSchedulerApiAdapter) Algorithm() malgorithm.PodScheduler {
	return k.mesosScheduler
}

func (k *mesosSchedulerApiAdapter) Offers() offers.Registry {
	return k.mesosScheduler.offers
}

func (k *mesosSchedulerApiAdapter) Tasks() podtask.Registry {
	return k.mesosScheduler.taskRegistry
}

func (k *mesosSchedulerApiAdapter) CreatePodTask(ctx api.Context, pod *api.Pod) (*podtask.T, error) {
	return podtask.New(ctx, "", *pod, k.mesosScheduler.executor)
}

func (k *mesosSchedulerApiAdapter) SlaveHostNameFor(id string) string {
	return k.mesosScheduler.slaveHostNames.HostName(id)
}

func (k *mesosSchedulerApiAdapter) KillTask(taskId string) error {
	killTaskId := mutil.NewTaskID(taskId)
	_, err := k.mesosScheduler.driver.KillTask(killTaskId)
	return err
}

func (k *mesosSchedulerApiAdapter) LaunchTask(task *podtask.T) error {
	// assume caller is holding scheduler lock
	taskList := []*mesos.TaskInfo{task.BuildTaskInfo()}
	offerIds := []*mesos.OfferID{task.Offer.Details().Id}
	filters := &mesos.Filters{}
	_, err := k.mesosScheduler.driver.LaunchTasks(offerIds, taskList, filters)
	return err
}

// k8smSchedulingAlgorithm implements the algorithm.ScheduleAlgorithm interface
type schedulerApiAlgorithmAdapter struct {
	api        schedapi.SchedulerApi
	podUpdates queue.FIFO
}

// Schedule implements the Scheduler interface of Kubernetes.
// It returns the selectedMachine's name and error (if there's any).
func (k *schedulerApiAlgorithmAdapter) Schedule(pod *api.Pod, unused algorithm.NodeLister) (string, error) {
	log.Infof("Try to schedule pod %v\n", pod.Name)
	ctx := api.WithNamespace(api.NewDefaultContext(), pod.Namespace)

	// default upstream scheduler passes pod.Name as binding.PodID
	podKey, err := podtask.MakePodKey(ctx, pod.Name)
	if err != nil {
		return "", err
	}

	k.api.Lock()
	defer k.api.Unlock()

	switch task, state := k.api.Tasks().ForPod(podKey); state {
	case podtask.StateUnknown:
		// There's a bit of a potential race here, a pod could have been yielded() and
		// then before we get *here* it could be deleted.
		// We use meta to index the pod in the store since that's what k8s reflector does.
		podName, err := cache.MetaNamespaceKeyFunc(pod)
		if err != nil {
			log.Warningf("aborting Schedule, unable to understand pod object %+v", pod)
			return "", merrors.NoSuchPodErr
		}
		if deleted := k.podUpdates.Poll(podName, queue.DELETE_EVENT); deleted {
			// avoid scheduling a pod that's been deleted between yieldPod() and Schedule()
			log.Infof("aborting Schedule, pod has been deleted %+v", pod)
			return "", merrors.NoSuchPodErr
		}
		return k.doSchedule(k.api.Tasks().Register(k.api.CreatePodTask(ctx, pod)))

	//TODO(jdef) it's possible that the pod state has diverged from what
	//we knew previously, we should probably update the task.Pod state here
	//before proceeding with scheduling
	case podtask.StatePending:
		if pod.UID != task.Pod.UID {
			// we're dealing with a brand new pod spec here, so the old one must have been
			// deleted -- and so our task store is out of sync w/ respect to reality
			//TODO(jdef) reconcile task
			return "", fmt.Errorf("task %v spec is out of sync with pod %v spec, aborting schedule", task.ID, pod.Name)
		} else if task.Has(podtask.Launched) {
			// task has been marked as "launched" but the pod binding creation may have failed in k8s,
			// but we're going to let someone else handle it, probably the mesos task error handler
			return "", fmt.Errorf("task %s has already been launched, aborting schedule", task.ID)
		} else {
			return k.doSchedule(task, nil)
		}

	default:
		return "", fmt.Errorf("task %s is not pending, nothing to schedule", task.ID)
	}
}

// Call ScheduleFunc and subtract some resources, returning the name of the machine the task is scheduled on
func (k *schedulerApiAlgorithmAdapter) doSchedule(task *podtask.T, err error) (string, error) {
	var offer offers.Perishable
	if task.HasAcceptedOffer() {
		// verify that the offer is still on the table
		offerId := task.GetOfferId()
		if offer, ok := k.api.Offers().Get(offerId); ok && !offer.HasExpired() {
			// skip tasks that have already have assigned offers
			offer = task.Offer
		} else {
			task.Offer.Release()
			task.Reset()
			if err = k.api.Tasks().Update(task); err != nil {
				return "", err
			}
		}
	}
	if err == nil && offer == nil {
		offer, err = k.api.Algorithm().SchedulePod(k.api.Offers(), k.api, task)
	}
	if err != nil {
		return "", err
	}
	details := offer.Details()
	if details == nil {
		return "", fmt.Errorf("offer already invalid/expired for task %v", task.ID)
	}
	slaveId := details.GetSlaveId().GetValue()
	if slaveHostName := k.api.SlaveHostNameFor(slaveId); slaveHostName == "" {
		// not much sense in Release()ing the offer here since its owner died
		offer.Release()
		k.api.Offers().Invalidate(details.Id.GetValue())
		return "", fmt.Errorf("Slave disappeared (%v) while scheduling task %v", slaveId, task.ID)
	} else {
		if task.Offer != nil && task.Offer != offer {
			return "", fmt.Errorf("task.offer assignment must be idempotent, task %+v: offer %+v", task, offer)
		}

		task.Offer = offer
		k.api.Algorithm().Procurement()(task, details) // TODO(jdef) why is nothing checking the error returned here?

		if err := k.api.Tasks().Update(task); err != nil {
			offer.Release()
			return "", err
		}
		return slaveHostName, nil
	}
}

type errorHandler struct {
	api     schedapi.SchedulerApi
	backoff *backoff.Backoff
	qr      *queuer.Queuer
}

// implementation of scheduling plugin's Error func; see plugin/pkg/scheduler
func (k *errorHandler) handleSchedulingError(pod *api.Pod, schedulingErr error) {

	if schedulingErr == merrors.NoSuchPodErr {
		log.V(2).Infof("Not rescheduling non-existent pod %v", pod.Name)
		return
	}

	log.Infof("Error scheduling %v: %v; retrying", pod.Name, schedulingErr)
	defer util.HandleCrash()

	// default upstream scheduler passes pod.Name as binding.PodID
	ctx := api.WithNamespace(api.NewDefaultContext(), pod.Namespace)
	podKey, err := podtask.MakePodKey(ctx, pod.Name)
	if err != nil {
		log.Errorf("Failed to construct pod key, aborting scheduling for pod %v: %v", pod.Name, err)
		return
	}

	k.backoff.GC()
	k.api.Lock()
	defer k.api.Unlock()

	switch task, state := k.api.Tasks().ForPod(podKey); state {
	case podtask.StateUnknown:
		// if we don't have a mapping here any more then someone deleted the pod
		log.V(2).Infof("Could not resolve pod to task, aborting pod reschdule: %s", podKey)
		return

	case podtask.StatePending:
		if task.Has(podtask.Launched) {
			log.V(2).Infof("Skipping re-scheduling for already-launched pod %v", podKey)
			return
		}
		breakoutEarly := queue.BreakChan(nil)
		if schedulingErr == malgorithm.NoSuitableOffersErr {
			log.V(3).Infof("adding backoff breakout handler for pod %v", podKey)
			breakoutEarly = queue.BreakChan(k.api.Offers().Listen(podKey, func(offer *mesos.Offer) bool {
				k.api.Lock()
				defer k.api.Unlock()
				switch task, state := k.api.Tasks().Get(task.ID); state {
				case podtask.StatePending:
					// Assess fitness of pod with the current offer. The scheduler normally
					// "backs off" when it can't find an offer that matches up with a pod.
					// The backoff period for a pod can terminate sooner if an offer becomes
					// available that matches up.
					return !task.Has(podtask.Launched) && k.api.Algorithm().FitPredicate()(task, offer, nil)
				default:
					// no point in continuing to check for matching offers
					return true
				}
			}))
		}
		delay := k.backoff.Get(podKey)
		log.V(3).Infof("requeuing pod %v with delay %v", podKey, delay)
		k.qr.Requeue(&queuer.Pod{Pod: pod, Delay: &delay, Notify: breakoutEarly})

	default:
		log.V(2).Infof("Task is no longer pending, aborting reschedule for pod %v", podKey)
	}
}

// Create creates a scheduler plugin and all supporting background functions.
func (k *MesosScheduler) NewDefaultPluginConfig(terminate <-chan struct{}, mux *http.ServeMux) *PluginConfig {
	// use ListWatch watching pods using the client by default
	return k.NewPluginConfig(terminate, mux, createAllPodsLW(k.client))
}

func (k *MesosScheduler) NewPluginConfig(terminate <-chan struct{}, mux *http.ServeMux,
	podsWatcher *cache.ListWatch) *PluginConfig {

	// Watch and queue pods that need scheduling.
	updates := make(chan queue.Entry, k.schedulerConfig.UpdatesBacklog)
	podUpdates := &podStoreAdapter{queue.NewHistorical(updates)}
	reflector := cache.NewReflector(podsWatcher, &api.Pod{}, podUpdates, 0)

	// lock that guards critial sections that involve transferring pods from
	// the store (cache) to the scheduling queue; its purpose is to maintain
	// an ordering (vs interleaving) of operations that's easier to reason about.
	kapi := &mesosSchedulerApiAdapter{mesosScheduler: k}
	q := queuer.New(podUpdates)
	podDeleter := operations.NewDeleter(kapi, q)
	eh := &errorHandler{
		api:     kapi,
		backoff: backoff.New(k.schedulerConfig.InitialPodBackoff.Duration, k.schedulerConfig.MaxPodBackoff.Duration),
		qr:      q,
	}
	startLatch := make(chan struct{})
	eventBroadcaster := record.NewBroadcaster()
	runtime.On(startLatch, func() {
		eventBroadcaster.StartRecordingToSink(k.client.Events(""))
		reflector.Run() // TODO(jdef) should listen for termination
		podDeleter.Run(updates, terminate)
		q.Run(terminate)

		q.InstallDebugHandlers(mux)
		podtask.InstallDebugHandlers(k.taskRegistry, mux)
	})
	return &PluginConfig{
		Config: &plugin.Config{
			NodeLister: nil,
			Algorithm: &schedulerApiAlgorithmAdapter{
				api:        kapi,
				podUpdates: podUpdates,
			},
			Binder:   operations.NewBinder(kapi),
			NextPod:  q.Yield,
			Error:    eh.handleSchedulingError,
			Recorder: eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduler"}),
		},
		api:      kapi,
		client:   k.client,
		qr:       q,
		deleter:  podDeleter,
		starting: startLatch,
	}
}

type PluginConfig struct {
	*plugin.Config
	api      schedapi.SchedulerApi
	client   *client.Client
	qr       *queuer.Queuer
	deleter  *operations.Deleter
	starting chan struct{} // startup latch
}

func NewPlugin(c *PluginConfig) PluginInterface {
	return &schedulerPlugin{
		config:   c.Config,
		api:      c.api,
		client:   c.client,
		qr:       c.qr,
		deleter:  c.deleter,
		starting: c.starting,
	}
}

type schedulerPlugin struct {
	config   *plugin.Config
	api      schedapi.SchedulerApi
	client   *client.Client
	qr       *queuer.Queuer
	deleter  *operations.Deleter
	starting chan struct{}
}

func (s *schedulerPlugin) Run(done <-chan struct{}) {
	defer close(s.starting)
	go runtime.Until(s.scheduleOne, pluginRecoveryDelay, done)
}

// hacked from GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/mesos_scheduler.go,
// with the Modeler stuff removed since we don't use it because we have mesos.
func (s *schedulerPlugin) scheduleOne() {
	pod := s.config.NextPod()

	// pods which are pre-scheduled (i.e. NodeName is set) are deleted by the kubelet
	// in upstream. Not so in Mesos because the kubelet hasn't see that pod yet. Hence,
	// the scheduler has to take care of this:
	if pod.Spec.NodeName != "" && pod.DeletionTimestamp != nil {
		log.V(3).Infof("deleting pre-scheduled, not yet running pod: %s/%s", pod.Namespace, pod.Name)
		s.client.Pods(pod.Namespace).Delete(pod.Name, api.NewDeleteOptions(0))
		return
	}

	log.V(3).Infof("Attempting to schedule: %+v", pod)
	dest, err := s.config.Algorithm.Schedule(pod, s.config.NodeLister) // call kubeScheduler.Schedule
	if err != nil {
		log.V(1).Infof("Failed to schedule: %+v", pod)
		s.config.Recorder.Eventf(pod, FailedScheduling, "Error scheduling: %v", err)
		s.config.Error(pod, err)
		return
	}
	b := &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name},
		Target: api.ObjectReference{
			Kind: "Node",
			Name: dest,
		},
	}
	if err := s.config.Binder.Bind(b); err != nil {
		log.V(1).Infof("Failed to bind pod: %+v", err)
		s.config.Recorder.Eventf(pod, FailedScheduling, "Binding rejected: %v", err)
		s.config.Error(pod, err)
		return
	}
	s.config.Recorder.Eventf(pod, Scheduled, "Successfully assigned %v to %v", pod.Name, dest)
}

// this pod may be out of sync with respect to the API server registry:
//      this pod   |  apiserver registry
//    -------------|----------------------
//      host=.*    |  404           ; pod was deleted
//      host=.*    |  5xx           ; failed to sync, try again later?
//      host=""    |  host=""       ; perhaps no updates to process?
//      host=""    |  host="..."    ; pod has been scheduled and assigned, is there a task assigned? (check TaskIdKey in binding?)
//      host="..." |  host=""       ; pod is no longer scheduled, does it need to be re-queued?
//      host="..." |  host="..."    ; perhaps no updates to process?
//
// TODO(jdef) this needs an integration test
func (s *schedulerPlugin) reconcileTask(t *podtask.T) {
	log.V(1).Infof("reconcile pod %v, assigned to slave %q", t.Pod.Name, t.Spec.AssignedSlave)
	ctx := api.WithNamespace(api.NewDefaultContext(), t.Pod.Namespace)
	pod, err := s.client.Pods(api.NamespaceValue(ctx)).Get(t.Pod.Name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// attempt to delete
			if err = s.deleter.DeleteOne(&queuer.Pod{Pod: &t.Pod}); err != nil && err != merrors.NoSuchPodErr && err != merrors.NoSuchTaskErr {
				log.Errorf("failed to delete pod: %v: %v", t.Pod.Name, err)
			}
		} else {
			//TODO(jdef) other errors should probably trigger a retry (w/ backoff).
			//For now, drop the pod on the floor
			log.Warning("aborting reconciliation for pod %v: %v", t.Pod.Name, err)
		}
		return
	}

	log.Infof("pod %v scheduled on %q according to apiserver", pod.Name, pod.Spec.NodeName)
	if t.Spec.AssignedSlave != pod.Spec.NodeName {
		if pod.Spec.NodeName == "" {
			// pod is unscheduled.
			// it's possible that we dropped the pod in the scheduler error handler
			// because of task misalignment with the pod (task.Has(podtask.Launched) == true)

			podKey, err := podtask.MakePodKey(ctx, pod.Name)
			if err != nil {
				log.Error(err)
				return
			}

			s.api.Lock()
			defer s.api.Unlock()

			if _, state := s.api.Tasks().ForPod(podKey); state != podtask.StateUnknown {
				//TODO(jdef) reconcile the task
				log.Errorf("task already registered for pod %v", pod.Name)
				return
			}

			now := time.Now()
			log.V(3).Infof("reoffering pod %v", podKey)
			s.qr.Reoffer(queuer.NewPodWithDeadline(pod, &now))
		} else {
			// pod is scheduled.
			// not sure how this happened behind our backs. attempt to reconstruct
			// at least a partial podtask.T record.
			//TODO(jdef) reconcile the task
			log.Errorf("pod already scheduled: %v", pod.Name)
		}
	} else {
		//TODO(jdef) for now, ignore the fact that the rest of the spec may be different
		//and assume that our knowledge of the pod aligns with that of the apiserver
		log.Error("pod reconciliation does not support updates; not yet implemented")
	}
}

func parseSelectorOrDie(s string) fields.Selector {
	selector, err := fields.ParseSelector(s)
	if err != nil {
		panic(err)
	}
	return selector
}

// createAllPodsLW returns a listWatch that finds all pods
func createAllPodsLW(cl *client.Client) *cache.ListWatch {
	return cache.NewListWatchFromClient(cl, "pods", api.NamespaceAll, parseSelectorOrDie(""))
}

// Consumes *api.Pod, produces *Pod; the k8s reflector wants to push *api.Pod
// objects at us, but we want to store more flexible (Pod) type defined in
// this package. The adapter implementation facilitates this. It's a little
// hackish since the object type going in is different than the object type
// coming out -- you've been warned.
type podStoreAdapter struct {
	queue.FIFO
}

func (psa *podStoreAdapter) Add(obj interface{}) error {
	pod := obj.(*api.Pod)
	return psa.FIFO.Add(&queuer.Pod{Pod: pod})
}

func (psa *podStoreAdapter) Update(obj interface{}) error {
	pod := obj.(*api.Pod)
	return psa.FIFO.Update(&queuer.Pod{Pod: pod})
}

func (psa *podStoreAdapter) Delete(obj interface{}) error {
	pod := obj.(*api.Pod)
	return psa.FIFO.Delete(&queuer.Pod{Pod: pod})
}

func (psa *podStoreAdapter) Get(obj interface{}) (interface{}, bool, error) {
	pod := obj.(*api.Pod)
	return psa.FIFO.Get(&queuer.Pod{Pod: pod})
}

// Replace will delete the contents of the store, using instead the
// given map. This store implementation does NOT take ownership of the map.
func (psa *podStoreAdapter) Replace(objs []interface{}, resourceVersion string) error {
	newobjs := make([]interface{}, len(objs))
	for i, v := range objs {
		pod := v.(*api.Pod)
		newobjs[i] = &queuer.Pod{Pod: pod}
	}
	return psa.FIFO.Replace(newobjs, resourceVersion)
}
