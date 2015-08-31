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
	"io"
	"math"
	"net/http"
	"reflect"
	"sync"
	"time"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	mutil "github.com/mesos/mesos-go/mesosutil"
	bindings "github.com/mesos/mesos-go/scheduler"
	execcfg "k8s.io/kubernetes/contrib/mesos/pkg/executor/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/messages"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	offerMetrics "k8s.io/kubernetes/contrib/mesos/pkg/offers/metrics"
	"k8s.io/kubernetes/contrib/mesos/pkg/proc"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	schedcfg "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/metrics"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	mresource "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resource"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/uid"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/util"
)

type Slave struct {
	HostName string
}

func newSlave(hostName string) *Slave {
	return &Slave{
		HostName: hostName,
	}
}

type slaveStorage struct {
	sync.Mutex
	slaves map[string]*Slave // SlaveID => slave.
}

func newSlaveStorage() *slaveStorage {
	return &slaveStorage{
		slaves: make(map[string]*Slave),
	}
}

// Create a mapping between a slaveID and slave if not existing.
func (self *slaveStorage) checkAndAdd(slaveId, slaveHostname string) {
	self.Lock()
	defer self.Unlock()
	_, exists := self.slaves[slaveId]
	if !exists {
		self.slaves[slaveId] = newSlave(slaveHostname)
	}
}

func (self *slaveStorage) getSlaveIds() []string {
	self.Lock()
	defer self.Unlock()
	slaveIds := make([]string, 0, len(self.slaves))
	for slaveID := range self.slaves {
		slaveIds = append(slaveIds, slaveID)
	}
	return slaveIds
}

func (self *slaveStorage) getSlave(slaveId string) (*Slave, bool) {
	self.Lock()
	defer self.Unlock()
	slave, exists := self.slaves[slaveId]
	return slave, exists
}

type PluginInterface interface {
	// the apiserver may have a different state for the pod than we do
	// so reconcile our records, but only for this one pod
	reconcileTask(*podtask.T)

	// execute the Scheduling plugin, should start a go routine and return immediately
	Run(<-chan struct{})
}

// KubernetesScheduler implements:
// 1: A mesos scheduler.
// 2: A kubernetes scheduler plugin.
// 3: A kubernetes pod.Registry.
type KubernetesScheduler struct {
	// We use a lock here to avoid races
	// between invoking the mesos callback
	// and the invoking the pod registry interfaces.
	// In particular, changes to podtask.T objects are currently guarded by this lock.
	*sync.RWMutex

	// Config related, write-once

	schedcfg                 *schedcfg.Config
	executor                 *mesos.ExecutorInfo
	executorGroup            uint64
	scheduleFunc             PodScheduleFunc
	client                   *client.Client
	etcdClient               tools.EtcdClient
	failoverTimeout          float64 // in seconds
	reconcileInterval        int64
	defaultContainerCPULimit mresource.CPUShares
	defaultContainerMemLimit mresource.MegaBytes

	// Mesos context.

	driver         bindings.SchedulerDriver // late initialization
	frameworkId    *mesos.FrameworkID
	masterInfo     *mesos.MasterInfo
	registered     bool
	registration   chan struct{} // signal chan that closes upon first successful registration
	onRegistration sync.Once
	offers         offers.Registry
	slaves         *slaveStorage

	// unsafe state, needs to be guarded

	taskRegistry podtask.Registry

	// via deferred init

	plugin             PluginInterface
	reconciler         *Reconciler
	reconcileCooldown  time.Duration
	asRegisteredMaster proc.Doer
	terminate          <-chan struct{} // signal chan, closes when we should kill background tasks
}

type Config struct {
	Schedcfg                 schedcfg.Config
	Executor                 *mesos.ExecutorInfo
	ScheduleFunc             PodScheduleFunc
	Client                   *client.Client
	EtcdClient               tools.EtcdClient
	FailoverTimeout          float64
	ReconcileInterval        int64
	ReconcileCooldown        time.Duration
	DefaultContainerCPULimit mresource.CPUShares
	DefaultContainerMemLimit mresource.MegaBytes
}

// New creates a new KubernetesScheduler
func New(config Config) *KubernetesScheduler {
	var k *KubernetesScheduler
	k = &KubernetesScheduler{
		schedcfg:                 &config.Schedcfg,
		RWMutex:                  new(sync.RWMutex),
		executor:                 config.Executor,
		executorGroup:            uid.Parse(config.Executor.ExecutorId.GetValue()).Group(),
		scheduleFunc:             config.ScheduleFunc,
		client:                   config.Client,
		etcdClient:               config.EtcdClient,
		failoverTimeout:          config.FailoverTimeout,
		reconcileInterval:        config.ReconcileInterval,
		defaultContainerCPULimit: config.DefaultContainerCPULimit,
		defaultContainerMemLimit: config.DefaultContainerMemLimit,
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				// filter the offers: the executor IDs must not identify a kubelet-
				// executor with a group that doesn't match ours
				for _, eid := range o.GetExecutorIds() {
					execuid := uid.Parse(eid.GetValue())
					if execuid.Name() == execcfg.DefaultInfoID && execuid.Group() != k.executorGroup {
						return false
					}
				}
				return true
			},
			DeclineOffer: func(id string) <-chan error {
				errOnce := proc.NewErrorOnce(k.terminate)
				errOuter := k.asRegisteredMaster.Do(func() {
					var err error
					defer errOnce.Report(err)
					offerId := mutil.NewOfferID(id)
					filters := &mesos.Filters{}
					_, err = k.driver.DeclineOffer(offerId, filters)
				})
				return errOnce.Send(errOuter).Err()
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     config.Schedcfg.OfferLingerTTL.Duration,
			TTL:           config.Schedcfg.OfferTTL.Duration,
			ListenerDelay: config.Schedcfg.ListenerDelay.Duration,
		}),
		slaves:            newSlaveStorage(),
		taskRegistry:      podtask.NewInMemoryRegistry(),
		reconcileCooldown: config.ReconcileCooldown,
		registration:      make(chan struct{}),
		asRegisteredMaster: proc.DoerFunc(func(proc.Action) <-chan error {
			return proc.ErrorChanf("cannot execute action with unregistered scheduler")
		}),
	}
	return k
}

func (k *KubernetesScheduler) Init(electedMaster proc.Process, pl PluginInterface, mux *http.ServeMux) error {
	log.V(1).Infoln("initializing kubernetes mesos scheduler")

	k.asRegisteredMaster = proc.DoerFunc(func(a proc.Action) <-chan error {
		if !k.registered {
			return proc.ErrorChanf("failed to execute action, scheduler is disconnected")
		}
		return electedMaster.Do(a)
	})
	k.terminate = electedMaster.Done()
	k.plugin = pl
	k.offers.Init(k.terminate)
	k.InstallDebugHandlers(mux)
	return k.recoverTasks()
}

func (k *KubernetesScheduler) asMaster() proc.Doer {
	k.RLock()
	defer k.RUnlock()
	return k.asRegisteredMaster
}

func (k *KubernetesScheduler) InstallDebugHandlers(mux *http.ServeMux) {
	wrappedHandler := func(uri string, h http.Handler) {
		mux.HandleFunc(uri, func(w http.ResponseWriter, r *http.Request) {
			ch := make(chan struct{})
			closer := runtime.Closer(ch)
			proc.OnError(k.asMaster().Do(func() {
				defer closer()
				h.ServeHTTP(w, r)
			}), func(err error) {
				defer closer()
				log.Warningf("failed HTTP request for %s: %v", uri, err)
				w.WriteHeader(http.StatusServiceUnavailable)
			}, k.terminate)
			select {
			case <-time.After(k.schedcfg.HttpHandlerTimeout.Duration):
				log.Warningf("timed out waiting for request to be processed")
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			case <-ch: // noop
			}
		})
	}
	requestReconciliation := func(uri string, requestAction func()) {
		wrappedHandler(uri, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestAction()
			w.WriteHeader(http.StatusNoContent)
		}))
	}
	requestReconciliation("/debug/actions/requestExplicit", k.reconciler.RequestExplicit)
	requestReconciliation("/debug/actions/requestImplicit", k.reconciler.RequestImplicit)

	wrappedHandler("/debug/actions/kamikaze", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		slaves := k.slaves.getSlaveIds()
		for _, slaveId := range slaves {
			_, err := k.driver.SendFrameworkMessage(
				k.executor.ExecutorId,
				mutil.NewSlaveID(slaveId),
				messages.Kamikaze)
			if err != nil {
				log.Warningf("failed to send kamikaze message to slave %s: %v", slaveId, err)
			} else {
				io.WriteString(w, fmt.Sprintf("kamikaze slave %s\n", slaveId))
			}
		}
		io.WriteString(w, "OK")
	}))
}

func (k *KubernetesScheduler) Registration() <-chan struct{} {
	return k.registration
}

// Registered is called when the scheduler registered with the master successfully.
func (k *KubernetesScheduler) Registered(drv bindings.SchedulerDriver, fid *mesos.FrameworkID, mi *mesos.MasterInfo) {
	log.Infof("Scheduler registered with the master: %v with frameworkId: %v\n", mi, fid)

	k.driver = drv
	k.frameworkId = fid
	k.masterInfo = mi
	k.registered = true

	k.onRegistration.Do(func() { k.onInitialRegistration(drv) })
	k.reconciler.RequestExplicit()
}

func (k *KubernetesScheduler) storeFrameworkId() {
	// TODO(jdef): port FrameworkId store to generic Kubernetes config store as soon as available
	_, err := k.etcdClient.Set(meta.FrameworkIDKey, k.frameworkId.GetValue(), uint64(k.failoverTimeout))
	if err != nil {
		log.Errorf("failed to renew frameworkId TTL: %v", err)
	}
}

// Reregistered is called when the scheduler re-registered with the master successfully.
// This happends when the master fails over.
func (k *KubernetesScheduler) Reregistered(drv bindings.SchedulerDriver, mi *mesos.MasterInfo) {
	log.Infof("Scheduler reregistered with the master: %v\n", mi)

	k.driver = drv
	k.masterInfo = mi
	k.registered = true

	k.onRegistration.Do(func() { k.onInitialRegistration(drv) })
	k.reconciler.RequestExplicit()
}

// perform one-time initialization actions upon the first registration event received from Mesos.
func (k *KubernetesScheduler) onInitialRegistration(driver bindings.SchedulerDriver) {
	defer close(k.registration)

	if k.failoverTimeout > 0 {
		refreshInterval := k.schedcfg.FrameworkIdRefreshInterval.Duration
		if k.failoverTimeout < k.schedcfg.FrameworkIdRefreshInterval.Duration.Seconds() {
			refreshInterval = time.Duration(math.Max(1, k.failoverTimeout/2)) * time.Second
		}
		go runtime.Until(k.storeFrameworkId, refreshInterval, k.terminate)
	}

	r1 := k.makeTaskRegistryReconciler()
	r2 := k.makePodRegistryReconciler()

	k.reconciler = newReconciler(k.asRegisteredMaster, k.makeCompositeReconciler(r1, r2),
		k.reconcileCooldown, k.schedcfg.ExplicitReconciliationAbortTimeout.Duration, k.terminate)
	go k.reconciler.Run(driver)

	if k.reconcileInterval > 0 {
		ri := time.Duration(k.reconcileInterval) * time.Second
		time.AfterFunc(k.schedcfg.InitialImplicitReconciliationDelay.Duration, func() { runtime.Until(k.reconciler.RequestImplicit, ri, k.terminate) })
		log.Infof("will perform implicit task reconciliation at interval: %v after %v", ri, k.schedcfg.InitialImplicitReconciliationDelay.Duration)
	}
}

// Disconnected is called when the scheduler loses connection to the master.
func (k *KubernetesScheduler) Disconnected(driver bindings.SchedulerDriver) {
	log.Infof("Master disconnected!\n")

	k.registered = false

	// discard all cached offers to avoid unnecessary TASK_LOST updates
	k.offers.Invalidate("")
}

// ResourceOffers is called when the scheduler receives some offers from the master.
func (k *KubernetesScheduler) ResourceOffers(driver bindings.SchedulerDriver, offers []*mesos.Offer) {
	log.V(2).Infof("Received offers %+v", offers)

	// Record the offers in the global offer map as well as each slave's offer map.
	k.offers.Add(offers)
	for _, offer := range offers {
		slaveId := offer.GetSlaveId().GetValue()
		k.slaves.checkAndAdd(slaveId, offer.GetHostname())
	}
}

// OfferRescinded is called when the resources are recinded from the scheduler.
func (k *KubernetesScheduler) OfferRescinded(driver bindings.SchedulerDriver, offerId *mesos.OfferID) {
	log.Infof("Offer rescinded %v\n", offerId)

	oid := offerId.GetValue()
	k.offers.Delete(oid, offerMetrics.OfferRescinded)
}

// StatusUpdate is called when a status update message is sent to the scheduler.
func (k *KubernetesScheduler) StatusUpdate(driver bindings.SchedulerDriver, taskStatus *mesos.TaskStatus) {

	source, reason := "none", "none"
	if taskStatus.Source != nil {
		source = (*taskStatus.Source).String()
	}
	if taskStatus.Reason != nil {
		reason = (*taskStatus.Reason).String()
	}
	taskState := taskStatus.GetState()
	metrics.StatusUpdates.WithLabelValues(source, reason, taskState.String()).Inc()

	message := "none"
	if taskStatus.Message != nil {
		message = *taskStatus.Message
	}

	log.Infof(
		"task status update %q from %q for task %q on slave %q executor %q for reason %q with message %q",
		taskState.String(),
		source,
		taskStatus.TaskId.GetValue(),
		taskStatus.SlaveId.GetValue(),
		taskStatus.ExecutorId.GetValue(),
		reason,
		message,
	)

	switch taskState {
	case mesos.TaskState_TASK_RUNNING, mesos.TaskState_TASK_FINISHED, mesos.TaskState_TASK_STARTING, mesos.TaskState_TASK_STAGING:
		if _, state := k.taskRegistry.UpdateStatus(taskStatus); state == podtask.StateUnknown {
			if taskState != mesos.TaskState_TASK_FINISHED {
				//TODO(jdef) what if I receive this after a TASK_LOST or TASK_KILLED?
				//I don't want to reincarnate then..  TASK_LOST is a special case because
				//the master is stateless and there are scenarios where I may get TASK_LOST
				//followed by TASK_RUNNING.
				//TODO(jdef) consider running this asynchronously since there are API server
				//calls that may be made
				k.reconcileNonTerminalTask(driver, taskStatus)
			} // else, we don't really care about FINISHED tasks that aren't registered
			return
		}
		if _, exists := k.slaves.getSlave(taskStatus.GetSlaveId().GetValue()); !exists {
			// a registered task has an update reported by a slave that we don't recognize.
			// this should never happen! So we don't reconcile it.
			log.Errorf("Ignore status %+v because the slave does not exist", taskStatus)
			return
		}
	case mesos.TaskState_TASK_FAILED, mesos.TaskState_TASK_ERROR:
		if task, _ := k.taskRegistry.UpdateStatus(taskStatus); task != nil {
			if task.Has(podtask.Launched) && !task.Has(podtask.Bound) {
				go k.plugin.reconcileTask(task)
				return
			}
		} else {
			// unknown task failed, not much we can do about it
			return
		}
		// last-ditch effort to reconcile our records
		fallthrough
	case mesos.TaskState_TASK_LOST, mesos.TaskState_TASK_KILLED:
		k.reconcileTerminalTask(driver, taskStatus)
	default:
		log.Errorf(
			"unknown task status %q from %q for task %q on slave %q executor %q for reason %q with message %q",
			taskState.String(),
			source,
			taskStatus.TaskId.GetValue(),
			taskStatus.SlaveId.GetValue(),
			taskStatus.ExecutorId.GetValue(),
			reason,
			message,
		)
	}
}

func (k *KubernetesScheduler) reconcileTerminalTask(driver bindings.SchedulerDriver, taskStatus *mesos.TaskStatus) {
	task, state := k.taskRegistry.UpdateStatus(taskStatus)

	if (state == podtask.StateRunning || state == podtask.StatePending) &&
		((taskStatus.GetSource() == mesos.TaskStatus_SOURCE_MASTER && taskStatus.GetReason() == mesos.TaskStatus_REASON_RECONCILIATION) ||
			(taskStatus.GetSource() == mesos.TaskStatus_SOURCE_SLAVE && taskStatus.GetReason() == mesos.TaskStatus_REASON_EXECUTOR_TERMINATED) ||
			(taskStatus.GetSource() == mesos.TaskStatus_SOURCE_SLAVE && taskStatus.GetReason() == mesos.TaskStatus_REASON_EXECUTOR_UNREGISTERED)) {
		//--
		// pod-task has metadata that refers to:
		// (1) a task that Mesos no longer knows about, or else
		// (2) a pod that the Kubelet will never report as "failed"
		// For now, destroy the pod and hope that there's a replication controller backing it up.
		// TODO(jdef) for case #2 don't delete the pod, just update it's status to Failed
		pod := &task.Pod
		log.Warningf("deleting rogue pod %v/%v for lost task %v", pod.Namespace, pod.Name, task.ID)
		if err := k.client.Pods(pod.Namespace).Delete(pod.Name, nil); err != nil && !errors.IsNotFound(err) {
			log.Errorf("failed to delete pod %v/%v for terminal task %v: %v", pod.Namespace, pod.Name, task.ID, err)
		}
	} else if taskStatus.GetReason() == mesos.TaskStatus_REASON_EXECUTOR_TERMINATED || taskStatus.GetReason() == mesos.TaskStatus_REASON_EXECUTOR_UNREGISTERED {
		// attempt to prevent dangling pods in the pod and task registries
		log.V(1).Infof("request explicit reconciliation to clean up for task %v after executor reported (terminated/unregistered)", taskStatus.TaskId.GetValue())
		k.reconciler.RequestExplicit()
	} else if taskStatus.GetState() == mesos.TaskState_TASK_LOST && state == podtask.StateRunning && taskStatus.ExecutorId != nil && taskStatus.SlaveId != nil {
		//TODO(jdef) this may not be meaningful once we have proper checkpointing and master detection
		//If we're reconciling and receive this then the executor may be
		//running a task that we need it to kill. It's possible that the framework
		//is unrecognized by the master at this point, so KillTask is not guaranteed
		//to do anything. The underlying driver transport may be able to send a
		//FrameworkMessage directly to the slave to terminate the task.
		log.V(2).Info("forwarding TASK_LOST message to executor %v on slave %v", taskStatus.ExecutorId, taskStatus.SlaveId)
		data := fmt.Sprintf("task-lost:%s", task.ID) //TODO(jdef) use a real message type
		if _, err := driver.SendFrameworkMessage(taskStatus.ExecutorId, taskStatus.SlaveId, data); err != nil {
			log.Error(err.Error())
		}
	}
}

// reconcile an unknown (from the perspective of our registry) non-terminal task
func (k *KubernetesScheduler) reconcileNonTerminalTask(driver bindings.SchedulerDriver, taskStatus *mesos.TaskStatus) {
	// attempt to recover task from pod info:
	// - task data may contain an api.PodStatusResult; if status.reason == REASON_RECONCILIATION then status.data == nil
	// - the Name can be parsed by container.ParseFullName() to yield a pod Name and Namespace
	// - pull the pod metadata down from the api server
	// - perform task recovery based on pod metadata
	taskId := taskStatus.TaskId.GetValue()
	if taskStatus.GetReason() == mesos.TaskStatus_REASON_RECONCILIATION && taskStatus.GetSource() == mesos.TaskStatus_SOURCE_MASTER {
		// there will be no data in the task status that we can use to determine the associated pod
		switch taskStatus.GetState() {
		case mesos.TaskState_TASK_STAGING:
			// there is still hope for this task, don't kill it just yet
			//TODO(jdef) there should probably be a limit for how long we tolerate tasks stuck in this state
			return
		default:
			// for TASK_{STARTING,RUNNING} we should have already attempted to recoverTasks() for.
			// if the scheduler failed over before the executor fired TASK_STARTING, then we should *not*
			// be processing this reconciliation update before we process the one from the executor.
			// point: we don't know what this task is (perhaps there was unrecoverable metadata in the pod),
			// so it gets killed.
			log.Errorf("killing non-terminal, unrecoverable task %v", taskId)
		}
	} else if podStatus, err := podtask.ParsePodStatusResult(taskStatus); err != nil {
		// possible rogue pod exists at this point because we can't identify it; should kill the task
		log.Errorf("possible rogue pod; illegal task status data for task %v, expected an api.PodStatusResult: %v", taskId, err)
	} else if name, namespace, err := container.ParsePodFullName(podStatus.Name); err != nil {
		// possible rogue pod exists at this point because we can't identify it; should kill the task
		log.Errorf("possible rogue pod; illegal api.PodStatusResult, unable to parse full pod name from: '%v' for task %v: %v",
			podStatus.Name, taskId, err)
	} else if pod, err := k.client.Pods(namespace).Get(name); err == nil {
		if t, ok, err := podtask.RecoverFrom(*pod); ok {
			log.Infof("recovered task %v from metadata in pod %v/%v", taskId, namespace, name)
			_, err := k.taskRegistry.Register(t, nil)
			if err != nil {
				// someone beat us to it?!
				log.Warningf("failed to register recovered task: %v", err)
				return
			} else {
				k.taskRegistry.UpdateStatus(taskStatus)
			}
			return
		} else if err != nil {
			//should kill the pod and the task
			log.Errorf("killing pod, failed to recover task from pod %v/%v: %v", namespace, name, err)
			if err := k.client.Pods(namespace).Delete(name, nil); err != nil {
				log.Errorf("failed to delete pod %v/%v: %v", namespace, name, err)
			}
		} else {
			//this is pretty unexpected: we received a TASK_{STARTING,RUNNING} message, but the apiserver's pod
			//metadata is not appropriate for task reconstruction -- which should almost certainly never
			//be the case unless someone swapped out the pod on us (and kept the same namespace/name) while
			//we were failed over.

			//kill this task, allow the newly launched scheduler to schedule the new pod
			log.Warningf("unexpected pod metadata for task %v in apiserver, assuming new unscheduled pod spec: %+v", taskId, pod)
		}
	} else if errors.IsNotFound(err) {
		// pod lookup failed, should delete the task since the pod is no longer valid; may be redundant, that's ok
		log.Infof("killing task %v since pod %v/%v no longer exists", taskId, namespace, name)
	} else if errors.IsServerTimeout(err) {
		log.V(2).Infof("failed to reconcile task due to API server timeout: %v", err)
		return
	} else {
		log.Errorf("unexpected API server error, aborting reconcile for task %v: %v", taskId, err)
		return
	}
	if _, err := driver.KillTask(taskStatus.TaskId); err != nil {
		log.Errorf("failed to kill task %v: %v", taskId, err)
	}
}

// FrameworkMessage is called when the scheduler receives a message from the executor.
func (k *KubernetesScheduler) FrameworkMessage(driver bindings.SchedulerDriver,
	executorId *mesos.ExecutorID, slaveId *mesos.SlaveID, message string) {
	log.Infof("Received messages from executor %v of slave %v, %v\n", executorId, slaveId, message)
}

// SlaveLost is called when some slave is lost.
func (k *KubernetesScheduler) SlaveLost(driver bindings.SchedulerDriver, slaveId *mesos.SlaveID) {
	log.Infof("Slave %v is lost\n", slaveId)

	sid := slaveId.GetValue()
	k.offers.InvalidateForSlave(sid)

	// TODO(jdef): delete slave from our internal list? probably not since we may need to reconcile
	// tasks. it would be nice to somehow flag the slave as lost so that, perhaps, we can periodically
	// flush lost slaves older than X, and for which no tasks or pods reference.

	// unfinished tasks/pods will be dropped. use a replication controller if you want pods to
	// be restarted when slaves die.
}

// ExecutorLost is called when some executor is lost.
func (k *KubernetesScheduler) ExecutorLost(driver bindings.SchedulerDriver, executorId *mesos.ExecutorID, slaveId *mesos.SlaveID, status int) {
	log.Infof("Executor %v of slave %v is lost, status: %v\n", executorId, slaveId, status)
	// TODO(yifan): Restart any unfinished tasks of the executor.
}

// Error is called when there is an unrecoverable error in the scheduler or scheduler driver.
// The driver should have been aborted before this is invoked.
func (k *KubernetesScheduler) Error(driver bindings.SchedulerDriver, message string) {
	log.Fatalf("fatal scheduler error: %v\n", message)
}

// filter func used for explicit task reconciliation, selects only non-terminal tasks which
// have been communicated to mesos (read: launched).
func explicitTaskFilter(t *podtask.T) bool {
	switch t.State {
	case podtask.StateRunning:
		return true
	case podtask.StatePending:
		return t.Has(podtask.Launched)
	default:
		return false
	}
}

// invoke the given ReconcilerAction funcs in sequence, aborting the sequence if reconciliation
// is cancelled. if any other errors occur the composite reconciler will attempt to complete the
// sequence, reporting only the last generated error.
func (k *KubernetesScheduler) makeCompositeReconciler(actions ...ReconcilerAction) ReconcilerAction {
	if x := len(actions); x == 0 {
		// programming error
		panic("no actions specified for composite reconciler")
	} else if x == 1 {
		return actions[0]
	}
	chained := func(d bindings.SchedulerDriver, c <-chan struct{}, a, b ReconcilerAction) <-chan error {
		ech := a(d, c)
		ch := make(chan error, 1)
		go func() {
			select {
			case <-k.terminate:
			case <-c:
			case e := <-ech:
				if e != nil {
					ch <- e
					return
				}
				ech = b(d, c)
				select {
				case <-k.terminate:
				case <-c:
				case e := <-ech:
					if e != nil {
						ch <- e
						return
					}
					close(ch)
					return
				}
			}
			ch <- fmt.Errorf("aborting composite reconciler action")
		}()
		return ch
	}
	result := func(d bindings.SchedulerDriver, c <-chan struct{}) <-chan error {
		return chained(d, c, actions[0], actions[1])
	}
	for i := 2; i < len(actions); i++ {
		i := i
		next := func(d bindings.SchedulerDriver, c <-chan struct{}) <-chan error {
			return chained(d, c, ReconcilerAction(result), actions[i])
		}
		result = next
	}
	return ReconcilerAction(result)
}

// reconciler action factory, performs explicit task reconciliation for non-terminal
// tasks listed in the scheduler's internal taskRegistry.
func (k *KubernetesScheduler) makeTaskRegistryReconciler() ReconcilerAction {
	return ReconcilerAction(func(drv bindings.SchedulerDriver, cancel <-chan struct{}) <-chan error {
		taskToSlave := make(map[string]string)
		for _, t := range k.taskRegistry.List(explicitTaskFilter) {
			if t.Spec.SlaveID != "" {
				taskToSlave[t.ID] = t.Spec.SlaveID
			}
		}
		return proc.ErrorChan(k.explicitlyReconcileTasks(drv, taskToSlave, cancel))
	})
}

// reconciler action factory, performs explicit task reconciliation for non-terminal
// tasks identified by annotations in the Kubernetes pod registry.
func (k *KubernetesScheduler) makePodRegistryReconciler() ReconcilerAction {
	return ReconcilerAction(func(drv bindings.SchedulerDriver, cancel <-chan struct{}) <-chan error {
		ctx := api.NewDefaultContext()
		podList, err := k.client.Pods(api.NamespaceValue(ctx)).List(labels.Everything(), fields.Everything())
		if err != nil {
			return proc.ErrorChanf("failed to reconcile pod registry: %v", err)
		}
		taskToSlave := make(map[string]string)
		for _, pod := range podList.Items {
			if len(pod.Annotations) == 0 {
				continue
			}
			taskId, found := pod.Annotations[meta.TaskIdKey]
			if !found {
				continue
			}
			slaveId, found := pod.Annotations[meta.SlaveIdKey]
			if !found {
				continue
			}
			taskToSlave[taskId] = slaveId
		}
		return proc.ErrorChan(k.explicitlyReconcileTasks(drv, taskToSlave, cancel))
	})
}

// execute an explicit task reconciliation, as per http://mesos.apache.org/documentation/latest/reconciliation/
func (k *KubernetesScheduler) explicitlyReconcileTasks(driver bindings.SchedulerDriver, taskToSlave map[string]string, cancel <-chan struct{}) error {
	log.Info("explicit reconcile tasks")

	// tell mesos to send us the latest status updates for all the non-terminal tasks that we know about
	statusList := []*mesos.TaskStatus{}
	remaining := util.KeySet(reflect.ValueOf(taskToSlave))
	for taskId, slaveId := range taskToSlave {
		if slaveId == "" {
			delete(taskToSlave, taskId)
			continue
		}
		statusList = append(statusList, &mesos.TaskStatus{
			TaskId:  mutil.NewTaskID(taskId),
			SlaveId: mutil.NewSlaveID(slaveId),
			State:   mesos.TaskState_TASK_RUNNING.Enum(), // req'd field, doesn't have to reflect reality
		})
	}

	select {
	case <-cancel:
		return reconciliationCancelledErr
	default:
		if _, err := driver.ReconcileTasks(statusList); err != nil {
			return err
		}
	}

	start := time.Now()
	first := true
	for backoff := 1 * time.Second; first || remaining.Len() > 0; backoff = backoff * 2 {
		first = false
		// nothing to do here other than wait for status updates..
		if backoff > k.schedcfg.ExplicitReconciliationMaxBackoff.Duration {
			backoff = k.schedcfg.ExplicitReconciliationMaxBackoff.Duration
		}
		select {
		case <-cancel:
			return reconciliationCancelledErr
		case <-time.After(backoff):
			for taskId := range remaining {
				if task, _ := k.taskRegistry.Get(taskId); task != nil && explicitTaskFilter(task) && task.UpdatedTime.Before(start) {
					// keep this task in remaining list
					continue
				}
				remaining.Delete(taskId)
			}
		}
	}
	return nil
}

var (
	reconciliationCancelledErr = fmt.Errorf("explicit task reconciliation cancelled")
)

type ReconcilerAction func(driver bindings.SchedulerDriver, cancel <-chan struct{}) <-chan error

type Reconciler struct {
	proc.Doer
	Action                             ReconcilerAction
	explicit                           chan struct{}   // send an empty struct to trigger explicit reconciliation
	implicit                           chan struct{}   // send an empty struct to trigger implicit reconciliation
	done                               <-chan struct{} // close this when you want the reconciler to exit
	cooldown                           time.Duration
	explicitReconciliationAbortTimeout time.Duration
}

func newReconciler(doer proc.Doer, action ReconcilerAction,
	cooldown, explicitReconciliationAbortTimeout time.Duration, done <-chan struct{}) *Reconciler {
	return &Reconciler{
		Doer:     doer,
		explicit: make(chan struct{}, 1),
		implicit: make(chan struct{}, 1),
		cooldown: cooldown,
		explicitReconciliationAbortTimeout: explicitReconciliationAbortTimeout,
		done: done,
		Action: func(driver bindings.SchedulerDriver, cancel <-chan struct{}) <-chan error {
			// trigged the reconciler action in the doer's execution context,
			// but it could take a while and the scheduler needs to be able to
			// process updates, the callbacks for which ALSO execute in the SAME
			// deferred execution context -- so the action MUST be executed async.
			errOnce := proc.NewErrorOnce(cancel)
			return errOnce.Send(doer.Do(func() {
				// only triggers the action if we're the currently elected,
				// registered master and runs the action async.
				go func() {
					var err <-chan error
					defer errOnce.Send(err)
					err = action(driver, cancel)
				}()
			})).Err()
		},
	}
}

func (r *Reconciler) RequestExplicit() {
	select {
	case r.explicit <- struct{}{}: // noop
	default: // request queue full; noop
	}
}

func (r *Reconciler) RequestImplicit() {
	select {
	case r.implicit <- struct{}{}: // noop
	default: // request queue full; noop
	}
}

// execute task reconciliation, returns when r.done is closed. intended to run as a goroutine.
// if reconciliation is requested while another is in progress, the in-progress operation will be
// cancelled before the new reconciliation operation begins.
func (r *Reconciler) Run(driver bindings.SchedulerDriver) {
	var cancel, finished chan struct{}
requestLoop:
	for {
		select {
		case <-r.done:
			return
		default: // proceed
		}
		select {
		case <-r.implicit:
			metrics.ReconciliationRequested.WithLabelValues("implicit").Inc()
			select {
			case <-r.done:
				return
			case <-r.explicit:
				break // give preference to a pending request for explicit
			default: // continue
				// don't run implicit reconciliation while explicit is ongoing
				if finished != nil {
					select {
					case <-finished: // continue w/ implicit
					default:
						log.Infoln("skipping implicit reconcile because explicit reconcile is ongoing")
						continue requestLoop
					}
				}
				errOnce := proc.NewErrorOnce(r.done)
				errCh := r.Do(func() {
					var err error
					defer errOnce.Report(err)
					log.Infoln("implicit reconcile tasks")
					metrics.ReconciliationExecuted.WithLabelValues("implicit").Inc()
					if _, err = driver.ReconcileTasks([]*mesos.TaskStatus{}); err != nil {
						log.V(1).Infof("failed to request implicit reconciliation from mesos: %v", err)
					}
				})
				proc.OnError(errOnce.Send(errCh).Err(), func(err error) {
					log.Errorf("failed to run implicit reconciliation: %v", err)
				}, r.done)
				goto slowdown
			}
		case <-r.done:
			return
		case <-r.explicit: // continue
			metrics.ReconciliationRequested.WithLabelValues("explicit").Inc()
		}

		if cancel != nil {
			close(cancel)
			cancel = nil

			// play nice and wait for the prior operation to finish, complain
			// if it doesn't
			select {
			case <-r.done:
				return
			case <-finished: // noop, expected
			case <-time.After(r.explicitReconciliationAbortTimeout): // very unexpected
				log.Error("reconciler action failed to stop upon cancellation")
			}
		}
		// copy 'finished' to 'fin' here in case we end up with simultaneous go-routines,
		// if cancellation takes too long or fails - we don't want to close the same chan
		// more than once
		cancel = make(chan struct{})
		finished = make(chan struct{})
		go func(fin chan struct{}) {
			startedAt := time.Now()
			defer func() {
				metrics.ReconciliationLatency.Observe(metrics.InMicroseconds(time.Since(startedAt)))
			}()

			metrics.ReconciliationExecuted.WithLabelValues("explicit").Inc()
			defer close(fin)
			err := <-r.Action(driver, cancel)
			if err == reconciliationCancelledErr {
				metrics.ReconciliationCancelled.WithLabelValues("explicit").Inc()
				log.Infoln(err.Error())
			} else if err != nil {
				log.Errorf("reconciler action failed: %v", err)
			}
		}(finished)
	slowdown:
		// don't allow reconciliation to run very frequently, either explicit or implicit
		select {
		case <-r.done:
			return
		case <-time.After(r.cooldown): // noop
		}
	} // for
}

func (ks *KubernetesScheduler) recoverTasks() error {
	ctx := api.NewDefaultContext()
	podList, err := ks.client.Pods(api.NamespaceValue(ctx)).List(labels.Everything(), fields.Everything())
	if err != nil {
		log.V(1).Infof("failed to recover pod registry, madness may ensue: %v", err)
		return err
	}
	recoverSlave := func(t *podtask.T) {

		slaveId := t.Spec.SlaveID
		ks.slaves.checkAndAdd(slaveId, t.Offer.Host())
	}
	for _, pod := range podList.Items {
		if t, ok, err := podtask.RecoverFrom(pod); err != nil {
			log.Errorf("failed to recover task from pod, will attempt to delete '%v/%v': %v", pod.Namespace, pod.Name, err)
			err := ks.client.Pods(pod.Namespace).Delete(pod.Name, nil)
			//TODO(jdef) check for temporary or not-found errors
			if err != nil {
				log.Errorf("failed to delete pod '%v/%v': %v", pod.Namespace, pod.Name, err)
			}
		} else if ok {
			ks.taskRegistry.Register(t, nil)
			recoverSlave(t)
			log.Infof("recovered task %v from pod %v/%v", t.ID, pod.Namespace, pod.Name)
		}
	}
	return nil
}
