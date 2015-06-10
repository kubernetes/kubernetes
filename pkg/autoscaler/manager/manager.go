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

package manager

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
	sampleadvisors "github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler/advisors/sample"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/workqueue"
	"github.com/golang/glog"
)

// How often auto scale policies are assessed and applied.
const (
	// How often auto scaling policies are assessed.
	DefaultAssessmentInterval = 30 * time.Second

	// How often auto scaling policies are applied.
	DefaultAutoScalingInterval = 60 * time.Second

	// One second.
	OneSecond = 1 * time.Second
)

// AutoScaleManager tracks and manages registered plugins and advisors.
type AutoScaleManager struct {
	Client client.Interface //  Kube client

	AssessmentInterval  time.Duration //  How often to assess policies.
	AutoScalingInterval time.Duration //  How often to scale.

	mutex sync.Mutex // Lock for controlling exclusive access.

	plugins  map[string]autoscaler.AutoScalerPlugin // Plugins.
	advisors map[string]autoscaler.Advisor          // Advisors.

	queue *workqueue.Type // Work queue.
}

// Create a new auto scaler manager instance.
func NewAutoScaleManager(kubeClient client.Interface, assessEvery, scaleEvery time.Duration) *AutoScaleManager {
	if assessEvery < OneSecond {
		assessEvery = DefaultAssessmentInterval
	}

	if scaleEvery < OneSecond {
		scaleEvery = DefaultAutoScalingInterval
	}

	return &AutoScaleManager{
		Client: kubeClient,

		//  Intervals for assessment and auto-scaling.
		AssessmentInterval:  assessEvery,
		AutoScalingInterval: scaleEvery,

		queue: workqueue.New(),
	}
}

// Register an auto scaler plugin with the manager.
func (m *AutoScaleManager) Register(plugin autoscaler.AutoScalerPlugin) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if nil == m.plugins {
		m.plugins = make(map[string]autoscaler.AutoScalerPlugin, 0)
	}

	name := plugin.Name()

	if _, registered := m.plugins[name]; registered {
		return fmt.Errorf("Autoscaler plugin %q was registered more than once", name)
	}

	m.plugins[name] = plugin

	glog.Infof("Registered autoscaler plugin %q", name)
	return nil
}

// {Un,De}registers an auto scaler plugin with the manager.
func (m *AutoScaleManager) Deregister(name string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if nil == m.plugins {
		return fmt.Errorf("no autoscaler plugin %q was registered", name)
	}

	if _, registered := m.plugins[name]; !registered {
		return fmt.Errorf("no autoscaler plugin %q was registered", name)
	}

	delete(m.plugins, name)

	glog.Infof("Deregistered autoscaler plugin %q", name)
	return nil
}

// FindPluginByName finds an autoscaler plugin by name. If no plugin is
// found, returns an error.
func (m *AutoScaleManager) FindPluginByName(name string) (*autoscaler.AutoScalerPlugin, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	for _, v := range m.plugins {
		if v.Name() == name {
			return &v, nil
		}
	}

	return nil, fmt.Errorf("no autoscaler plugin matched name %q", name)
}

// Add an advisor.
func (m *AutoScaleManager) AddAdvisor(advisor autoscaler.Advisor) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if nil == m.advisors {
		m.advisors = make(map[string]autoscaler.Advisor, 0)
	}

	name := advisor.Name()

	if _, exists := m.advisors[name]; exists {
		return fmt.Errorf("Advisor %q added more than once", name)
	}

	m.advisors[name] = advisor
	err := advisor.Initialize(m.Client)
	if nil == err {
		glog.Infof("Added auto scale advisor %q", name)
	}

	return err
}

// Remove an advisor.
func (m *AutoScaleManager) RemoveAdvisor(name string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if nil == m.advisors {
		return fmt.Errorf("no Advisors exist", name)
	}

	if _, exists := m.advisors[name]; !exists {
		return fmt.Errorf("no Advisor %q exists", name)
	}

	delete(m.advisors, name)

	glog.Infof("Removed auto scale advisor %q", name)
	return nil
}

// Returns a list of all the advisors the manager is using.
func (m *AutoScaleManager) GetAdvisors() []autoscaler.Advisor {
	advisors := make([]autoscaler.Advisor, len(m.advisors))

	m.mutex.Lock()
	defer m.mutex.Unlock()

	idx := 0
	for _, ad := range m.advisors {
		advisors[idx] = ad
		idx += 1
	}

	return advisors
}

// Adds all the default advisors.
func (m *AutoScaleManager) AddDefaultAdvisors() error {
	allErrors := make([]error, 0)

	truthinessAdvisor := sampleadvisors.NewTruthinessScalingAdvisor()
	falsinessAdvisor := sampleadvisors.NewFalsinessScalingAdvisor()
	advisors := []autoscaler.Advisor{truthinessAdvisor, falsinessAdvisor}

	for _, ad := range advisors {
		err := m.AddAdvisor(ad)
		if err != nil {
			allErrors = append(allErrors, err)
		}
	}

	return errors.NewAggregate(allErrors)
}

// scheduleWork checks all autoscale policies and enqueues policies that
// need to be assessed (if not recently scaled). The workqueue tasks are
// asynchronously processes by one of the queueWorkers.
func (m *AutoScaleManager) scheduleWork() {
	autoscalers, err := m.Client.AutoScalers(api.NamespaceAll).List(labels.Everything(), fields.Everything())
	if err != nil {
		// TODO(ramr): Generate an event in lieu of a log message.
		glog.Error(err)
		return
	}

	for _, policy := range autoscalers.Items {
		// Check if policy was recently assessed and if so, its a
		// NOOP here - otherwise enqueue it for one of the workers
		// to pick it up.
		ts := policy.Status.LastActionTimestamp.Add(m.AutoScalingInterval)
		nextScaleAt := util.Time{ts}
		if !util.Now().Before(nextScaleAt) {
			data, err := json.Marshal(policy)
			if err != nil {
				glog.Error("Error marshalling auto scaler policy: %v", err)
				continue
			}

			m.queue.Add(string(data))
		}
	}
}

// Run begins watching autoscaler configurations and processing them.
func (m *AutoScaleManager) Run(workers int, stopCh <-chan struct{}) {
	defer util.HandleCrash()

	go util.Until(m.scheduleWork, m.AssessmentInterval, stopCh)

	for i := 0; i < workers; i++ {
		go util.Until(m.queueWorker, time.Second, stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down AutoScale Manager")
	m.queue.ShutDown()
}

// Runs a worker thread that dequeues and processes workqueue items and
// marks them as done.
func (m *AutoScaleManager) queueWorker() {
	for {
		func() {
			key, shutdown := m.queue.Get()
			if shutdown {
				return
			}

			defer m.queue.Done(key)
			data := []byte(key.(string))

			var policy api.AutoScaler
			if err := json.Unmarshal(data, &policy); err != nil {
				// TODO(ramr): Generate an event in lieu of a log message.
				glog.Error("processing auto scaling policy: %v", err)
			}

			err := m.processPolicy(policy)
			// TODO(ramr): Generate an event in lieu of a log message.
			if err != nil {
				glog.Error("processing auto scaling policy: %v", err)
			}
		}()
	}
}

// Processes an autoscaler policy using all configured plugins and advisors.
func (m *AutoScaleManager) processPolicy(policy api.AutoScaler) error {
	advisors := m.GetAdvisors()
	allActions := make([]autoscaler.ScalingAction, 0)
	allErrors := make([]error, 0)

	// Ensure we have a target to operate on (replication controllers
	// to autoscale) before assessing the policy with the advisors.
	selector := labels.SelectorFromSet(policy.Spec.TargetSelector)
	rcList, err := m.Client.ReplicationControllers(api.NamespaceAll).List(selector)
	if err != nil {
		return err
	}

	if len(rcList.Items) < 1 {
		// No target to autoscale - do nothing.
		glog.Infof("No autoscaler target (RC) for selector=%+v", policy.Spec.TargetSelector)
		return nil
	}

	for _, plugin := range m.plugins {
		actions, err := plugin.Assess(policy.Spec, advisors)
		if err != nil {
			allErrors = append(allErrors, err)
			continue
		}

		for _, action := range actions {
			allActions = append(allActions, action)
		}
	}

	// Process actions (speak louder than words!!) and return errors.
	if len(allActions) > 0 {
		glog.Infof("Processing autoscaler policy: %+v, actions=%+v", policy, allActions)
		err := m.processAutoScalerActions(&policy, allActions)
		if err != nil {
			// TODO: Generate an event in lieu of a log message.
			glog.Error("processing auto scale action: %v", err)
		}
	}

	return errors.NewAggregate(allErrors)
}

// Process the actions to take for the auto scaler policy.
func (m *AutoScaleManager) processAutoScalerActions(policy *api.AutoScaler, actions []autoscaler.ScalingAction) error {
	allErrors := make([]error, 0)
	delta := 0

	action := ReconcileActions(actions)
	switch action.Type {
	case api.AutoScaleActionTypeNone:
		return errors.NewAggregate(allErrors)

	case api.AutoScaleActionTypeScaleUp:
		delta = action.ScaleBy

	case api.AutoScaleActionTypeScaleDown:
		delta = 0 - action.ScaleBy
	}

	updatedList := make([]string, 0)

	selector := labels.SelectorFromSet(policy.Spec.TargetSelector)
	rcList, err := m.Client.ReplicationControllers(api.NamespaceAll).List(selector)
	for _, rc := range rcList.Items {
		desiredReplicas := rc.Spec.Replicas + delta
		cnt := ensureReplicasAreInRange(desiredReplicas,
			policy.Spec.MinAutoScaleCount,
			policy.Spec.MaxAutoScaleCount)

		if cnt != rc.Spec.Replicas {
			glog.Infof("Updating RC %v for action %v from %v to %v [desired %v]",
				rc.Name, action.Type, rc.Spec.Replicas, cnt,
				desiredReplicas)
			rc.Spec.Replicas = cnt
			_, err := m.Client.ReplicationControllers(rc.Namespace).Update(&rc)
			if err != nil {
				allErrors = append(allErrors, err)
			} else {
				updatedList = append(updatedList, rc.Namespace+"/"+rc.Name)
				glog.Infof("Updated RC %v to %v for %v",
					rc.Name, rc.Spec.Replicas, action.Type)
			}
		}
	}

	status := api.AutoScalerStatus{
		LastActionTrigger:   action.Trigger,
		LastActionTimestamp: util.Now(),
	}

	status.LastActionTrigger.ActionType = action.Type
	status.LastActionTrigger.ScaleBy = action.ScaleBy

	policy.Status = status

	_, err = m.Client.AutoScalers(policy.Namespace).UpdateStatus(policy)

	if err != nil {
		allErrors = append(allErrors, err)
	}

	// TODO: log an event that the following replication controllers
	//       ${updatedList} were updated (${action.ActionType}) due to
	//       trigger ${policy.Status.LastActionTrigger}
	//       at ${policy.Status.LastActionTimestamp}.
	glog.Infof("Updated RC %+v - autoscaler status=%+v", updatedList, status)
	return errors.NewAggregate(allErrors)
}

// Ensures the replica count is within a specific range of values.
//   0 <= $min <= $replicas <= $max
func ensureReplicasAreInRange(replicas, minValue, maxValue int) int {
	// Ensure replicas is within Min/Max Scale range.
	if maxValue > 0 {
		if replicas > maxValue {
			replicas = maxValue
		}
	}

	if minValue >= 0 {
		if replicas < minValue {
			replicas = minValue
		}
	}

	// Replica count can't go below zero.
	if replicas < 0 {
		replicas = 0
	}

	return replicas
}
