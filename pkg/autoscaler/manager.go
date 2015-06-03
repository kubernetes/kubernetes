package autoscaler

import (
	"fmt"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/golang/glog"
)

// How often auto scale policies are assessed and applied.
const DefaultAssessmentInterval = 30 * time.Second
const DefaultAutoScalingInterval = 60 * time.Second

// AutoScaleManager tracks and manages registered plugins and sources.
type AutoScaleManager struct {
	Client   client.Interface //  Kube client
	Interval time.Duration    //  How often to assess policies.

	mutex  sync.Mutex   // Lock for controlling exclusive access.
	ticker *time.Ticker // Ticker used for assessment interval.

	plugins map[string]AutoScalerPlugin // Autoscaler plugins.
	sources map[string]MonitoringSource // Monitoring sources.
}

// Create a new auto scaler manager instance.
func NewAutoScaleManager(kubeClient client.Interface) *AutoScaleManager {
	return &AutoScaleManager{
		Client:   kubeClient,
		Interval: DefaultAssessmentInterval,
	}
}

// Register an auto scaler plugin with the manager.
func (m *AutoScaleManager) Register(plugin AutoScalerPlugin) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if nil == m.plugins {
		m.plugins = make(map[string]AutoScalerPlugin, 0)
	}

	name := plugin.Name()

	if _, registered := m.plugins[name]; registered {
		return fmt.Errorf("Autoscaler plugin %q was registered more than once", name)
	}

	m.plugins[name] = plugin
	return nil
}

// {Un,De}registers an auto scaler plugin with the manager.
func (m *AutoScaleManager) Deregister(name string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if nil == m.plugins {
		return fmt.Errorf("no Autoscaler plugin %q was registered", name)
	}

	if _, registered := m.plugins[name]; !registered {
		return fmt.Errorf("no Autoscaler plugin %q was registered", name)
	}

	delete(m.plugins, name)
	return nil
}

// FindPluginByName finds an autoscaler plugin by name. If no plugin is
// found, returns an error.
func (m *AutoScaleManager) FindPluginByName(name string) (*AutoScalerPlugin, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	for _, v := range m.plugins {
		if v.Name() == name {
			return &v, nil
		}
	}

	return nil, fmt.Errorf("no autoscaler plugin matched name %q", name)
}

// Adds a monitoring source.
func (m *AutoScaleManager) AddMonitoringSource(source MonitoringSource) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if nil == m.sources {
		m.sources = make(map[string]MonitoringSource, 0)
	}

	name := source.Name()

	if _, exists := m.sources[name]; exists {
		return fmt.Errorf("MonitoringSource %q added more than once", name)
	}

	m.sources[name] = source
	return nil
}

// Removes a monitoring source.
func (m *AutoScaleManager) RemoveMonitoringSource(name string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if nil == m.sources {
		return fmt.Errorf("no MonitoringSource %q exists", name)
	}

	if _, exists := m.sources[name]; !exists {
		return fmt.Errorf("no MonitoringSoure %q was exists", name)
	}

	delete(m.sources, name)
	return nil
}

// Returns a list of all the monitoring sources the manager is using.
func (m *AutoScaleManager) GetMonitoringSources() []*MonitoringSource {
	sources := make([]*MonitoringSource, len(m.sources))

	m.mutex.Lock()
	defer m.mutex.Unlock()

	idx := 0
	for _, src := range m.sources {
		sources[idx] = &src
		idx += 1
	}

	return sources
}

// Process all the auto scalers (all the auto scale policies).
func (m *AutoScaleManager) ProcessAutoScalePolicies() {
	sources := m.GetMonitoringSources()
	autoscalers, err := m.Client.AutoScalers(api.NamespaceAll).List(labels.Everything(), fields.Everything())
	if err != nil {
		// TODO: Generate an event in lieu of a log message.
		glog.Error(err)
		return
	}

	for _, policy := range autoscalers.Items {
		actions, err := m.assessAutoScaler(policy, sources)

		// Log error & process actions (speaks louder than words)!!
		if err != nil {
			// TODO: Generate an event in lieu of a log message.
			glog.Error(err)
		}

		if len(actions) > 0 {
			err := m.processAutoScalerActions(&policy, actions)
			if err != nil {
				// TODO: Generate an event in lieu of a log message.
				glog.Error(err)
			}
		}
	}
}

// Start the autoscaler manager processing loop (runs till doomsday).
func (m *AutoScaleManager) Run() error {
	if m.ticker != nil {
		return fmt.Errorf("autoscaler manager is already running")
	}

	m.ticker = time.NewTicker(m.Interval)
	defer m.ticker.Stop()

	for {
		select {
		case <-m.ticker.C:
			m.ProcessAutoScalePolicies()
		}
	}

	return nil
}

// Stop the autoscaler manager processing loop.
func (m *AutoScaleManager) Stop() error {
	if nil == m.ticker {
		return fmt.Errorf("autoscaler manager has already been stopped")
	}

	m.ticker.Stop()
	m.ticker = nil
	return nil
}

// Assesses the auto scaler to get a list of desired actions.
func (m *AutoScaleManager) assessAutoScaler(autoscaler api.AutoScaler, sources []*MonitoringSource) ([]AutoScaleAction, error) {
	allActions := make([]AutoScaleAction, 0)
	allErrors := make([]error, 0)

	// Check if policy was recently assessed and if so, its a NOOP here.
	ts := autoscaler.Status.LastActionTimestamp.Add(DefaultAutoScalingInterval)
	scaleAt := util.Time{ts}
	if util.Now().Before(scaleAt) {
		return allActions, errors.NewAggregate(allErrors)
	}

	for _, plugin := range m.plugins {
		actions, err := plugin.Assess(autoscaler.Spec, sources)
		if err != nil {
			allErrors = append(allErrors, err)
			continue
		}

		for _, action := range actions {
			allActions = append(allActions, action)
		}
	}

	return allActions, errors.NewAggregate(allErrors)
}

// Process the actions to take for the auto scaler.
func (m *AutoScaleManager) processAutoScalerActions(autoscaler *api.AutoScaler, actions []AutoScaleAction) error {
	allErrors := make([]error, 0)
	delta := 0

	action := ReconcileActions(actions)
	switch action.ScaleType {
	case api.AutoScaleActionTypeNone:
		return errors.NewAggregate(allErrors)

	case api.AutoScaleActionTypeScaleUp:
		delta = action.ScaleBy

	case api.AutoScaleActionTypeScaleDown:
		delta = 0 - action.ScaleBy
	}

	updatedList := make([]string, 0)

	selector := labels.SelectorFromSet(autoscaler.Spec.TargetSelector)
	rcList, err := m.Client.ReplicationControllers(api.NamespaceAll).List(selector)
	for _, rc := range rcList.Items {
		cnt := ensureReplicasAreInRange(rc.Spec.Replicas + delta,
			autoscaler.Spec.MaxAutoScaleCount,
			autoscaler.Spec.MinAutoScaleCount)
		if cnt != rc.Spec.Replicas {
			rc.Spec.Replicas = cnt
			_, err := m.Client.ReplicationControllers(rc.Namespace).Update(&rc)
			if err != nil {
				allErrors = append(allErrors, err)
			} else {
				updatedList = append(updatedList, rc.Namespace + "/" + rc.Name)
			}
		}
	}

	autoscaler.Status.LastActionTrigger = *action.Trigger
	autoscaler.Status.LastActionTimestamp = util.Now()

	_, err = m.Client.AutoScalers(autoscaler.Namespace).UpdateStatus(autoscaler)

	if err != nil {
		allErrors = append(allErrors, err)
	}

	// TODO: log an event that the following replication controllers
	//       ${updatedList} were updated (${action.ActionType}) due to
	//       trigger ${autoscaler.Status.LastActionTrigger}
	//       at ${autoscaler.Status.LastActionTimestamp}.

	return errors.NewAggregate(allErrors)
}

func ensureReplicasAreInRange(replicas int, minValue, maxValue int) int {
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
