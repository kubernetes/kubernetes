package health

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
	"k8s.io/klog/v2"
)

var (
	defaultProbeResponseTimeout = 1 * time.Second
	defaultProbeInterval        = 2 * time.Second

	defaultUnhealthyProbesThreshold = 3
	defaultHealthyProbesThreshold   = 5
)

type Prober struct {
	// targetProvider provides a list of targets to monitor
	// it also can schedule refreshing the list by simply calling Enqueue method
	targetProvider TargetProvider

	// client is an HTTP client that is used to probe health checks for targets
	client *http.Client

	// probeInterval specifies a time interval at which health checks are send
	probeInterval time.Duration

	// unhealthyProbesThreshold specifies consecutive failed health checks after which a target is considered unhealthy
	unhealthyProbesThreshold int

	// healthyProbesThreshold  specifies consecutive successful health checks after which a target is considered healthy
	healthyProbesThreshold int

	healthyTargets   []string
	unhealthyTargets []string
	targetsToMonitor []string

	consecutiveSuccessfulProbes map[string]int
	consecutiveFailedProbes     map[string][]error

	refreshTargetsLock sync.Mutex
	refreshTargets     bool

	// exportedHealthyTargets holds a copy of healthyTargets
	exportedHealthyTargets atomic.Value

	// exportedUnhealthyTargets holds a copy of unhealthyTargets
	exportedUnhealthyTargets atomic.Value

	// listeners holds a list of interested parties to be notified when the list of healthy targets changes
	listeners []Listener

	// metrics specifies a set of methods that are used to register various metrics
	metrics *Metrics
}

var _ Listener = &Prober{}
var _ Notifier = &Prober{}

// New creates a health monitor that periodically sends requests to the provided targets to check their health.
//
// The following methods allows you to configure behaviour of the monitor after creation.
//
//	WithUnHealthyProbesThreshold - that specifies consecutive failed health checks after which a target is considered unhealthy
//	                               the default value is: 3
//
//	WithHealthyProbesThreshold   - that specifies consecutive successful health checks after which a target is considered healthy
//	                               the default value is: 5
//
//	WithProbeResponseTimeout     - that specifies a time limit for requests made by the HTTP client for the health check
//	                               the default value is: 1 second
//
//	WithProbeInterval            - that specifies a time interval at which health checks are send
//	                               the default value is: 2 seconds
//
//	WithMetrics                  - that specifies a set of methods that are used to register various metrics
//	                               the default value is: no metrics
//
// Additionally the monitor implements Listener and Notifier interfaces.
//
// The health monitor automatically registers for notification if the target provided also implements the Notifier interface.
// It is implicit so that the provider can provide a static or a dynamic list of targets.
//
// Interested parties can register a listener for notifications about healthy/unhealthy targets changes via AddListener.
// TODO: instead of restConfig we could accept transport so that it is reused instead of creating a new connection to targets
//
//	reusing the transport has the advantage of using the same connection as other clients
func New(targetProvider TargetProvider, restConfig *rest.Config) (*Prober, error) {
	client, err := createHealthCheckHTTPClient(defaultProbeResponseTimeout, restConfig)
	if err != nil {
		return nil, err
	}

	hm := &Prober{
		client:                   client,
		targetProvider:           targetProvider,
		targetsToMonitor:         targetProvider.CurrentTargetsList(),
		probeInterval:            defaultProbeInterval,
		unhealthyProbesThreshold: defaultUnhealthyProbesThreshold,
		healthyProbesThreshold:   defaultHealthyProbesThreshold,

		consecutiveSuccessfulProbes: map[string]int{},
		consecutiveFailedProbes:     map[string][]error{},

		metrics: &Metrics{
			HealthyTargetsTotal:        noopMetrics{}.TargetsTotal,
			CurrentHealthyTargets:      noopMetrics{}.TargetsGauge,
			UnHealthyTargetsTotal:      noopMetrics{}.TargetsTotal,
			ReadyzProtocolRequestTotal: noopMetrics{}.TargetsWithCodeTotal,
		},
	}
	hm.exportedHealthyTargets.Store([]string{})
	hm.exportedUnhealthyTargets.Store([]string{})

	if notifier, ok := targetProvider.(Notifier); ok {
		notifier.AddListener(hm)
	}

	return hm, nil
}

// Run starts monitoring the provided targets until stop channel is closed
// This method is blocking and it is meant to be launched in a separate goroutine
func (sm *Prober) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	klog.Infof("Starting the health monitor with Interval = %v, Timeout = %v, HealthyThreshold = %v, UnhealthyThreshold = %v ", sm.probeInterval, sm.client.Timeout, sm.healthyProbesThreshold, sm.unhealthyProbesThreshold)
	defer klog.Info("Shutting down the health monitor")

	wait.Until(sm.healthCheckRegisteredTargets, sm.probeInterval, ctx.Done())
}

// Enqueue schedules refreshing the target list on the next probeInterval
// This method is used by the TargetProvider to notify that the list has changed
func (sm *Prober) Enqueue() {
	sm.refreshTargetsLock.Lock()
	defer sm.refreshTargetsLock.Unlock()
	sm.refreshTargets = true
}

// Targets returns a list of healthy and unhealthy targets
func (sm *Prober) Targets() ([]string, []string) {
	return sm.exportedHealthyTargets.Load().([]string), sm.exportedUnhealthyTargets.Load().([]string)
}

// AddListener adds a listener to be notified when the list of healthy targets changes
//
// Note:
// this method is not thread safe and mustn't be called after calling StartMonitoring() method
func (sm *Prober) AddListener(listener Listener) {
	sm.listeners = append(sm.listeners, listener)
}

type targetErrTuple struct {
	target string
	err    error
}

// refreshTargetsLocked updates the internal targets list to monitor if it was requested (via the Enqueue method)
func (sm *Prober) refreshTargetsLocked() {
	sm.refreshTargetsLock.Lock()
	defer sm.refreshTargetsLock.Unlock()
	if !sm.refreshTargets {
		return
	}

	sm.refreshTargets = false
	freshTargets := sm.targetProvider.CurrentTargetsList()
	freshTargetSet := sets.New(freshTargets...)

	currentTargetsSet := sets.New(sm.targetsToMonitor...)
	newTargetsToMonitorSet := freshTargetSet.Difference(currentTargetsSet)
	if newTargetsToMonitorSet.Len() > 0 {
		klog.V(2).Infof("health monitor observed new targets = %v", sets.List(newTargetsToMonitorSet))
	}

	removedTargetsToMonitorSet := currentTargetsSet.Difference(freshTargetSet)
	if removedTargetsToMonitorSet.Len() > 0 {
		klog.V(2).Infof("health monitor will stop checking the following targets targets = %v", sets.List(removedTargetsToMonitorSet))
		for targetToRemove := range removedTargetsToMonitorSet {
			delete(sm.consecutiveSuccessfulProbes, targetToRemove)
			delete(sm.consecutiveFailedProbes, targetToRemove)
		}

		healthyTargetsSet := sets.New(sm.healthyTargets...)
		healthyTargetsSet.Delete(removedTargetsToMonitorSet.UnsortedList()...)
		sm.healthyTargets = sets.List(healthyTargetsSet)

		unhealthyTargetsSet := sets.New(sm.unhealthyTargets...)
		unhealthyTargetsSet.Delete(removedTargetsToMonitorSet.UnsortedList()...)
		sm.unhealthyTargets = sets.List(unhealthyTargetsSet)
	}

	sm.targetsToMonitor = freshTargets
}

func (sm *Prober) healthCheckRegisteredTargets() {
	sm.refreshTargetsLocked()
	var wg sync.WaitGroup
	resTargetErrTupleCh := make(chan targetErrTuple, len(sm.targetsToMonitor))

	for i := 0; i < len(sm.targetsToMonitor); i++ {
		wg.Add(1)
		go func(target string) {
			defer wg.Done()
			err := sm.healthCheckSingleTarget(target)
			resTargetErrTupleCh <- targetErrTuple{target, err}
		}(sm.targetsToMonitor[i])
	}
	wg.Wait()
	close(resTargetErrTupleCh)

	currentHealthCheckProbes := make([]targetErrTuple, 0, len(sm.targetsToMonitor))
	for svrErrTuple := range resTargetErrTupleCh {
		currentHealthCheckProbes = append(currentHealthCheckProbes, svrErrTuple)
	}

	sm.updateHealthChecksFor(currentHealthCheckProbes)
}

// updateHealthChecksFor examines the health of targets based on the provided probes and the current configuration.
// It also notifies interested parties about changes in the health condition.
// Interested parties can be registered by calling AddListener method.
func (sm *Prober) updateHealthChecksFor(currentHealthCheckProbes []targetErrTuple) {
	newUnhealthyTargets := []string{}
	newHealthyTargets := []string{}

	for _, svrErrTuple := range currentHealthCheckProbes {
		if svrErrTuple.err != nil {
			delete(sm.consecutiveSuccessfulProbes, svrErrTuple.target)

			unhealthyProbesSlice := sm.consecutiveFailedProbes[svrErrTuple.target]
			if len(unhealthyProbesSlice) < sm.unhealthyProbesThreshold {
				unhealthyProbesSlice = append(unhealthyProbesSlice, svrErrTuple.err)
				sm.consecutiveFailedProbes[svrErrTuple.target] = unhealthyProbesSlice
				if len(unhealthyProbesSlice) == sm.unhealthyProbesThreshold {
					newUnhealthyTargets = append(newUnhealthyTargets, svrErrTuple.target)
				}
			}
			continue
		}

		delete(sm.consecutiveFailedProbes, svrErrTuple.target)

		healthyProbesCounter := sm.consecutiveSuccessfulProbes[svrErrTuple.target]
		if healthyProbesCounter < sm.healthyProbesThreshold {
			healthyProbesCounter++
			sm.consecutiveSuccessfulProbes[svrErrTuple.target] = healthyProbesCounter
			if healthyProbesCounter == sm.healthyProbesThreshold {
				newHealthyTargets = append(newHealthyTargets, svrErrTuple.target)
			}
		}
	}

	newUnhealthyTargetsSet := sets.New(newUnhealthyTargets...)
	newHealthyTargetsSet := sets.New(newHealthyTargets...)
	notifyListeners := false

	// detect unhealthy targets
	previouslyUnhealthyTargetsSet := sets.New(sm.unhealthyTargets...)
	currentlyUnhealthyTargetsSet := previouslyUnhealthyTargetsSet.Union(newUnhealthyTargetsSet)
	currentlyUnhealthyTargetsSet.Delete(newHealthyTargetsSet.UnsortedList()...)
	if !currentlyUnhealthyTargetsSet.Equal(previouslyUnhealthyTargetsSet) {
		sm.unhealthyTargets = sets.List(currentlyUnhealthyTargetsSet)
		klog.V(2).Infof("observed the following unhealthy targets %v", sm.unhealthyTargets)
		logUnhealthyTargets(sm.unhealthyTargets, currentHealthCheckProbes)

		exportedUnhealthyTargets := make([]string, len(sm.unhealthyTargets))
		for index, unhealthyTarget := range sm.unhealthyTargets {
			exportedUnhealthyTargets[index] = unhealthyTarget
			sm.metrics.UnHealthyTargetsTotal(unhealthyTarget)
		}
		sm.exportedUnhealthyTargets.Store(exportedUnhealthyTargets)
		notifyListeners = true
	}

	// detect healthy targets
	previouslyHealthyTargetsSet := sets.New(sm.healthyTargets...)
	currentlyHealthyTargetsSet := previouslyHealthyTargetsSet.Union(newHealthyTargetsSet)
	currentlyHealthyTargetsSet.Delete(newUnhealthyTargetsSet.UnsortedList()...)
	if !currentlyHealthyTargetsSet.Equal(previouslyHealthyTargetsSet) {
		sm.healthyTargets = sets.List(currentlyHealthyTargetsSet)
		klog.V(2).Infof("observed the following healthy targets %v", sm.healthyTargets)

		exportedHealthyTargets := make([]string, len(sm.healthyTargets))
		for index, healthyTarget := range sm.healthyTargets {
			exportedHealthyTargets[index] = healthyTarget
			sm.metrics.HealthyTargetsTotal(healthyTarget)
		}
		sm.exportedHealthyTargets.Store(exportedHealthyTargets)
		notifyListeners = true
	}

	if notifyListeners {
		// something has changed update the currently healthy targets metric
		sm.metrics.CurrentHealthyTargets(float64(len(sm.healthyTargets)))

		// notify listeners about the new healthy/unhealthy targets
		for _, listener := range sm.listeners {
			listener.Enqueue()
		}
	}
}

func (sm *Prober) healthCheckSingleTarget(target string) error {
	// TODO: make the protocol, port and the path configurable
	targetURL, err := url.Parse(fmt.Sprintf("https://%s/%s", target, "readyz"))
	if err != nil {
		return err
	}
	newReq, err := http.NewRequest("GET", targetURL.String(), nil)
	if err != nil {
		return err
	}

	resp, err := sm.client.Do(newReq)
	if err != nil {
		sm.metrics.ReadyzProtocolRequestTotal("<error>", target)
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode != http.StatusInternalServerError {
			sm.metrics.ReadyzProtocolRequestTotal(strconv.Itoa(resp.StatusCode), target)
		}
		return fmt.Errorf("bad status from %v: %v, expected HTTP 200", targetURL.String(), resp.StatusCode)
	}

	return err
}

func createHealthCheckHTTPClient(responseTimeout time.Duration, restConfig *rest.Config) (*http.Client, error) {
	transportConfig, err := restConfig.TransportConfig()
	if err != nil {
		return nil, err
	}

	tlsConfig, err := transport.TLSConfigFor(transportConfig)
	if err != nil {
		return nil, err
	}

	client := &http.Client{
		Transport: utilnet.SetTransportDefaults(&http.Transport{
			TLSClientConfig: tlsConfig,
		}),
		Timeout: responseTimeout,
	}

	return client, nil
}

func logUnhealthyTargets(unhealthyTargets []string, currentHealthCheckProbes []targetErrTuple) {
	for _, unhealthyTarget := range unhealthyTargets {
		errorsForUnhealthyTarget := []error{}
		for _, svrErrTuple := range currentHealthCheckProbes {
			if svrErrTuple.target == unhealthyTarget {
				errorsForUnhealthyTarget = append(errorsForUnhealthyTarget, svrErrTuple.err)
			}
		}
		klog.V(2).Infof("the following target %v became unhealthy due to %v", unhealthyTarget, utilerrors.NewAggregate(errorsForUnhealthyTarget).Error())
	}
}
