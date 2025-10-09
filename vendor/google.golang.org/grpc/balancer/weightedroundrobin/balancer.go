/*
 *
 * Copyright 2023 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package weightedroundrobin provides an implementation of the weighted round
// robin LB policy, as defined in [gRFC A58].
//
// # Experimental
//
// Notice: This package is EXPERIMENTAL and may be changed or removed in a
// later release.
//
// [gRFC A58]: https://github.com/grpc/proposal/blob/master/A58-client-side-weighted-round-robin-lb-policy.md
package weightedroundrobin

import (
	"encoding/json"
	"fmt"
	rand "math/rand/v2"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/endpointsharding"
	"google.golang.org/grpc/balancer/pickfirst/pickfirstleaf"
	"google.golang.org/grpc/balancer/weightedroundrobin/internal"
	"google.golang.org/grpc/balancer/weightedtarget"
	"google.golang.org/grpc/connectivity"
	estats "google.golang.org/grpc/experimental/stats"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/grpcsync"
	iserviceconfig "google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/orca"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"

	v3orcapb "github.com/cncf/xds/go/xds/data/orca/v3"
)

// Name is the name of the weighted round robin balancer.
const Name = "weighted_round_robin"

var (
	rrFallbackMetric = estats.RegisterInt64Count(estats.MetricDescriptor{
		Name:           "grpc.lb.wrr.rr_fallback",
		Description:    "EXPERIMENTAL. Number of scheduler updates in which there were not enough endpoints with valid weight, which caused the WRR policy to fall back to RR behavior.",
		Unit:           "{update}",
		Labels:         []string{"grpc.target"},
		OptionalLabels: []string{"grpc.lb.locality"},
		Default:        false,
	})

	endpointWeightNotYetUsableMetric = estats.RegisterInt64Count(estats.MetricDescriptor{
		Name:           "grpc.lb.wrr.endpoint_weight_not_yet_usable",
		Description:    "EXPERIMENTAL. Number of endpoints from each scheduler update that don't yet have usable weight information (i.e., either the load report has not yet been received, or it is within the blackout period).",
		Unit:           "{endpoint}",
		Labels:         []string{"grpc.target"},
		OptionalLabels: []string{"grpc.lb.locality"},
		Default:        false,
	})

	endpointWeightStaleMetric = estats.RegisterInt64Count(estats.MetricDescriptor{
		Name:           "grpc.lb.wrr.endpoint_weight_stale",
		Description:    "EXPERIMENTAL. Number of endpoints from each scheduler update whose latest weight is older than the expiration period.",
		Unit:           "{endpoint}",
		Labels:         []string{"grpc.target"},
		OptionalLabels: []string{"grpc.lb.locality"},
		Default:        false,
	})
	endpointWeightsMetric = estats.RegisterFloat64Histo(estats.MetricDescriptor{
		Name:           "grpc.lb.wrr.endpoint_weights",
		Description:    "EXPERIMENTAL. Weight of each endpoint, recorded on every scheduler update. Endpoints without usable weights will be recorded as weight 0.",
		Unit:           "{endpoint}",
		Labels:         []string{"grpc.target"},
		OptionalLabels: []string{"grpc.lb.locality"},
		Default:        false,
	})
)

func init() {
	balancer.Register(bb{})
}

type bb struct{}

func (bb) Build(cc balancer.ClientConn, bOpts balancer.BuildOptions) balancer.Balancer {
	b := &wrrBalancer{
		ClientConn:       cc,
		target:           bOpts.Target.String(),
		metricsRecorder:  cc.MetricsRecorder(),
		addressWeights:   resolver.NewAddressMapV2[*endpointWeight](),
		endpointToWeight: resolver.NewEndpointMap[*endpointWeight](),
		scToWeight:       make(map[balancer.SubConn]*endpointWeight),
	}

	b.child = endpointsharding.NewBalancer(b, bOpts, balancer.Get(pickfirstleaf.Name).Build, endpointsharding.Options{})
	b.logger = prefixLogger(b)
	b.logger.Infof("Created")
	return b
}

func (bb) ParseConfig(js json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	lbCfg := &lbConfig{
		// Default values as documented in A58.
		OOBReportingPeriod:      iserviceconfig.Duration(10 * time.Second),
		BlackoutPeriod:          iserviceconfig.Duration(10 * time.Second),
		WeightExpirationPeriod:  iserviceconfig.Duration(3 * time.Minute),
		WeightUpdatePeriod:      iserviceconfig.Duration(time.Second),
		ErrorUtilizationPenalty: 1,
	}
	if err := json.Unmarshal(js, lbCfg); err != nil {
		return nil, fmt.Errorf("wrr: unable to unmarshal LB policy config: %s, error: %v", string(js), err)
	}

	if lbCfg.ErrorUtilizationPenalty < 0 {
		return nil, fmt.Errorf("wrr: errorUtilizationPenalty must be non-negative")
	}

	// For easier comparisons later, ensure the OOB reporting period is unset
	// (0s) when OOB reports are disabled.
	if !lbCfg.EnableOOBLoadReport {
		lbCfg.OOBReportingPeriod = 0
	}

	// Impose lower bound of 100ms on weightUpdatePeriod.
	if !internal.AllowAnyWeightUpdatePeriod && lbCfg.WeightUpdatePeriod < iserviceconfig.Duration(100*time.Millisecond) {
		lbCfg.WeightUpdatePeriod = iserviceconfig.Duration(100 * time.Millisecond)
	}

	return lbCfg, nil
}

func (bb) Name() string {
	return Name
}

// updateEndpointsLocked updates endpoint weight state based off new update, by
// starting and clearing any endpoint weights needed.
//
// Caller must hold b.mu.
func (b *wrrBalancer) updateEndpointsLocked(endpoints []resolver.Endpoint) {
	endpointSet := resolver.NewEndpointMap[*endpointWeight]()
	addressSet := resolver.NewAddressMapV2[*endpointWeight]()
	for _, endpoint := range endpoints {
		endpointSet.Set(endpoint, nil)
		for _, addr := range endpoint.Addresses {
			addressSet.Set(addr, nil)
		}
		ew, ok := b.endpointToWeight.Get(endpoint)
		if !ok {
			ew = &endpointWeight{
				logger:            b.logger,
				connectivityState: connectivity.Connecting,
				// Initially, we set load reports to off, because they are not
				// running upon initial endpointWeight creation.
				cfg:             &lbConfig{EnableOOBLoadReport: false},
				metricsRecorder: b.metricsRecorder,
				target:          b.target,
				locality:        b.locality,
			}
			for _, addr := range endpoint.Addresses {
				b.addressWeights.Set(addr, ew)
			}
			b.endpointToWeight.Set(endpoint, ew)
		}
		ew.updateConfig(b.cfg)
	}

	for _, endpoint := range b.endpointToWeight.Keys() {
		if _, ok := endpointSet.Get(endpoint); ok {
			// Existing endpoint also in new endpoint list; skip.
			continue
		}
		b.endpointToWeight.Delete(endpoint)
		for _, addr := range endpoint.Addresses {
			if _, ok := addressSet.Get(addr); !ok { // old endpoints to be deleted can share addresses with new endpoints, so only delete if necessary
				b.addressWeights.Delete(addr)
			}
		}
		// SubConn map will get handled in updateSubConnState
		// when receives SHUTDOWN signal.
	}
}

// wrrBalancer implements the weighted round robin LB policy.
type wrrBalancer struct {
	// The following fields are set at initialization time and read only after that,
	// so they do not need to be protected by a mutex.
	child               balancer.Balancer
	balancer.ClientConn // Embed to intercept NewSubConn operation
	logger              *grpclog.PrefixLogger
	target              string
	metricsRecorder     estats.MetricsRecorder

	mu               sync.Mutex
	cfg              *lbConfig // active config
	locality         string
	stopPicker       *grpcsync.Event
	addressWeights   *resolver.AddressMapV2[*endpointWeight]
	endpointToWeight *resolver.EndpointMap[*endpointWeight]
	scToWeight       map[balancer.SubConn]*endpointWeight
}

func (b *wrrBalancer) UpdateClientConnState(ccs balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("UpdateCCS: %v", ccs)
	}
	cfg, ok := ccs.BalancerConfig.(*lbConfig)
	if !ok {
		return fmt.Errorf("wrr: received nil or illegal BalancerConfig (type %T): %v", ccs.BalancerConfig, ccs.BalancerConfig)
	}

	// Note: empty endpoints and duplicate addresses across endpoints won't
	// explicitly error but will have undefined behavior.
	b.mu.Lock()
	b.cfg = cfg
	b.locality = weightedtarget.LocalityFromResolverState(ccs.ResolverState)
	b.updateEndpointsLocked(ccs.ResolverState.Endpoints)
	b.mu.Unlock()

	// This causes child to update picker inline and will thus cause inline
	// picker update.
	return b.child.UpdateClientConnState(balancer.ClientConnState{
		// Make pickfirst children use health listeners for outlier detection to
		// work.
		ResolverState: pickfirstleaf.EnableHealthListener(ccs.ResolverState),
	})
}

func (b *wrrBalancer) UpdateState(state balancer.State) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.stopPicker != nil {
		b.stopPicker.Fire()
		b.stopPicker = nil
	}

	childStates := endpointsharding.ChildStatesFromPicker(state.Picker)

	var readyPickersWeight []pickerWeightedEndpoint

	for _, childState := range childStates {
		if childState.State.ConnectivityState == connectivity.Ready {
			ew, ok := b.endpointToWeight.Get(childState.Endpoint)
			if !ok {
				// Should never happen, simply continue and ignore this endpoint
				// for READY pickers.
				continue
			}
			readyPickersWeight = append(readyPickersWeight, pickerWeightedEndpoint{
				picker:           childState.State.Picker,
				weightedEndpoint: ew,
			})
		}
	}
	// If no ready pickers are present, simply defer to the round robin picker
	// from endpoint sharding, which will round robin across the most relevant
	// pick first children in the highest precedence connectivity state.
	if len(readyPickersWeight) == 0 {
		b.ClientConn.UpdateState(balancer.State{
			ConnectivityState: state.ConnectivityState,
			Picker:            state.Picker,
		})
		return
	}

	p := &picker{
		v:               rand.Uint32(), // start the scheduler at a random point
		cfg:             b.cfg,
		weightedPickers: readyPickersWeight,
		metricsRecorder: b.metricsRecorder,
		locality:        b.locality,
		target:          b.target,
	}

	b.stopPicker = grpcsync.NewEvent()
	p.start(b.stopPicker)

	b.ClientConn.UpdateState(balancer.State{
		ConnectivityState: state.ConnectivityState,
		Picker:            p,
	})
}

type pickerWeightedEndpoint struct {
	picker           balancer.Picker
	weightedEndpoint *endpointWeight
}

func (b *wrrBalancer) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	addr := addrs[0] // The new pick first policy for DualStack will only ever create a SubConn with one address.
	var sc balancer.SubConn

	oldListener := opts.StateListener
	opts.StateListener = func(state balancer.SubConnState) {
		b.updateSubConnState(sc, state)
		oldListener(state)
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	ewi, ok := b.addressWeights.Get(addr)
	if !ok {
		// SubConn state updates can come in for a no longer relevant endpoint
		// weight (from the old system after a new config update is applied).
		return nil, fmt.Errorf("balancer is being closed; no new SubConns allowed")
	}
	sc, err := b.ClientConn.NewSubConn([]resolver.Address{addr}, opts)
	if err != nil {
		return nil, err
	}
	b.scToWeight[sc] = ewi
	return sc, nil
}

func (b *wrrBalancer) ResolverError(err error) {
	// Will cause inline picker update from endpoint sharding.
	b.child.ResolverError(err)
}

func (b *wrrBalancer) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, state)
}

func (b *wrrBalancer) updateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	b.mu.Lock()
	ew := b.scToWeight[sc]
	// updates from a no longer relevant SubConn update, nothing to do here but
	// forward state to state listener, which happens in wrapped listener. Will
	// eventually get cleared from scMap once receives Shutdown signal.
	if ew == nil {
		b.mu.Unlock()
		return
	}
	if state.ConnectivityState == connectivity.Shutdown {
		delete(b.scToWeight, sc)
	}
	b.mu.Unlock()

	// On the first READY SubConn/Transition for an endpoint, set pickedSC,
	// clear endpoint tracking weight state, and potentially start an OOB watch.
	if state.ConnectivityState == connectivity.Ready && ew.pickedSC == nil {
		ew.pickedSC = sc
		ew.mu.Lock()
		ew.nonEmptySince = time.Time{}
		ew.lastUpdated = time.Time{}
		cfg := ew.cfg
		ew.mu.Unlock()
		ew.updateORCAListener(cfg)
		return
	}

	// If the pickedSC (the one pick first uses for an endpoint) transitions out
	// of READY, stop OOB listener if needed and clear pickedSC so the next
	// created SubConn for the endpoint that goes READY will be chosen for
	// endpoint as the active SubConn.
	if state.ConnectivityState != connectivity.Ready && ew.pickedSC == sc {
		// The first SubConn that goes READY for an endpoint is what pick first
		// will pick. Only once that SubConn goes not ready will pick first
		// restart this cycle of creating SubConns and using the first READY
		// one. The lower level endpoint sharding will ping the Pick First once
		// this occurs to ExitIdle which will trigger a connection attempt.
		if ew.stopORCAListener != nil {
			ew.stopORCAListener()
		}
		ew.pickedSC = nil
	}
}

// Close stops the balancer.  It cancels any ongoing scheduler updates and
// stops any ORCA listeners.
func (b *wrrBalancer) Close() {
	b.mu.Lock()
	if b.stopPicker != nil {
		b.stopPicker.Fire()
		b.stopPicker = nil
	}
	b.mu.Unlock()

	// Ensure any lingering OOB watchers are stopped.
	for _, ew := range b.endpointToWeight.Values() {
		if ew.stopORCAListener != nil {
			ew.stopORCAListener()
		}
	}
	b.child.Close()
}

func (b *wrrBalancer) ExitIdle() {
	b.child.ExitIdle()
}

// picker is the WRR policy's picker.  It uses live-updating backend weights to
// update the scheduler periodically and ensure picks are routed proportional
// to those weights.
type picker struct {
	scheduler unsafe.Pointer // *scheduler; accessed atomically
	v         uint32         // incrementing value used by the scheduler; accessed atomically
	cfg       *lbConfig      // active config when picker created

	weightedPickers []pickerWeightedEndpoint // all READY pickers

	// The following fields are immutable.
	target          string
	locality        string
	metricsRecorder estats.MetricsRecorder
}

func (p *picker) endpointWeights(recordMetrics bool) []float64 {
	wp := make([]float64, len(p.weightedPickers))
	now := internal.TimeNow()
	for i, wpi := range p.weightedPickers {
		wp[i] = wpi.weightedEndpoint.weight(now, time.Duration(p.cfg.WeightExpirationPeriod), time.Duration(p.cfg.BlackoutPeriod), recordMetrics)
	}
	return wp
}

func (p *picker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	// Read the scheduler atomically.  All scheduler operations are threadsafe,
	// and if the scheduler is replaced during this usage, we want to use the
	// scheduler that was live when the pick started.
	sched := *(*scheduler)(atomic.LoadPointer(&p.scheduler))

	pickedPicker := p.weightedPickers[sched.nextIndex()]
	pr, err := pickedPicker.picker.Pick(info)
	if err != nil {
		logger.Errorf("ready picker returned error: %v", err)
		return balancer.PickResult{}, err
	}
	if !p.cfg.EnableOOBLoadReport {
		oldDone := pr.Done
		pr.Done = func(info balancer.DoneInfo) {
			if load, ok := info.ServerLoad.(*v3orcapb.OrcaLoadReport); ok && load != nil {
				pickedPicker.weightedEndpoint.OnLoadReport(load)
			}
			if oldDone != nil {
				oldDone(info)
			}
		}
	}
	return pr, nil
}

func (p *picker) inc() uint32 {
	return atomic.AddUint32(&p.v, 1)
}

func (p *picker) regenerateScheduler() {
	s := p.newScheduler(true)
	atomic.StorePointer(&p.scheduler, unsafe.Pointer(&s))
}

func (p *picker) start(stopPicker *grpcsync.Event) {
	p.regenerateScheduler()
	if len(p.weightedPickers) == 1 {
		// No need to regenerate weights with only one backend.
		return
	}

	go func() {
		ticker := time.NewTicker(time.Duration(p.cfg.WeightUpdatePeriod))
		defer ticker.Stop()
		for {
			select {
			case <-stopPicker.Done():
				return
			case <-ticker.C:
				p.regenerateScheduler()
			}
		}
	}()
}

// endpointWeight is the weight for an endpoint. It tracks the SubConn that will
// be picked for the endpoint, and other parameters relevant to computing the
// effective weight. When needed, it also tracks connectivity state, listens for
// metrics updates by implementing the orca.OOBListener interface and manages
// that listener.
type endpointWeight struct {
	// The following fields are immutable.
	logger          *grpclog.PrefixLogger
	target          string
	metricsRecorder estats.MetricsRecorder
	locality        string

	// The following fields are only accessed on calls into the LB policy, and
	// do not need a mutex.
	connectivityState connectivity.State
	stopORCAListener  func()
	// The first SubConn for the endpoint that goes READY when endpoint has no
	// READY SubConns yet, cleared on that sc disconnecting (i.e. going out of
	// READY). Represents what pick first will use as it's picked SubConn for
	// this endpoint.
	pickedSC balancer.SubConn

	// The following fields are accessed asynchronously and are protected by
	// mu.  Note that mu may not be held when calling into the stopORCAListener
	// or when registering a new listener, as those calls require the ORCA
	// producer mu which is held when calling the listener, and the listener
	// holds mu.
	mu            sync.Mutex
	weightVal     float64
	nonEmptySince time.Time
	lastUpdated   time.Time
	cfg           *lbConfig
}

func (w *endpointWeight) OnLoadReport(load *v3orcapb.OrcaLoadReport) {
	if w.logger.V(2) {
		w.logger.Infof("Received load report for subchannel %v: %v", w.pickedSC, load)
	}
	// Update weights of this endpoint according to the reported load.
	utilization := load.ApplicationUtilization
	if utilization == 0 {
		utilization = load.CpuUtilization
	}
	if utilization == 0 || load.RpsFractional == 0 {
		if w.logger.V(2) {
			w.logger.Infof("Ignoring empty load report for subchannel %v", w.pickedSC)
		}
		return
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	errorRate := load.Eps / load.RpsFractional
	w.weightVal = load.RpsFractional / (utilization + errorRate*w.cfg.ErrorUtilizationPenalty)
	if w.logger.V(2) {
		w.logger.Infof("New weight for subchannel %v: %v", w.pickedSC, w.weightVal)
	}

	w.lastUpdated = internal.TimeNow()
	if w.nonEmptySince.Equal(time.Time{}) {
		w.nonEmptySince = w.lastUpdated
	}
}

// updateConfig updates the parameters of the WRR policy and
// stops/starts/restarts the ORCA OOB listener.
func (w *endpointWeight) updateConfig(cfg *lbConfig) {
	w.mu.Lock()
	oldCfg := w.cfg
	w.cfg = cfg
	w.mu.Unlock()

	if cfg.EnableOOBLoadReport == oldCfg.EnableOOBLoadReport &&
		cfg.OOBReportingPeriod == oldCfg.OOBReportingPeriod {
		// Load reporting wasn't enabled before or after, or load reporting was
		// enabled before and after, and had the same period.  (Note that with
		// load reporting disabled, OOBReportingPeriod is always 0.)
		return
	}
	// (Re)start the listener to use the new config's settings for OOB
	// reporting.
	w.updateORCAListener(cfg)
}

func (w *endpointWeight) updateORCAListener(cfg *lbConfig) {
	if w.stopORCAListener != nil {
		w.stopORCAListener()
	}
	if !cfg.EnableOOBLoadReport {
		w.stopORCAListener = nil
		return
	}
	if w.pickedSC == nil { // No picked SC for this endpoint yet, nothing to listen on.
		return
	}
	if w.logger.V(2) {
		w.logger.Infof("Registering ORCA listener for %v with interval %v", w.pickedSC, cfg.OOBReportingPeriod)
	}
	opts := orca.OOBListenerOptions{ReportInterval: time.Duration(cfg.OOBReportingPeriod)}
	w.stopORCAListener = orca.RegisterOOBListener(w.pickedSC, w, opts)
}

// weight returns the current effective weight of the endpoint, taking into
// account the parameters.  Returns 0 for blacked out or expired data, which
// will cause the backend weight to be treated as the mean of the weights of the
// other backends. If forScheduler is set to true, this function will emit
// metrics through the metrics registry.
func (w *endpointWeight) weight(now time.Time, weightExpirationPeriod, blackoutPeriod time.Duration, recordMetrics bool) (weight float64) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if recordMetrics {
		defer func() {
			endpointWeightsMetric.Record(w.metricsRecorder, weight, w.target, w.locality)
		}()
	}

	// The endpoint has not received a load report (i.e. just turned READY with
	// no load report).
	if w.lastUpdated.Equal(time.Time{}) {
		endpointWeightNotYetUsableMetric.Record(w.metricsRecorder, 1, w.target, w.locality)
		return 0
	}

	// If the most recent update was longer ago than the expiration period,
	// reset nonEmptySince so that we apply the blackout period again if we
	// start getting data again in the future, and return 0.
	if now.Sub(w.lastUpdated) >= weightExpirationPeriod {
		if recordMetrics {
			endpointWeightStaleMetric.Record(w.metricsRecorder, 1, w.target, w.locality)
		}
		w.nonEmptySince = time.Time{}
		return 0
	}

	// If we don't have at least blackoutPeriod worth of data, return 0.
	if blackoutPeriod != 0 && (w.nonEmptySince.Equal(time.Time{}) || now.Sub(w.nonEmptySince) < blackoutPeriod) {
		if recordMetrics {
			endpointWeightNotYetUsableMetric.Record(w.metricsRecorder, 1, w.target, w.locality)
		}
		return 0
	}

	return w.weightVal
}
