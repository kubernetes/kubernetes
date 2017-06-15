// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package jaeger

import (
	"fmt"
	"math"
	"net/url"
	"sync"
	"time"

	"github.com/uber/jaeger-client-go/log"
	"github.com/uber/jaeger-client-go/thrift-gen/sampling"
	"github.com/uber/jaeger-client-go/utils"
)

const (
	defaultSamplingServerURL       = "http://localhost:5778/sampling"
	defaultSamplingRefreshInterval = time.Minute
	defaultMaxOperations           = 2000
)

// Sampler decides whether a new trace should be sampled or not.
type Sampler interface {
	// IsSampled decides whether a trace with given `id` and `operation`
	// should be sampled. This function will also return the tags that
	// can be used to identify the type of sampling that was applied to
	// the root span. Most simple samplers would return two tags,
	// sampler.type and sampler.param, similar to those used in the Configuration
	IsSampled(id TraceID, operation string) (sampled bool, tags []Tag)

	// Close does a clean shutdown of the sampler, stopping any background
	// go-routines it may have started.
	Close()

	// Equal checks if the `other` sampler is functionally equivalent
	// to this sampler.
	// TODO remove this function. This function is used to determine if 2 samplers are equivalent
	// which does not bode well with the adaptive sampler which has to create all the composite samplers
	// for the comparison to occur. This is expensive to do if only one sampler has changed.
	Equal(other Sampler) bool
}

// -----------------------

// ConstSampler is a sampler that always makes the same decision.
type ConstSampler struct {
	Decision bool
	tags     []Tag
}

// NewConstSampler creates a ConstSampler.
func NewConstSampler(sample bool) Sampler {
	tags := []Tag{
		{key: SamplerTypeTagKey, value: SamplerTypeConst},
		{key: SamplerParamTagKey, value: sample},
	}
	return &ConstSampler{Decision: sample, tags: tags}
}

// IsSampled implements IsSampled() of Sampler.
func (s *ConstSampler) IsSampled(id TraceID, operation string) (bool, []Tag) {
	return s.Decision, s.tags
}

// Close implements Close() of Sampler.
func (s *ConstSampler) Close() {
	// nothing to do
}

// Equal implements Equal() of Sampler.
func (s *ConstSampler) Equal(other Sampler) bool {
	if o, ok := other.(*ConstSampler); ok {
		return s.Decision == o.Decision
	}
	return false
}

// -----------------------

// ProbabilisticSampler is a sampler that randomly samples a certain percentage
// of traces.
type ProbabilisticSampler struct {
	samplingRate     float64
	samplingBoundary uint64
	tags             []Tag
}

const maxRandomNumber = ^(uint64(1) << 63) // i.e. 0x7fffffffffffffff

// NewProbabilisticSampler creates a sampler that randomly samples a certain percentage of traces specified by the
// samplingRate, in the range between 0.0 and 1.0.
//
// It relies on the fact that new trace IDs are 63bit random numbers themselves, thus making the sampling decision
// without generating a new random number, but simply calculating if traceID < (samplingRate * 2^63).
// TODO remove the error from this function for next major release
func NewProbabilisticSampler(samplingRate float64) (*ProbabilisticSampler, error) {
	if samplingRate < 0.0 || samplingRate > 1.0 {
		return nil, fmt.Errorf("Sampling Rate must be between 0.0 and 1.0, received %f", samplingRate)
	}
	return newProbabilisticSampler(samplingRate), nil
}

func newProbabilisticSampler(samplingRate float64) *ProbabilisticSampler {
	samplingRate = math.Max(0.0, math.Min(samplingRate, 1.0))
	tags := []Tag{
		{key: SamplerTypeTagKey, value: SamplerTypeProbabilistic},
		{key: SamplerParamTagKey, value: samplingRate},
	}
	return &ProbabilisticSampler{
		samplingRate:     samplingRate,
		samplingBoundary: uint64(float64(maxRandomNumber) * samplingRate),
		tags:             tags,
	}
}

// SamplingRate returns the sampling probability this sampled was constructed with.
func (s *ProbabilisticSampler) SamplingRate() float64 {
	return s.samplingRate
}

// IsSampled implements IsSampled() of Sampler.
func (s *ProbabilisticSampler) IsSampled(id TraceID, operation string) (bool, []Tag) {
	return s.samplingBoundary >= id.Low, s.tags
}

// Close implements Close() of Sampler.
func (s *ProbabilisticSampler) Close() {
	// nothing to do
}

// Equal implements Equal() of Sampler.
func (s *ProbabilisticSampler) Equal(other Sampler) bool {
	if o, ok := other.(*ProbabilisticSampler); ok {
		return s.samplingBoundary == o.samplingBoundary
	}
	return false
}

// -----------------------

type rateLimitingSampler struct {
	maxTracesPerSecond float64
	rateLimiter        utils.RateLimiter
	tags               []Tag
}

// NewRateLimitingSampler creates a sampler that samples at most maxTracesPerSecond. The distribution of sampled
// traces follows burstiness of the service, i.e. a service with uniformly distributed requests will have those
// requests sampled uniformly as well, but if requests are bursty, especially sub-second, then a number of
// sequential requests can be sampled each second.
func NewRateLimitingSampler(maxTracesPerSecond float64) Sampler {
	tags := []Tag{
		{key: SamplerTypeTagKey, value: SamplerTypeRateLimiting},
		{key: SamplerParamTagKey, value: maxTracesPerSecond},
	}
	return &rateLimitingSampler{
		maxTracesPerSecond: maxTracesPerSecond,
		rateLimiter:        utils.NewRateLimiter(maxTracesPerSecond, math.Max(maxTracesPerSecond, 1.0)),
		tags:               tags,
	}
}

// IsSampled implements IsSampled() of Sampler.
func (s *rateLimitingSampler) IsSampled(id TraceID, operation string) (bool, []Tag) {
	return s.rateLimiter.CheckCredit(1.0), s.tags
}

func (s *rateLimitingSampler) Close() {
	// nothing to do
}

func (s *rateLimitingSampler) Equal(other Sampler) bool {
	if o, ok := other.(*rateLimitingSampler); ok {
		return s.maxTracesPerSecond == o.maxTracesPerSecond
	}
	return false
}

// -----------------------

// GuaranteedThroughputProbabilisticSampler is a sampler that leverages both probabilisticSampler and
// rateLimitingSampler. The rateLimitingSampler is used as a guaranteed lower bound sampler such that
// every operation is sampled at least once in a time interval defined by the lowerBound. ie a lowerBound
// of 1.0 / (60 * 10) will sample an operation at least once every 10 minutes.
//
// The probabilisticSampler is given higher priority when tags are emitted, ie. if IsSampled() for both
// samplers return true, the tags for probabilisticSampler will be used.
type GuaranteedThroughputProbabilisticSampler struct {
	probabilisticSampler *ProbabilisticSampler
	lowerBoundSampler    Sampler
	tags                 []Tag
	samplingRate         float64
	lowerBound           float64
}

// NewGuaranteedThroughputProbabilisticSampler returns a delegating sampler that applies both
// probabilisticSampler and rateLimitingSampler.
func NewGuaranteedThroughputProbabilisticSampler(
	lowerBound, samplingRate float64,
) (*GuaranteedThroughputProbabilisticSampler, error) {
	return newGuaranteedThroughputProbabilisticSampler(lowerBound, samplingRate), nil
}

func newGuaranteedThroughputProbabilisticSampler(lowerBound, samplingRate float64) *GuaranteedThroughputProbabilisticSampler {
	s := &GuaranteedThroughputProbabilisticSampler{
		lowerBoundSampler: NewRateLimitingSampler(lowerBound),
		lowerBound:        lowerBound,
	}
	s.setProbabilisticSampler(samplingRate)
	return s
}

func (s *GuaranteedThroughputProbabilisticSampler) setProbabilisticSampler(samplingRate float64) {
	if s.probabilisticSampler == nil || s.samplingRate != samplingRate {
		s.probabilisticSampler = newProbabilisticSampler(samplingRate)
		s.samplingRate = s.probabilisticSampler.SamplingRate()
		s.tags = []Tag{
			{key: SamplerTypeTagKey, value: SamplerTypeLowerBound},
			{key: SamplerParamTagKey, value: s.samplingRate},
		}
	}
}

// IsSampled implements IsSampled() of Sampler.
func (s *GuaranteedThroughputProbabilisticSampler) IsSampled(id TraceID, operation string) (bool, []Tag) {
	if sampled, tags := s.probabilisticSampler.IsSampled(id, operation); sampled {
		s.lowerBoundSampler.IsSampled(id, operation)
		return true, tags
	}
	sampled, _ := s.lowerBoundSampler.IsSampled(id, operation)
	return sampled, s.tags
}

// Close implements Close() of Sampler.
func (s *GuaranteedThroughputProbabilisticSampler) Close() {
	s.probabilisticSampler.Close()
	s.lowerBoundSampler.Close()
}

// Equal implements Equal() of Sampler.
func (s *GuaranteedThroughputProbabilisticSampler) Equal(other Sampler) bool {
	// NB The Equal() function is expensive and will be removed. See adaptiveSampler.Equal() for
	// more information.
	return false
}

// this function should only be called while holding a Write lock
func (s *GuaranteedThroughputProbabilisticSampler) update(lowerBound, samplingRate float64) {
	s.setProbabilisticSampler(samplingRate)
	if s.lowerBound != lowerBound {
		s.lowerBoundSampler = NewRateLimitingSampler(lowerBound)
		s.lowerBound = lowerBound
	}
}

// -----------------------

type adaptiveSampler struct {
	sync.RWMutex

	samplers       map[string]*GuaranteedThroughputProbabilisticSampler
	defaultSampler *ProbabilisticSampler
	lowerBound     float64
	maxOperations  int
}

// NewAdaptiveSampler returns a delegating sampler that applies both probabilisticSampler and
// rateLimitingSampler via the guaranteedThroughputProbabilisticSampler. This sampler keeps track of all
// operations and delegates calls to the respective guaranteedThroughputProbabilisticSampler.
func NewAdaptiveSampler(strategies *sampling.PerOperationSamplingStrategies, maxOperations int) (Sampler, error) {
	return newAdaptiveSampler(strategies, maxOperations), nil
}

func newAdaptiveSampler(strategies *sampling.PerOperationSamplingStrategies, maxOperations int) Sampler {
	samplers := make(map[string]*GuaranteedThroughputProbabilisticSampler)
	for _, strategy := range strategies.PerOperationStrategies {
		sampler := newGuaranteedThroughputProbabilisticSampler(
			strategies.DefaultLowerBoundTracesPerSecond,
			strategy.ProbabilisticSampling.SamplingRate,
		)
		samplers[strategy.Operation] = sampler
	}
	return &adaptiveSampler{
		samplers:       samplers,
		defaultSampler: newProbabilisticSampler(strategies.DefaultSamplingProbability),
		lowerBound:     strategies.DefaultLowerBoundTracesPerSecond,
		maxOperations:  maxOperations,
	}
}

func (s *adaptiveSampler) IsSampled(id TraceID, operation string) (bool, []Tag) {
	s.RLock()
	sampler, ok := s.samplers[operation]
	if ok {
		defer s.RUnlock()
		return sampler.IsSampled(id, operation)
	}
	s.RUnlock()
	s.Lock()
	defer s.Unlock()

	// Check if sampler has already been created
	sampler, ok = s.samplers[operation]
	if ok {
		return sampler.IsSampled(id, operation)
	}
	// Store only up to maxOperations of unique ops.
	if len(s.samplers) >= s.maxOperations {
		return s.defaultSampler.IsSampled(id, operation)
	}
	newSampler := newGuaranteedThroughputProbabilisticSampler(s.lowerBound, s.defaultSampler.SamplingRate())
	s.samplers[operation] = newSampler
	return newSampler.IsSampled(id, operation)
}

func (s *adaptiveSampler) Close() {
	s.Lock()
	defer s.Unlock()
	for _, sampler := range s.samplers {
		sampler.Close()
	}
}

func (s *adaptiveSampler) Equal(other Sampler) bool {
	// NB The Equal() function is overly expensive for adaptiveSampler since it's composed of multiple
	// samplers which all need to be initialized before this function can be called for a comparison.
	// Therefore, adaptiveSampler uses the update() function to only alter the samplers that need
	// changing. Hence this function always returns false so that the update function can be called.
	// Once the Equal() function is removed from the Sampler API, this will no longer be needed.
	return false
}

func (s *adaptiveSampler) update(strategies *sampling.PerOperationSamplingStrategies) {
	s.Lock()
	defer s.Unlock()
	for _, strategy := range strategies.PerOperationStrategies {
		operation := strategy.Operation
		samplingRate := strategy.ProbabilisticSampling.SamplingRate
		lowerBound := strategies.DefaultLowerBoundTracesPerSecond
		if sampler, ok := s.samplers[operation]; ok {
			sampler.update(lowerBound, samplingRate)
		} else {
			sampler := newGuaranteedThroughputProbabilisticSampler(
				lowerBound,
				samplingRate,
			)
			s.samplers[operation] = sampler
		}
	}
	s.lowerBound = strategies.DefaultLowerBoundTracesPerSecond
	if s.defaultSampler.SamplingRate() != strategies.DefaultSamplingProbability {
		s.defaultSampler = newProbabilisticSampler(strategies.DefaultSamplingProbability)
	}
}

// -----------------------

// RemotelyControlledSampler is a delegating sampler that polls a remote server
// for the appropriate sampling strategy, constructs a corresponding sampler and
// delegates to it for sampling decisions.
type RemotelyControlledSampler struct {
	sync.RWMutex
	samplerOptions

	serviceName string
	timer       *time.Ticker
	manager     sampling.SamplingManager
	pollStopped sync.WaitGroup
}

type httpSamplingManager struct {
	serverURL string
}

func (s *httpSamplingManager) GetSamplingStrategy(serviceName string) (*sampling.SamplingStrategyResponse, error) {
	var out sampling.SamplingStrategyResponse
	v := url.Values{}
	v.Set("service", serviceName)
	if err := utils.GetJSON(s.serverURL+"?"+v.Encode(), &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// NewRemotelyControlledSampler creates a sampler that periodically pulls
// the sampling strategy from an HTTP sampling server (e.g. jaeger-agent).
func NewRemotelyControlledSampler(
	serviceName string,
	opts ...SamplerOption,
) *RemotelyControlledSampler {
	options := applySamplerOptions(opts...)
	sampler := &RemotelyControlledSampler{
		samplerOptions: options,
		serviceName:    serviceName,
		timer:          time.NewTicker(options.samplingRefreshInterval),
		manager:        &httpSamplingManager{serverURL: options.samplingServerURL},
	}

	go sampler.pollController()
	return sampler
}

func applySamplerOptions(opts ...SamplerOption) samplerOptions {
	options := samplerOptions{}
	for _, option := range opts {
		option(&options)
	}
	if options.sampler == nil {
		options.sampler = newProbabilisticSampler(0.001)
	}
	if options.logger == nil {
		options.logger = log.NullLogger
	}
	if options.maxOperations <= 0 {
		options.maxOperations = defaultMaxOperations
	}
	if options.samplingServerURL == "" {
		options.samplingServerURL = defaultSamplingServerURL
	}
	if options.metrics == nil {
		options.metrics = NewNullMetrics()
	}
	if options.samplingRefreshInterval <= 0 {
		options.samplingRefreshInterval = defaultSamplingRefreshInterval
	}
	return options
}

// IsSampled implements IsSampled() of Sampler.
func (s *RemotelyControlledSampler) IsSampled(id TraceID, operation string) (bool, []Tag) {
	s.RLock()
	defer s.RUnlock()
	return s.sampler.IsSampled(id, operation)
}

// Close implements Close() of Sampler.
func (s *RemotelyControlledSampler) Close() {
	s.RLock()
	s.timer.Stop()
	s.RUnlock()

	s.pollStopped.Wait()
}

// Equal implements Equal() of Sampler.
func (s *RemotelyControlledSampler) Equal(other Sampler) bool {
	// NB The Equal() function is expensive and will be removed. See adaptiveSampler.Equal() for
	// more information.
	if o, ok := other.(*RemotelyControlledSampler); ok {
		s.RLock()
		o.RLock()
		defer s.RUnlock()
		defer o.RUnlock()
		return s.sampler.Equal(o.sampler)
	}
	return false
}

func (s *RemotelyControlledSampler) pollController() {
	// in unit tests we re-assign the timer Ticker, so need to lock to avoid data races
	s.Lock()
	timer := s.timer
	s.Unlock()

	for range timer.C {
		s.updateSampler()
	}
	s.pollStopped.Add(1)
}

func (s *RemotelyControlledSampler) updateSampler() {
	res, err := s.manager.GetSamplingStrategy(s.serviceName)
	if err != nil {
		s.metrics.SamplerQueryFailure.Inc(1)
		return
	}
	s.Lock()
	defer s.Unlock()

	s.metrics.SamplerRetrieved.Inc(1)
	if strategies := res.GetOperationSampling(); strategies != nil {
		s.updateAdaptiveSampler(strategies)
	} else {
		err = s.updateRateLimitingOrProbabilisticSampler(res)
	}
	if err != nil {
		s.metrics.SamplerUpdateFailure.Inc(1)
		s.logger.Infof("Unable to handle sampling strategy response %+v. Got error: %v", res, err)
		return
	}
	s.metrics.SamplerUpdated.Inc(1)
}

// NB: this function should only be called while holding a Write lock
func (s *RemotelyControlledSampler) updateAdaptiveSampler(strategies *sampling.PerOperationSamplingStrategies) {
	if adaptiveSampler, ok := s.sampler.(*adaptiveSampler); ok {
		adaptiveSampler.update(strategies)
	} else {
		s.sampler = newAdaptiveSampler(strategies, s.maxOperations)
	}
}

// NB: this function should only be called while holding a Write lock
func (s *RemotelyControlledSampler) updateRateLimitingOrProbabilisticSampler(res *sampling.SamplingStrategyResponse) error {
	var newSampler Sampler
	if probabilistic := res.GetProbabilisticSampling(); probabilistic != nil {
		newSampler = newProbabilisticSampler(probabilistic.SamplingRate)
	} else if rateLimiting := res.GetRateLimitingSampling(); rateLimiting != nil {
		newSampler = NewRateLimitingSampler(float64(rateLimiting.MaxTracesPerSecond))
	} else {
		return fmt.Errorf("Unsupported sampling strategy type %v", res.GetStrategyType())
	}
	if !s.sampler.Equal(newSampler) {
		s.sampler = newSampler
	}
	return nil
}
