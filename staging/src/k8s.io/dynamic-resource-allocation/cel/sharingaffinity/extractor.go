/*
Copyright The Kubernetes Authors.

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

package sharingaffinity

import (
	"context"
	"errors"
	"sort"
	"sync"

	"cel.dev/cel-go/common/types"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/klog/v2"
	"k8s.io/utils/lru"
)

// PoolKey identifies a ResourceSlice pool by driver, pool name, and generation.
type PoolKey struct {
	Driver     string
	Name       string
	Generation int64
}

// PoolExtractors is the combined set of extractors published for a pool and
// pre-compiled CEL programs for the extractor expressions.
type PoolExtractors struct {
	Key        PoolKey
	Extractors []resourceapi.SharingAffinityExtractor
	compiled   []compiledExtractor
}

// Request identifies a claim request. ParentName is set for a subrequest.
type Request struct {
	Name       string
	ParentName string
}

// QualifiedName returns Name, or "<ParentName>/<Name>" for a subrequest.
func (r Request) QualifiedName() string {
	if r.ParentName == "" {
		return r.Name
	}
	return r.ParentName + "/" + r.Name
}

// RequestAffinity holds the sharing affinity values extracted for a request
// against a pool, along with whether the pool defines extractors and whether
// every extractor produced a value.
type RequestAffinity struct {
	// HasSharingAffinity is false only when the pool declares no extractors.
	HasSharingAffinity bool
	// AllExtracted requires a non-empty contribution from every declared extractor.
	AllExtracted bool
	// Values may be partial when AllExtracted is false.
	Values map[string]string
	// MissingExtractors contains the missing extractor names in sorted order.
	MissingExtractors []string
}

// Extractor compiles and evaluates sharing affinity. It is safe for concurrent
// use across pools and generations. Only immutable CEL programs are cached;
type Extractor struct {
	cacheMutex sync.Mutex
	cache      *lru.Cache
	metrics    MetricsRecorder
}

type cacheKey struct {
	env  environment.Type
	expr string
}

// MetricsRecorder records sharing-affinity evaluation and filter rejection events.
type MetricsRecorder struct {
	IncExtractorEvaluations func()
	IncLockConflicts        func()
	IncNoKeysExtracted      func()
}

// SetMetricsRecorder configures hooks for sharing-affinity scheduler metrics on this extractor.
func (e *Extractor) SetMetricsRecorder(recorder MetricsRecorder) {
	if e == nil {
		return
	}
	e.cacheMutex.Lock()
	defer e.cacheMutex.Unlock()
	e.metrics = recorder
}

// RecordExtractorEvaluation records one CEL extractor evaluation.
func (e *Extractor) RecordExtractorEvaluation() {
	if e != nil && e.metrics.IncExtractorEvaluations != nil {
		e.metrics.IncExtractorEvaluations()
	}
}

// RecordLockConflict records a candidate device rejection due to an incompatible affinity lock.
func (e *Extractor) RecordLockConflict() {
	if e != nil && e.metrics.IncLockConflicts != nil {
		e.metrics.IncLockConflicts()
	}
}

// RecordNoKeysExtracted records a candidate device rejection due to missing affinity extractors.
func (e *Extractor) RecordNoKeysExtracted() {
	if e != nil && e.metrics.IncNoKeysExtracted != nil {
		e.metrics.IncNoKeysExtracted()
	}
}

// NewExtractor returns an Extractor with a bounded expression-text program cache.
func NewExtractor(maxCacheEntries int) *Extractor {
	return &Extractor{
		cache: lru.New(max(1, maxCacheEntries)),
	}
}

// CompilePoolExtractors compiles the pool's CEL extractor expressions,
// validates their combined static cost, and returns pool with the compiled
// programs populated.
// The caller is responsible for validating pool metadata, including name uniqueness.
func (e *Extractor) CompilePoolExtractors(_ context.Context, pool PoolExtractors) (PoolExtractors, error) {
	compiled, err := e.compilePool(pool, "")
	if err != nil {
		return PoolExtractors{}, err
	}
	pool.compiled = compiled
	return pool, nil
}

// compilePool compiles all extractor expressions for a pool and enforces the combined cost limit.
func (e *Extractor) compilePool(pool PoolExtractors, requestName string) ([]compiledExtractor, error) {
	return compileExpressions(pool.Key, requestName, pool.Extractors, e.getOrCompile)
}

type parsedConfig struct {
	index      int
	parameters map[string]any
	metadata   configMetadata
}

// ExtractRequestAffinity evaluates a pool's sharing affinity extractors against
// the in-scope opaque configurations for req and returns the extracted affinity values.
func (e *Extractor) ExtractRequestAffinity(ctx context.Context, pool PoolExtractors, req Request, configs []resourceapi.DeviceClaimConfiguration) (result RequestAffinity, err error) {
	if len(pool.Extractors) == 0 {
		return RequestAffinity{AllExtracted: true}, nil
	}
	requestName := req.QualifiedName()
	logger := klog.FromContext(ctx).WithValues("pool", pool.Key.Name, "driver", pool.Key.Driver, "generation", pool.Key.Generation, "request", requestName)
	logger.V(6).Info("Extracting sharing affinity", "numExtractors", len(pool.Extractors), "numConfigs", len(configs))
	defer func() {
		if err != nil {
			logger.V(6).Info("Sharing affinity extraction failed", "err", err)
		} else {
			logger.V(6).Info("Sharing affinity extracted", "allExtracted", result.AllExtracted, "missingExtractors", result.MissingExtractors)
		}
	}()
	compiled := pool.compiled
	if len(compiled) == 0 {
		compiled, err = e.compilePool(pool, requestName)
		if err != nil {
			return RequestAffinity{}, err
		}
	}

	// Filter configs to those targeting req and the pool's driver, and unmarshal
	// their opaque JSON parameters once before evaluating extractors.
	parsed := make([]parsedConfig, 0, len(configs))
	for index, cfg := range configs {
		if !inScope(req, cfg) {
			continue
		}
		parameters, ok, err := parseOpaqueObject(pool, cfg)
		if err != nil {
			return RequestAffinity{}, &Error{Kind: ErrParameters, Pool: pool.Key, Request: requestName, ConfigIndex: index, Err: err}
		}
		if ok {
			parsed = append(parsed, parsedConfig{index: index, parameters: parameters, metadata: metadataOf(parameters)})
		}
	}

	result = RequestAffinity{HasSharingAffinity: true, Values: make(map[string]string, len(compiled))}
	b := newBudget()
	for _, extractor := range compiled {
		for _, cfg := range parsed {
			base := &Error{Pool: pool.Key, Request: requestName, Extractor: extractor.name, ConfigIndex: cfg.index}
			if !extractor.matcher.matches(cfg.metadata) {
				logger.V(7).Info("Sharing affinity config skipped", "extractor", extractor.name, "configIndex", cfg.index, "matched", false, "actualCost", uint64(0), "remainingBudget", b.remaining)
				continue
			}
			program, err := b.program(extractor.program)
			if err != nil {
				return RequestAffinity{}, classifyEvalError(ctx, base, err)
			}
			value, details, err := program.ContextEval(ctx, map[string]any{parametersVar: cfg.parameters})
			e.RecordExtractorEvaluation()
			var actual uint64
			if details != nil && details.ActualCost() != nil {
				actual = *details.ActualCost()
			}
			if err != nil {
				b.lastFailedCost = actual
				return RequestAffinity{}, classifyEvalError(ctx, base, err)
			}
			b.charge(actual)
			str, ok := value.(types.String)
			if !ok {
				return RequestAffinity{}, classifyEvalError(ctx, base, errNonStringResult)
			}
			logger.V(7).Info("Sharing affinity CEL result", "extractor", extractor.name, "configIndex", cfg.index, "matched", true, "actualCost", actual, "remainingBudget", b.remaining, "contributed", str != "")
			if str == "" {
				continue
			}
			if previous, exists := result.Values[extractor.name]; exists && previous != string(str) {
				base.Kind = ErrClaimSelfInconsistent
				return RequestAffinity{}, base
			}
			result.Values[extractor.name] = string(str)
		}
		if _, exists := result.Values[extractor.name]; !exists {
			result.MissingExtractors = append(result.MissingExtractors, extractor.name)
		}
	}
	sort.Strings(result.MissingExtractors)
	result.AllExtracted = len(result.MissingExtractors) == 0
	return result, nil
}

// getOrCompile returns the cached compiled expression for (expression, envType),
// compiling and caching it on a cache miss.
func (e *Extractor) getOrCompile(expression string, envType environment.Type) (*compiledExpression, error) {
	key := cacheKey{env: envType, expr: expression}
	if cached, ok := e.getCached(key); ok {
		return cached, nil
	}
	compiled, err := compileExpression(expression, envType)
	if err != nil {
		return nil, err
	}
	e.cacheMutex.Lock()
	e.cache.Add(key, compiled)
	e.cacheMutex.Unlock()
	return compiled, nil
}

// getCached looks up key in the LRU program cache.
func (e *Extractor) getCached(key cacheKey) (*compiledExpression, bool) {
	// lru.Cache.Get moves the accessed element to the front of its internal
	// doubly-linked list, so it requires an exclusive lock.
	e.cacheMutex.Lock()
	defer e.cacheMutex.Unlock()
	cached, ok := e.cache.Get(key)
	if !ok {
		return nil, false
	}
	return cached.(*compiledExpression), true
}

// inScope preserves claim-wide, request and qualified-subrequest targeting.
func inScope(req Request, cfg resourceapi.DeviceClaimConfiguration) bool {
	if len(cfg.Requests) == 0 {
		return true
	}
	qualified := req.QualifiedName()
	for _, request := range cfg.Requests {
		if request == qualified || req.ParentName != "" && request == req.ParentName {
			return true
		}
	}
	return false
}

// parseOpaqueObject decodes cfg's opaque JSON parameters when cfg targets the
// pool's driver or is driver-agnostic, and verifies the top-level JSON value
// is a non-null object.
func parseOpaqueObject(pool PoolExtractors, cfg resourceapi.DeviceClaimConfiguration) (map[string]any, bool, error) {
	if cfg.Opaque == nil || cfg.Opaque.Driver != "" && cfg.Opaque.Driver != pool.Key.Driver {
		return nil, false, nil
	}
	var decoded any
	if err := json.Unmarshal(cfg.Opaque.Parameters.Raw, &decoded); err != nil {
		return nil, true, redactParameters(err)
	}
	parameters, ok := decoded.(map[string]any)
	if !ok || parameters == nil {
		return nil, true, redactParameters(errors.New("expected a non-null JSON object"))
	}
	return parameters, true, nil
}
