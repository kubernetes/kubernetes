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

	"github.com/google/cel-go/common/types"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/klog/v2"
	"k8s.io/utils/keymutex"
	"k8s.io/utils/lru"
)

// PoolKey identifies a ResourceSlice pool by driver, pool name, and generation.
type PoolKey struct {
	Driver     string
	Name       string
	Generation int64
}

// PoolExtractors is the combined set of extractors published for a pool.
type PoolExtractors struct {
	Key        PoolKey
	Extractors []resourceapi.SharingAffinityExtractor
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

// Result distinguishes an unconstrained pool from an unsatisfied extraction.
type Result struct {
	// Constrained is false only when the pool declares no extractors.
	Constrained bool
	// Satisfied requires a non-empty contribution from every declared extractor.
	Satisfied bool
	// Values may be partial when Satisfied is false.
	Values map[string]string
	// Unsatisfied contains the missing extractor names in sorted order.
	Unsatisfied []string
}

// Extractor compiles and evaluates sharing affinity. It is safe for concurrent
// use across pools and generations. Only immutable CEL programs are cached;
// predicates, names, results and budgets always belong to the current request.
type Extractor struct {
	compileMutex keymutex.KeyMutex
	cacheMutex   sync.RWMutex
	cache        *lru.Cache
}

type cacheKey struct {
	env  environment.Type
	expr string
}

// NewExtractor returns an Extractor with a bounded expression-text program cache.
func NewExtractor(maxCacheEntries int) *Extractor {
	return &Extractor{
		compileMutex: keymutex.NewHashed(0),
		cache:        lru.New(max(1, maxCacheEntries)),
	}
}

// Compile optionally pre-warms the cache and validates the complete pool set.
func (e *Extractor) Compile(ctx context.Context, pool PoolExtractors) error {
	if err := ctxError(ctx); err != nil {
		return err
	}
	_, err := e.compilePool(pool)
	return err
}

func (e *Extractor) compilePool(pool PoolExtractors) ([]compiledExtractor, error) {
	compiled, err := compileExtractorSet(pool.Extractors, storedEnv, e.getOrCompile)
	if err != nil {
		var detail *Error
		if errors.As(err, &detail) {
			detail.Pool = pool.Key
		}
	}
	return compiled, err
}

type parsedConfig struct {
	index      int
	parameters map[string]any
	metadata   configMetadata
}

// Extract runs once per pool/request, with a result reusable across candidate
// devices. Unsatisfied extraction is not an error. Errors always return a zero
// Result, and Classify distinguishes claim, definition and runtime failures.
func (e *Extractor) Extract(ctx context.Context, pool PoolExtractors, req Request, configs []resourceapi.DeviceClaimConfiguration) (result Result, err error) {
	if len(pool.Extractors) == 0 {
		return Result{Satisfied: true}, nil
	}
	requestName := req.QualifiedName()
	logger := klog.FromContext(ctx).WithValues("pool", pool.Key.Name, "driver", pool.Key.Driver, "generation", pool.Key.Generation, "request", requestName)
	logger.V(6).Info("Extracting sharing affinity", "numExtractors", len(pool.Extractors), "numConfigs", len(configs))
	defer func() {
		if err != nil {
			logger.V(5).Info("Sharing affinity extraction failed", "err", err)
		} else {
			logger.V(5).Info("Sharing affinity extracted", "satisfied", result.Satisfied, "unsatisfied", result.Unsatisfied)
		}
	}()
	if err := ctxError(ctx); err != nil {
		return Result{}, err
	}
	compiled, err := e.compilePool(pool)
	if err != nil {
		var detail *Error
		if errors.As(err, &detail) {
			detail.Request = requestName
		}
		return Result{}, err
	}

	// Decode once, after scope checks but before any evaluation. A metadata
	// nonmatch must not hide either a malformed definition or malformed JSON.
	parsed := make([]parsedConfig, 0, len(configs))
	for index, cfg := range configs {
		if err := ctxError(ctx); err != nil {
			return Result{}, err
		}
		if !inScope(req, cfg) {
			continue
		}
		parameters, ok, err := parseOpaqueObject(pool, cfg)
		if err != nil {
			return Result{}, &Error{Kind: ErrParameters, Pool: pool.Key, Request: requestName, ConfigIndex: index, Err: err}
		}
		if ok {
			parsed = append(parsed, parsedConfig{index: index, parameters: parameters, metadata: metadataOf(parameters)})
		}
	}

	result = Result{Constrained: true, Values: make(map[string]string, len(compiled))}
	b := newBudget()
	for _, extractor := range compiled {
		for _, cfg := range parsed {
			base := &Error{Pool: pool.Key, Request: requestName, Extractor: extractor.name, ConfigIndex: cfg.index}
			if err := ctxError(ctx); err != nil {
				base.Kind, base.Err = ErrCELEval, err
				return Result{}, base
			}
			if !extractor.matcher.matches(cfg.metadata) {
				logger.V(7).Info("Sharing affinity config skipped", "extractor", extractor.name, "configIndex", cfg.index, "matched", false, "actualCost", uint64(0), "remainingBudget", b.remaining)
				continue
			}
			program, err := b.program(extractor.program)
			if err != nil {
				return Result{}, classifyEvalError(ctx, base, err)
			}
			value, details, err := program.ContextEval(ctx, map[string]any{parametersVar: cfg.parameters})
			var actual uint64
			if details != nil && details.ActualCost() != nil {
				actual = *details.ActualCost()
			}
			if err != nil {
				b.lastFailedCost = actual
				return Result{}, classifyEvalError(ctx, base, err)
			}
			if err := ctxError(ctx); err != nil {
				return Result{}, classifyEvalError(ctx, base, err)
			}
			b.charge(actual)
			str, ok := value.(types.String)
			if !ok {
				return Result{}, classifyEvalError(ctx, base, errNonStringResult)
			}
			logger.V(7).Info("Sharing affinity CEL result", "extractor", extractor.name, "configIndex", cfg.index, "matched", true, "actualCost", actual, "remainingBudget", b.remaining, "contributed", str != "")
			if str == "" {
				continue
			}
			if previous, exists := result.Values[extractor.name]; exists && previous != string(str) {
				base.Kind = ErrClaimSelfInconsistent
				return Result{}, base
			}
			result.Values[extractor.name] = string(str)
		}
		if _, exists := result.Values[extractor.name]; !exists {
			result.Unsatisfied = append(result.Unsatisfied, extractor.name)
		}
	}
	if err := ctxError(ctx); err != nil {
		return Result{}, err
	}
	sort.Strings(result.Unsatisfied)
	result.Satisfied = len(result.Unsatisfied) == 0
	return result, nil
}

func (e *Extractor) getOrCompile(expression string, envType environment.Type) (*compiledExpression, error) {
	key := cacheKey{env: envType, expr: expression}
	lockKey := string(envType) + "\x00" + expression
	e.compileMutex.LockKey(lockKey)
	defer func() {
		// Unlock only fails for an unknown key.
		_ = e.compileMutex.UnlockKey(lockKey)
	}()
	e.cacheMutex.RLock()
	cached, ok := e.cache.Get(key)
	e.cacheMutex.RUnlock()
	if ok {
		return cached.(*compiledExpression), nil
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
