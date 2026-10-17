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

package affinity

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/json"
	dracel "k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/klog/v2"
	"k8s.io/utils/keymutex"
	"k8s.io/utils/lru"
	"k8s.io/utils/ptr"
)

// PoolKey identifies a ResourceSlice pool by driver, pool name, and generation.
type PoolKey struct {
	Driver     string
	Name       string
	Generation int64
}

// PoolExtractors is the set of SharingAffinity extractors published per pool
type PoolExtractors struct {
	Key        PoolKey
	Extractors []resourceapi.SharingAffinityExtractor
}

// Request identifies the claim request being filtered. ParentName is set for a
// subrequest, so QualifiedName is "<ParentName>/<Name>".
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

// CandidateDevice is the view of a device that selectors and keys are evaluated
// against. ID is used only for error and log attribution.
type CandidateDevice struct {
	ID     string
	Driver string
	Device resourceapi.Device
}

// Extractor compiles and evaluates a pool's SharingAffinity CEL extractors. It
// is safe for concurrent use and is meant to be long-lived and shared across
// pools, generations, and scheduling cycles: compiled programs are cached by
// expression text, and selector programs via the shared selectorCache.
type Extractor struct {
	compileMutex  keymutex.KeyMutex
	cacheMutex    sync.RWMutex
	cache         *lru.Cache
	selectorCache *dracel.Cache
}

// compiledKey is one affinity key of a keySet: the key name, its source CEL
// expression, and the compiled program that produces the key's value.
type compiledKey struct {
	key        string
	expression string
	program    *compiledExpression
}

// compiledKeySet is the compiled form of one SharingAffinity extractor entry:
// an optional device selector plus the affinity keys it yields. The selector
// gates which candidate devices this key set applies to (nil = all devices).
type compiledKeySet struct {
	selector *resourceapi.DeviceSelector
	keys     []compiledKey
}

// NewExtractor returns an Extractor whose compiled-program cache holds up to
// maxCacheEntries entries. selectorCache is the shared DRA selector CEL cache
// used for device-selector matching; a fresh one is created when nil.
func NewExtractor(maxCacheEntries int, selectorCache *dracel.Cache) *Extractor {
	if maxCacheEntries <= 0 {
		maxCacheEntries = 1
	}
	if selectorCache == nil {
		selectorCache = dracel.NewCache(maxCacheEntries, dracel.Features{})
	}
	return &Extractor{
		compileMutex:  keymutex.NewHashed(0),
		cache:         lru.New(maxCacheEntries),
		selectorCache: selectorCache,
	}
}

// Compile pre-warms the compiled-program cache for the pool's extractors and
// returns an error if any extractor expression fails to compile. It is optional:
// Extract compiles lazily on demand.
func (e *Extractor) Compile(ctx context.Context, pool PoolExtractors) error {
	logger := klog.FromContext(ctx)
	logger.V(6).Info("Compiling sharing affinity extractors", "pool", pool.Key.Name, "driver", pool.Key.Driver, "generation", pool.Key.Generation, "numExtractors", len(pool.Extractors))

	_, err := e.compileKeySets(pool)
	if err != nil {
		return err
	}
	return nil
}

// Extract returns the sharing-affinity key/value map the device would carry if
// allocated to req, obtained by running the pool's compiled CEL extractors
// against the request's in-scope opaque configs. It is called once per
// (request, candidate device) pair.
//
// An empty map means the device carries no affinity constraint for this
// request. A non-nil error means the device is not a viable candidate for req;
// it is a typed *Error (see errors.go) classifying the failure.
func (e *Extractor) Extract(ctx context.Context, pool PoolExtractors, req Request, device CandidateDevice, configs []resourceapi.DeviceClaimConfiguration) (map[string]string, error) {
	// Compile the pool's extractors into keySets. Each keySet is one
	// sharingAffinity entry: a device selector plus its keyed CEL programs.
	keySets, err := e.compileKeySets(pool)
	if err != nil {
		return nil, err
	}

	// Gather the opaque-config objects in scope for this request: those that
	// target req (or the whole claim, via inScope) and carry opaque parameters
	// for this driver. Each is parsed into a map that CEL reads as `object`.
	requestName := req.QualifiedName()
	reqScopedObjects := make([]map[string]any, 0, len(configs))
	for _, cfg := range configs {
		if !inScope(req, cfg) {
			continue
		}
		object, ok, err := parseOpaqueObject(pool, cfg)
		if err != nil {
			return nil, &Error{Kind: ErrCELEval, Pool: pool.Key, Request: requestName, Device: device.ID, ExtractorIndex: -1, Err: err}
		}
		if !ok {
			continue
		}
		reqScopedObjects = append(reqScopedObjects, object)
	}

	// Evaluate every applicable keySet and merge its keys into result.
	result := map[string]string{}
	for keySetIndex, keySet := range keySets {
		// Selector gate: skip keySets whose device selector does not match this
		// device (a nil selector matches every device).
		matches, err := selectorMatches(ctx, e.selectorCache, keySet.selector, device)
		if err != nil {
			return nil, &Error{Kind: ErrCELEval, Pool: pool.Key, Request: requestName, Device: device.ID, ExtractorIndex: keySetIndex, Err: err}
		}
		if !matches {
			continue
		}

		// Phase 1 — produce: run each key's CEL against every in-scope object. A
		// non-empty return is the key's value; "" means "does not apply to this
		// object" and is skipped. anyProduced tracks whether this keySet
		// contributed anything at all.
		keySetValues := map[string]string{}
		anyProduced := false
		for _, compiledKey := range keySet.keys {
			for _, object := range reqScopedObjects {
				value, details, err := compiledKey.program.eval(ctx, object)
				if err != nil {
					return nil, classifyEvalError(&Error{
						Pool:           pool.Key,
						Request:        requestName,
						Device:         device.ID,
						ExtractorIndex: keySetIndex,
						Key:            compiledKey.key,
						Expression:     compiledKey.expression,
					}, err)
				}
				klog.FromContext(ctx).V(7).Info("Sharing affinity CEL result", "pool", pool.Key.Name, "driver", pool.Key.Driver, "generation", pool.Key.Generation, "request", requestName, "device", device.ID, "extractor", keySetIndex, "key", compiledKey.key, "value", value, "actualCost", ptr.Deref(details.ActualCost(), 0))
				if value == "" {
					continue
				}
				anyProduced = true
				// Self-inconsistency: one request's own objects must not resolve
				// the same key to conflicting values.
				if previous, ok := keySetValues[compiledKey.key]; ok && previous != value {
					return nil, &Error{Kind: ErrClaimSelfInconsistent, Pool: pool.Key, Request: requestName, Device: device.ID, ExtractorIndex: keySetIndex, Key: compiledKey.key}
				}
				keySetValues[compiledKey.key] = value
			}
		}
		// A keySet that produced nothing simply does not apply to this claim; it
		// is not an error.
		if !anyProduced {
			continue
		}

		// Phase 2 — finalize (strict, all-or-nothing): once a keySet produces any
		// key it must produce all of them, and applicable keySets must agree on
		// the value of any key they share for this device.
		for _, compiledKey := range keySet.keys {
			value, ok := keySetValues[compiledKey.key]
			if !ok {
				// Partially satisfied: some keys of this keySet produced a value,
				// this one did not.
				return nil, &Error{Kind: ErrMissingField, Pool: pool.Key, Request: requestName, Device: device.ID, ExtractorIndex: keySetIndex, Key: compiledKey.key}
			}
			// Per-device collision: two applicable keySets set the same key to
			// different values for this device. Sharing a key across keySets
			// (e.g. disjoint selectors) is allowed by design; only a
			// value conflict on a single device is rejected.
			if previous, exists := result[compiledKey.key]; exists && previous != value {
				return nil, &Error{Kind: ErrSliceKeyCollision, Pool: pool.Key, Request: requestName, Device: device.ID, ExtractorIndex: keySetIndex, Key: compiledKey.key}
			}
			result[compiledKey.key] = value
		}
	}

	return result, nil
}

// compileKeySets compiles every extractor in the pool into a compiledKeySet,
// reusing cached programs. Each extractor's keys are compiled in sorted order so
// that, when several keys fail, the first reported error (here and in Extract)
// is deterministic rather than dependent on Go's map iteration order.
func (e *Extractor) compileKeySets(pool PoolExtractors) ([]compiledKeySet, error) {
	keySets := make([]compiledKeySet, 0, len(pool.Extractors))
	for keySetIndex, extractor := range pool.Extractors {
		keys := make([]string, 0, len(extractor.CEL))
		for key := range extractor.CEL {
			keys = append(keys, key)
		}
		sort.Strings(keys)

		keySet := compiledKeySet{selector: extractor.Selector, keys: make([]compiledKey, 0, len(keys))}
		for _, key := range keys {
			expression := extractor.CEL[key]
			program, err := e.getOrCompile(expression)
			if err != nil {
				kind := ErrCELEval
				if errors.Is(err, errCostLimitExceeded) {
					kind = ErrCostExceeded
				}
				return nil, &Error{Kind: kind, Pool: pool.Key, ExtractorIndex: keySetIndex, Key: key, Expression: expression, Err: err}
			}
			keySet.keys = append(keySet.keys, compiledKey{key: key, expression: expression, program: program})
		}
		keySets = append(keySets, keySet)
	}
	return keySets, nil
}

// getOrCompile returns the compiled program for expression, compiling and
// caching it on first use.
func (e *Extractor) getOrCompile(expression string) (*compiledExpression, error) {
	// The cache is keyed by the expression text alone, so identical expressions
	// compile to identical programs regardless of driver, pool, or generation
	// and can safely share one cache entry.
	e.compileMutex.LockKey(expression)
	defer func() {
		//nolint:errcheck // Only returns an error for unknown keys, which isn't the case here.
		e.compileMutex.UnlockKey(expression)
	}()

	if cached := e.get(expression); cached != nil {
		return cached, nil
	}

	compiled, err := compileExpression(expression, compileOptions{})
	if err != nil {
		return nil, err
	}
	e.add(expression, compiled)
	return compiled, nil
}

// add stores a compiled program in the cache.
func (e *Extractor) add(expression string, compiled *compiledExpression) {
	e.cacheMutex.Lock()
	defer e.cacheMutex.Unlock()
	e.cache.Add(expression, compiled)
}

// get returns the cached compiled program for expression, or nil if absent.
func (e *Extractor) get(expression string) *compiledExpression {
	e.cacheMutex.RLock()
	defer e.cacheMutex.RUnlock()
	cached, ok := e.cache.Get(expression)
	if !ok {
		return nil
	}
	return cached.(*compiledExpression)
}

// inScope reports whether cfg applies to req. A config with no Requests applies
// to the whole claim; otherwise it must name the request's qualified name or,
// for a subrequest, its parent request.
func inScope(req Request, cfg resourceapi.DeviceClaimConfiguration) bool {
	if len(cfg.Requests) == 0 {
		return true
	}
	qualified := req.QualifiedName()
	for _, request := range cfg.Requests {
		if request == qualified {
			return true
		}
		if req.ParentName != "" && request == req.ParentName {
			return true
		}
	}
	return false
}

// parseOpaqueObject parses cfg's opaque parameters into a CEL-readable object.
func parseOpaqueObject(pool PoolExtractors, cfg resourceapi.DeviceClaimConfiguration) (map[string]any, bool, error) {
	if cfg.Opaque == nil {
		return nil, false, nil
	}
	if cfg.Opaque.Driver != "" && cfg.Opaque.Driver != pool.Key.Driver {
		return nil, false, nil
	}
	object := map[string]any{}
	if len(cfg.Opaque.Parameters.Raw) == 0 {
		return object, true, nil
	}
	if err := json.Unmarshal(cfg.Opaque.Parameters.Raw, &object); err != nil {
		return nil, true, fmt.Errorf("unmarshal opaque parameters: %w", err)
	}
	return object, true, nil
}

// classifyEvalError sets base.Kind from a CEL evaluation error and attaches the
// enhanced error. A non-string result, a missing field ("no such key"), and a
// cost-limit breach are distinguished; anything else becomes a generic
// ErrCELEval.
func classifyEvalError(base *Error, err error) error {
	base.Err = dracel.EnhanceRuntimeError(err)
	errString := base.Err.Error()
	switch {
	case errors.Is(err, errNonStringResult):
		base.Kind = ErrNonStringReturn
	case strings.Contains(errString, "no such key:"):
		base.Kind = ErrMissingField
	case strings.Contains(errString, "cost") && (strings.Contains(errString, "exceeded") || strings.Contains(errString, "limit")):
		base.Kind = ErrCostExceeded
	default:
		base.Kind = ErrCELEval
	}
	return base
}

// selectorMatches reports whether device satisfies the keySet's device selector,
// evaluating the selector's CEL through the shared DRA selector cache. A nil
// selector (or nil CEL) matches every device.
func selectorMatches(ctx context.Context, cache *dracel.Cache, selector *resourceapi.DeviceSelector, device CandidateDevice) (bool, error) {
	if selector == nil || selector.CEL == nil {
		return true, nil
	}
	compiled := cache.GetOrCompile(selector.CEL.Expression)
	if compiled.Error != nil {
		return false, compiled.Error
	}
	matches, _, err := compiled.DeviceMatches(ctx, dracel.Device{
		Driver:                   device.Driver,
		AllowMultipleAllocations: device.Device.AllowMultipleAllocations,
		Attributes:               device.Device.Attributes,
		Capacity:                 device.Device.Capacity,
	})
	if err != nil {
		return false, err
	}
	return matches, nil
}
