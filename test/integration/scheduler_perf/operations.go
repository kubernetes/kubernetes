/*
Copyright 2019 The Kubernetes Authors.

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

package benchmark

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"html/template"
	"maps"
	"os"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/restmapper"
	"k8s.io/klog/v2"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"
)

// createAny defines an op where some object gets created from a YAML file.
// The nameset can be specified.
type createAny struct {
	// Must match createAnyOpcode.
	Opcode operationCode
	// Namespace the object should be created in. Must be empty for cluster-scoped objects.
	Namespace string
	// Path to spec file describing the object to create.
	// This will be processed with text/template.
	// .Index will be in the range [0, Count-1] when creating
	// more than one object. .Count is the total number of objects.
	TemplatePath string
	// Count determines how many objects get created. Defaults to 1 if unset.
	Count      *int
	CountParam string
	// Params to be passed to the template.
	// Values with `$` prefix will be resolved to the workload parameters.
	TemplateParams map[string]any
}

var _ runnableOp = &createAny{}

func (c *createAny) isValid(allowParameterization bool) error {
	if c.TemplatePath == "" {
		return fmt.Errorf("TemplatePath must be set")
	}
	// The namespace can only be checked during later because we don't know yet
	// whether the object is namespaced or cluster-scoped.
	return nil
}

func (c *createAny) collectsMetrics() bool {
	return false
}

func (c createAny) patchParams(w *workload) (realOp, error) {
	if c.CountParam != "" {
		count, err := w.Params.get(c.CountParam[1:])
		if err != nil {
			return nil, err
		}
		c.Count = ptr.To(count)
	}
	if len(c.TemplateParams) > 0 {
		var err error
		c.TemplateParams, err = resolveTemplateParams(c.TemplateParams, w)
		if err != nil {
			return nil, err
		}
	}
	return &c, c.isValid(false)
}

func (c *createAny) requiredNamespaces() []string {
	if c.Namespace == "" {
		return nil
	}
	return []string{c.Namespace}
}

func (c *createAny) run(tCtx ktesting.TContext) {
	count := 1
	if c.Count != nil {
		count = *c.Count
	}
	for index := 0; index < count; index++ {
		env := map[string]any{"Index": index, "Count": count}
		for k, v := range c.TemplateParams {
			env[k] = v
		}
		c.create(tCtx, env)
	}
}

func (c *createAny) create(tCtx ktesting.TContext, env map[string]any) {
	var obj *unstructured.Unstructured
	if err := getSpecFromTextTemplateFile(c.TemplatePath, env, &obj); err != nil {
		tCtx.Fatalf("%s: parsing failed: %v", c.TemplatePath, err)
	}

	// Not caching the discovery result isn't very efficient, but good enough when
	// createAny isn't done often.
	mapping, err := restMappingFromUnstructuredObj(tCtx, obj)
	if err != nil {
		tCtx.Fatalf("%s: %v", c.TemplatePath, err)
	}
	resourceClient := tCtx.Dynamic().Resource(mapping.Resource)

	create := func() error {
		options := metav1.CreateOptions{
			// If the YAML input is invalid, then we want the
			// apiserver to tell us via an error. This can
			// happen because decoding into an unstructured object
			// doesn't validate.
			FieldValidation: "Strict",
		}
		if c.Namespace != "" {
			if mapping.Scope.Name() != meta.RESTScopeNameNamespace {
				return fmt.Errorf("namespace %q set for %q, but %q has scope %q", c.Namespace, c.TemplatePath, mapping.GroupVersionKind, mapping.Scope.Name())
			}
			_, err = resourceClient.Namespace(c.Namespace).Create(tCtx, obj, options)
		} else {
			if mapping.Scope.Name() != meta.RESTScopeNameRoot {
				return fmt.Errorf("namespace not set for %q, but %q has scope %q", c.TemplatePath, mapping.GroupVersionKind, mapping.Scope.Name())
			}
			_, err = resourceClient.Create(tCtx, obj, options)
		}
		return err
	}
	// Retry, some errors (like CRD just created and type not ready for use yet) are temporary.
	ctx, cancel := context.WithTimeout(tCtx, 20*time.Second)
	defer cancel()
	for {
		err := create()
		if err == nil {
			return
		}
		select {
		case <-ctx.Done():
			tCtx.Fatalf("%s: timed out (%q) while creating %q, last error was: %v", c.TemplatePath, context.Cause(ctx), klog.KObj(obj), err)
		case <-time.After(time.Second):
		}
	}
}

func getSpecFromTextTemplateFile(path string, env map[string]any, spec interface{}) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	tmpl, err := template.New("object").Funcs(getTemplateFuncs()).Parse(string(content))
	if err != nil {
		return err
	}
	var buffer bytes.Buffer
	if err := tmpl.Execute(&buffer, env); err != nil {
		return err
	}

	return yaml.UnmarshalStrict(buffer.Bytes(), spec)
}

func restMappingFromUnstructuredObj(tCtx ktesting.TContext, obj *unstructured.Unstructured) (*meta.RESTMapping, error) {
	discoveryCache := memory.NewMemCacheClient(tCtx.Client().Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryCache)
	gv, err := schema.ParseGroupVersion(obj.GetAPIVersion())
	if err != nil {
		return nil, fmt.Errorf("extract group+version from object %q: %w", klog.KObj(obj), err)
	}
	gk := schema.GroupKind{Group: gv.Group, Kind: obj.GetKind()}

	mapping, err := restMapper.RESTMapping(gk, gv.Version)
	if err != nil {
		// Cached mapping might be stale, refresh on next try.
		restMapper.Reset()
		return nil, fmt.Errorf("failed mapping %q to resource: %w", gk, err)
	}
	return mapping, nil
}

// createNodesOp defines an op where nodes are created as a part of a workload.
type createNodesOp struct {
	// Must be "createNodes".
	Opcode operationCode
	// Number of nodes to create. Parameterizable through CountParam.
	Count int
	// Template parameter for Count.
	CountParam string
	// Path to spec file describing the nodes to create.
	// Optional
	NodeTemplatePath *string
	// At most one of the following strategies can be defined. Defaults
	// to TrivialNodePrepareStrategy if unspecified.
	// Optional
	NodeAllocatableStrategy  *testutils.NodeAllocatableStrategy
	LabelNodePrepareStrategy *testutils.LabelNodePrepareStrategy
	UniqueNodeLabelStrategy  *testutils.UniqueNodeLabelStrategy
	// Params to be passed to the template.
	// Values with `$` prefix will be resolved to the workload parameters.
	TemplateParams map[string]any
}

func (cno *createNodesOp) isValid(allowParameterization bool) error {
	if !isValidCount(allowParameterization, cno.Count, cno.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", cno.Count, cno.CountParam)
	}
	return nil
}

func (*createNodesOp) collectsMetrics() bool {
	return false
}

func (cno createNodesOp) patchParams(w *workload) (realOp, error) {
	if cno.CountParam != "" {
		var err error
		cno.Count, err = w.Params.get(cno.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	if len(cno.TemplateParams) > 0 {
		var err error
		cno.TemplateParams, err = resolveTemplateParams(cno.TemplateParams, w)
		if err != nil {
			return nil, err
		}
	}
	return &cno, (&cno).isValid(false)
}

// createNamespacesOp defines an op for creating namespaces
type createNamespacesOp struct {
	// Must be "createNamespaces".
	Opcode operationCode
	// Name prefix of the Namespace. The format is "<prefix>-<number>", where number is
	// between 0 and count-1.
	Prefix string
	// Number of namespaces to create. Parameterizable through CountParam.
	Count int
	// Template parameter for Count. Takes precedence over Count if both set.
	CountParam string
	// Path to spec file describing the Namespaces to create.
	// Optional
	NamespaceTemplatePath *string
}

func (cmo *createNamespacesOp) isValid(allowParameterization bool) error {
	if !isValidCount(allowParameterization, cmo.Count, cmo.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", cmo.Count, cmo.CountParam)
	}
	return nil
}

func (*createNamespacesOp) collectsMetrics() bool {
	return false
}

func (cmo createNamespacesOp) patchParams(w *workload) (realOp, error) {
	if cmo.CountParam != "" {
		var err error
		cmo.Count, err = w.Params.get(cmo.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return &cmo, (&cmo).isValid(false)
}

// createPodsOp defines an op where pods are scheduled as a part of a workload.
// The test can block on the completion of this op before moving forward or
// continue asynchronously.
type createPodsOp struct {
	// Must be "createPods".
	Opcode operationCode
	// Number of pods to schedule. Parameterizable through CountParam.
	Count int
	// Template parameter for Count.
	CountParam string
	// If false, Count pods get created rapidly. This can be used to
	// measure how quickly the scheduler can fill up a cluster.
	//
	// If true, Count pods get created, the operation waits for
	// a pod to get scheduled, deletes it and then creates another.
	// This continues until the configured Duration is over.
	// Metrics collection, if enabled, runs in parallel.
	//
	// This mode can be used to measure how the scheduler behaves
	// in a steady state where the cluster is always at roughly the
	// same level of utilization. Pods can be created in a separate,
	// earlier operation to simulate non-empty clusters.
	//
	// Note that the operation will delete any scheduled pod in
	// the namespace, so use different namespaces for pods that
	// are supposed to be kept running.
	SteadyState bool
	// Template parameter for SteadyState.
	SteadyStateParam string
	// How long to keep the cluster in a steady state.
	Duration metav1.Duration
	// Template parameter for Duration.
	DurationParam string
	// Whether to enable metrics collection for this createPodsOp.
	// Optional. Both CollectMetrics and SkipWaitToCompletion cannot be true at
	// the same time for a particular createPodsOp.
	CollectMetrics bool
	// Namespace the pods should be created in. Defaults to a unique
	// namespace of the format "namespace-<number>".
	// Optional
	Namespace *string
	// Path to spec file describing the pods to schedule.
	// If nil, DefaultPodTemplatePath will be used.
	// Optional
	PodTemplatePath *string
	// Whether to wait for all pods in this op to get scheduled.
	// Defaults to false if not specified.
	// Optional
	SkipWaitToCompletion bool
	// Persistent volume settings for the pods to be scheduled.
	// Optional
	PersistentVolumeTemplatePath      *string
	PersistentVolumeClaimTemplatePath *string
	// Params to be passed to the template.
	// Values with `$` prefix will be resolved to the workload parameters.
	TemplateParams map[string]any
}

func (cpo *createPodsOp) isValid(allowParameterization bool) error {
	if !isValidCount(allowParameterization, cpo.Count, cpo.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", cpo.Count, cpo.CountParam)
	}
	if cpo.CollectMetrics && cpo.SkipWaitToCompletion {
		// While it's technically possible to achieve this, the additional
		// complexity is not worth it, especially given that we don't have any
		// use-cases right now.
		return fmt.Errorf("collectMetrics and skipWaitToCompletion cannot be true at the same time")
	}
	if cpo.SkipWaitToCompletion && cpo.SteadyState {
		return errors.New("skipWaitToCompletion and steadyState cannot be true at the same time")
	}
	if cpo.SteadyState && !allowParameterization && cpo.Duration.Duration <= 0 {
		return errors.New("whfen creating pods in a steady state, the test duration must be > 0")
	}
	return nil
}

func (cpo *createPodsOp) collectsMetrics() bool {
	return cpo.CollectMetrics
}

func (cpo createPodsOp) patchParams(w *workload) (realOp, error) {
	if cpo.CountParam != "" {
		var err error
		cpo.Count, err = w.Params.get(cpo.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	if cpo.DurationParam != "" {
		durationStr, err := getParam[string](w.Params, cpo.DurationParam[1:])
		if err != nil {
			return nil, err
		}
		if cpo.Duration.Duration, err = time.ParseDuration(durationStr); err != nil {
			return nil, fmt.Errorf("parsing duration parameter %s: %w", cpo.DurationParam, err)
		}
	}
	if cpo.SteadyStateParam != "" {
		var err error
		cpo.SteadyState, err = getParam[bool](w.Params, cpo.SteadyStateParam[1:])
		if err != nil {
			return nil, err
		}
	}
	if len(cpo.TemplateParams) > 0 {
		var err error
		cpo.TemplateParams, err = resolveTemplateParams(cpo.TemplateParams, w)
		if err != nil {
			return nil, err
		}
	}
	return &cpo, (&cpo).isValid(false)
}

// createPodSetsOp defines an op where a set of createPodsOps is created in each unique namespace.
type createPodSetsOp struct {
	// Must be "createPodSets".
	Opcode operationCode
	// Number of sets to create.
	Count int
	// Template parameter for Count.
	CountParam string
	// Each set of pods will be created in a namespace of the form namespacePrefix-<number>,
	// where number is from 0 to count-1
	NamespacePrefix string
	// The template of a createPodsOp.
	CreatePodsOp createPodsOp
}

func (cpso *createPodSetsOp) isValid(allowParameterization bool) error {
	if !isValidCount(allowParameterization, cpso.Count, cpso.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", cpso.Count, cpso.CountParam)
	}
	return cpso.CreatePodsOp.isValid(allowParameterization)
}

func (cpso *createPodSetsOp) collectsMetrics() bool {
	return cpso.CreatePodsOp.CollectMetrics
}

func (cpso createPodSetsOp) patchParams(w *workload) (realOp, error) {
	if cpso.CountParam != "" {
		var err error
		cpso.Count, err = w.Params.get(cpso.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return &cpso, (&cpso).isValid(true)
}

// deletePodsOp defines an op where previously created pods are deleted.
// The test can block on the completion of this op before moving forward or
// continue asynchronously.
type deletePodsOp struct {
	// Must be "deletePods".
	Opcode operationCode
	// Namespace the pods should be deleted from.
	Namespace string
	// Labels used to filter the pods to delete.
	// If empty, it will delete all Pods in the namespace.
	// Optional.
	LabelSelector map[string]string
	// Whether to wait for all pods in this op to be deleted.
	// Defaults to false if not specified.
	// Optional
	SkipWaitToCompletion bool
	// Number of pods to be deleted per second.
	// If zero, all pods are deleted at once.
	// Optional
	DeletePodsPerSecond int
}

func (dpo *deletePodsOp) isValid(allowParameterization bool) error {
	if dpo.Opcode != deletePodsOpcode {
		return fmt.Errorf("invalid opcode %q; expected %q", dpo.Opcode, deletePodsOpcode)
	}
	if dpo.DeletePodsPerSecond < 0 {
		return fmt.Errorf("invalid DeletePodsPerSecond=%d; should be non-negative", dpo.DeletePodsPerSecond)
	}
	return nil
}

func (dpo *deletePodsOp) collectsMetrics() bool {
	return false
}

func (dpo deletePodsOp) patchParams(w *workload) (realOp, error) {
	return &dpo, nil
}

// churnOp defines an op where services are created as a part of a workload.
type churnOp struct {
	// Must be "churnOp".
	Opcode operationCode
	// Value must be one of the followings:
	// - recreate. In this mode, API objects will be created for N cycles, and then
	//   deleted in the next N cycles. N is specified by the "Number" field.
	// - create. In this mode, API objects will be created (without deletion) until
	//   reaching a threshold - which is specified by the "Number" field.
	Mode string
	// Maximum number of API objects to be created.
	// Defaults to 0, which means unlimited.
	Number int
	// Intervals of churning. Defaults to 500 millisecond.
	IntervalMilliseconds int64
	// Namespace the churning objects should be created in. Defaults to a unique
	// namespace of the format "namespace-<number>".
	// Optional
	Namespace *string
	// Path of API spec files.
	TemplatePaths []string
}

func (co *churnOp) isValid(_ bool) error {
	if co.Mode != Recreate && co.Mode != Create {
		return fmt.Errorf("invalid mode: %v. must be one of %v", co.Mode, []string{Recreate, Create})
	}
	if co.Number < 0 {
		return fmt.Errorf("number (%v) cannot be negative", co.Number)
	}
	if co.Mode == Recreate && co.Number == 0 {
		return fmt.Errorf("number cannot be 0 when mode is %v", Recreate)
	}
	if len(co.TemplatePaths) == 0 {
		return fmt.Errorf("at least one template spec file needs to be specified")
	}
	return nil
}

func (*churnOp) collectsMetrics() bool {
	return false
}

func (co churnOp) patchParams(w *workload) (realOp, error) {
	return &co, nil
}

type SchedulingStage string

const (
	Scheduled SchedulingStage = "Scheduled"
	Attempted SchedulingStage = "Attempted"
)

// barrierOp defines an op that can be used to wait until all scheduled pods of
// one or many namespaces have been bound to nodes. This is useful when pods
// were scheduled with SkipWaitToCompletion set to true.
type barrierOp struct {
	// Must be "barrier".
	Opcode operationCode
	// Namespaces to block on. Empty array or not specifying this field signifies
	// that the barrier should block on all namespaces.
	Namespaces []string
	// Labels used to filter the pods to block on.
	// If empty, it won't filter the labels.
	// Optional.
	LabelSelector map[string]string
	// Determines what stage of pods scheduling the barrier should wait for.
	// If empty, it is interpreted as "Scheduled".
	// Optional
	StageRequirement SchedulingStage
}

func (bo *barrierOp) isValid(allowParameterization bool) error {
	if bo.StageRequirement != "" && bo.StageRequirement != Scheduled && bo.StageRequirement != Attempted {
		return fmt.Errorf("invalid StageRequirement %s", bo.StageRequirement)
	}
	return nil
}

func (*barrierOp) collectsMetrics() bool {
	return false
}

func (bo barrierOp) patchParams(w *workload) (realOp, error) {
	if bo.StageRequirement == "" {
		bo.StageRequirement = Scheduled
	}
	return &bo, nil
}

// sleepOp defines an op that can be used to sleep for a specified amount of time.
// This is useful in simulating workloads that require some sort of time-based synchronisation.
type sleepOp struct {
	// Must be "sleep".
	Opcode operationCode
	// Duration of sleep.
	Duration metav1.Duration
	// Template parameter for Duration.
	DurationParam string
}

func (so *sleepOp) isValid(_ bool) error {
	return nil
}

func (so *sleepOp) collectsMetrics() bool {
	return false
}

func (so sleepOp) patchParams(w *workload) (realOp, error) {
	if so.DurationParam != "" {
		durationStr, err := getParam[string](w.Params, so.DurationParam[1:])
		if err != nil {
			return nil, err
		}
		if so.Duration.Duration, err = time.ParseDuration(durationStr); err != nil {
			return nil, fmt.Errorf("invalid duration parameter %s: %w", so.DurationParam, err)
		}
	}
	return &so, nil
}

// startCollectingMetricsOp defines an op that starts metrics collectors.
// stopCollectingMetricsOp has to be used after this op to finish collecting.
type startCollectingMetricsOp struct {
	// Must be "startCollectingMetrics".
	Opcode operationCode
	// Name appended to workload's name in results.
	Name string
	// Namespaces for which the scheduling throughput metric is calculated.
	Namespaces []string
	// Labels used to filter the pods for which the scheduling throughput metric is collected.
	// If empty, it will collect the metric for all pods in the selected namespaces.
	// Optional.
	LabelSelector map[string]string
	// List of collectors to use. If empty, defaults to:
	// "Throughput", "Metrics", "Memory", "SchedulingDuration".
	// Optional.
	Collectors []string
}

func (scm *startCollectingMetricsOp) isValid(_ bool) error {
	if len(scm.Namespaces) == 0 {
		return fmt.Errorf("namespaces cannot be empty")
	}
	return nil
}

func (*startCollectingMetricsOp) collectsMetrics() bool {
	return false
}

func (scm startCollectingMetricsOp) patchParams(_ *workload) (realOp, error) {
	return &scm, nil
}

// stopCollectingMetricsOp defines an op that stops collecting the metrics
// and writes them into the result slice.
// startCollectingMetricsOp has be used before this op to begin collecting.
type stopCollectingMetricsOp struct {
	// Must be "stopCollectingMetrics".
	Opcode operationCode
}

func (scm *stopCollectingMetricsOp) isValid(_ bool) error {
	return nil
}

func (*stopCollectingMetricsOp) collectsMetrics() bool {
	return true
}

func (scm stopCollectingMetricsOp) patchParams(_ *workload) (realOp, error) {
	return &scm, nil
}

// resolveTemplateParams resolves the template parameters using the workload parameters.
func resolveTemplateParams(templateParams map[string]any, w *workload) (map[string]any, error) {
	if len(templateParams) == 0 {
		return templateParams, nil
	}
	resolved := maps.Clone(templateParams)
	for k, v := range resolved {
		if s, ok := v.(string); ok && strings.HasPrefix(s, "$") {
			paramKey := s[1:]
			if val, found := w.Params.params[paramKey]; found {
				w.Params.isUsed[paramKey] = true
				resolved[k] = val
				continue
			}
			return nil, fmt.Errorf("parameter %q not found", paramKey)
		}
	}
	return resolved, nil
}

func getTemplateFuncs() template.FuncMap {
	return template.FuncMap{
		"AddFloat":      addFloat,
		"AddInt":        addInt,
		"DivideFloat":   divideFloat,
		"DivideInt":     divideInt,
		"Mod":           mod,
		"MultiplyFloat": multiplyFloat,
		"MultiplyInt":   multiplyInt,
		"SubtractFloat": subtractFloat,
		"SubtractInt":   subtractInt,
	}
}

func toFloat64(val any) float64 {
	switch i := val.(type) {
	case float64:
		return i
	case float32:
		return float64(i)
	case int64:
		return float64(i)
	case int32:
		return float64(i)
	case int:
		return float64(i)
	case uint64:
		return float64(i)
	case uint32:
		return float64(i)
	case uint:
		return float64(i)
	case string:
		f, err := strconv.ParseFloat(i, 64)
		if err == nil {
			return f
		}
	}
	panic(fmt.Sprintf("cannot cast %v to float64", val))
}

func addInt(numbers ...any) int {
	return int(addFloat(numbers...))
}

func subtractInt(i, j any) int {
	return int(subtractFloat(i, j))
}

func multiplyInt(numbers ...any) int {
	return int(multiplyFloat(numbers...))
}

func divideInt(i, j any) int {
	return int(divideFloat(i, j))
}

func addFloat(numbers ...any) float64 {
	sum := 0.0
	for _, number := range numbers {
		sum += toFloat64(number)
	}
	return sum
}

func subtractFloat(i, j any) float64 {
	typedI := toFloat64(i)
	typedJ := toFloat64(j)
	return typedI - typedJ
}

func multiplyFloat(numbers ...any) float64 {
	product := 1.0
	for _, number := range numbers {
		product *= toFloat64(number)
	}
	return product
}

func divideFloat(i, j any) float64 {
	typedI := toFloat64(i)
	typedJ := toFloat64(j)
	if typedJ == 0 {
		panic("division by zero")
	}
	return typedI / typedJ
}

func mod(a, b any) int {
	return int(toFloat64(a)) % int(toFloat64(b))
}
