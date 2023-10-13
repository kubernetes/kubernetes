/*
Copyright 2022 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"path"
	"reflect"
	"regexp"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

// Feature is the name of a certain feature that the cluster under test must have.
// Such features are different from feature gates.
type Feature string

// Environment is the name for the environment in which a test can run, like
// "Linux" or "Windows".
type Environment string

// NodeFeature is the name of a feature that a node must support. To be
// removed, see
// https://github.com/kubernetes/enhancements/tree/master/keps/sig-testing/3041-node-conformance-and-features#nodefeature.
type NodeFeature string

type Valid[T comparable] struct {
	items  sets.Set[T]
	frozen bool
}

// Add registers a new valid item name. The expected usage is
//
//	var SomeFeature = framework.ValidFeatures.Add("Some")
//
// during the init phase of an E2E suite. Individual tests should not register
// their own, to avoid uncontrolled proliferation of new items. E2E suites can,
// but don't have to, enforce that by freezing the set of valid names.
func (v *Valid[T]) Add(item T) T {
	if v.frozen {
		RecordBug(NewBug(fmt.Sprintf(`registry %T is already frozen, "%v" must not be added anymore`, *v, item), 1))
	}
	if v.items == nil {
		v.items = sets.New[T]()
	}
	if v.items.Has(item) {
		RecordBug(NewBug(fmt.Sprintf(`registry %T already contains "%v", it must not be added again`, *v, item), 1))
	}
	v.items.Insert(item)
	return item
}

func (v *Valid[T]) Freeze() {
	v.frozen = true
}

// These variables contain the parameters that [WithFeature], [WithEnvironment]
// and [WithNodeFeatures] accept. The framework itself has no pre-defined
// constants. Test suites and tests may define their own and then add them here
// before calling these With functions.
var (
	ValidFeatures     Valid[Feature]
	ValidEnvironments Valid[Environment]
	ValidNodeFeatures Valid[NodeFeature]
)

var errInterface = reflect.TypeOf((*error)(nil)).Elem()

// IgnoreNotFound can be used to wrap an arbitrary function in a call to
// [ginkgo.DeferCleanup]. When the wrapped function returns an error that
// `apierrors.IsNotFound` considers as "not found", the error is ignored
// instead of failing the test during cleanup. This is useful for cleanup code
// that just needs to ensure that some object does not exist anymore.
func IgnoreNotFound(in any) any {
	inType := reflect.TypeOf(in)
	inValue := reflect.ValueOf(in)
	return reflect.MakeFunc(inType, func(args []reflect.Value) []reflect.Value {
		out := inValue.Call(args)
		if len(out) > 0 {
			lastValue := out[len(out)-1]
			last := lastValue.Interface()
			if last != nil && lastValue.Type().Implements(errInterface) && apierrors.IsNotFound(last.(error)) {
				out[len(out)-1] = reflect.Zero(errInterface)
			}
		}
		return out
	}).Interface()
}

// AnnotatedLocation can be used to provide more informative source code
// locations by passing the result as additional parameter to a
// BeforeEach/AfterEach/DeferCleanup/It/etc.
func AnnotatedLocation(annotation string) types.CodeLocation {
	return AnnotatedLocationWithOffset(annotation, 1)
}

// AnnotatedLocationWithOffset skips additional call stack levels. With 0 as offset
// it is identical to [AnnotatedLocation].
func AnnotatedLocationWithOffset(annotation string, offset int) types.CodeLocation {
	codeLocation := types.NewCodeLocation(offset + 1)
	codeLocation.FileName = path.Base(codeLocation.FileName)
	codeLocation = types.NewCustomCodeLocation(annotation + " | " + codeLocation.String())
	return codeLocation
}

// SIGDescribe returns a wrapper function for ginkgo.Describe which injects
// the SIG name as annotation. The parameter should be lowercase with
// no spaces and no sig- or SIG- prefix.
func SIGDescribe(sig string) func(string, ...interface{}) bool {
	if !sigRE.MatchString(sig) || strings.HasPrefix(sig, "sig-") {
		panic(fmt.Sprintf("SIG label must be lowercase, no spaces and no sig- prefix, got instead: %q", sig))
	}
	return func(text string, args ...interface{}) bool {
		args = append(args, ginkgo.Label("sig-"+sig))
		if text == "" {
			text = fmt.Sprintf("[sig-%s]", sig)
		} else {
			text = fmt.Sprintf("[sig-%s] %s", sig, text)
		}
		return registerInSuite(ginkgo.Describe, text, args)
	}
}

var sigRE = regexp.MustCompile(`^[a-z]+(-[a-z]+)*$`)

// ConformanceIt is wrapper function for ginkgo It.  Adds "[Conformance]" tag and makes static analysis easier.
func ConformanceIt(text string, args ...interface{}) bool {
	args = append(args, ginkgo.Offset(1), WithConformance())
	return It(text, args...)
}

// It is a wrapper around [ginkgo.It] which supports framework With* labels as
// optional arguments in addition to those already supported by ginkgo itself,
// like [ginkgo.Label] and [gingko.Offset].
//
// Text and arguments may be mixed. The final text is a concatenation
// of the text arguments and special tags from the With functions.
func It(text string, args ...interface{}) bool {
	return registerInSuite(ginkgo.It, text, args)
}

// It is a shorthand for the corresponding package function.
func (f *Framework) It(text string, args ...interface{}) bool {
	return registerInSuite(ginkgo.It, text, args)
}

// Describe is a wrapper around [ginkgo.Describe] which supports framework
// With* labels as optional arguments in addition to those already supported by
// ginkgo itself, like [ginkgo.Label] and [gingko.Offset].
//
// Text and arguments may be mixed. The final text is a concatenation
// of the text arguments and special tags from the With functions.
func Describe(text string, args ...interface{}) bool {
	return registerInSuite(ginkgo.Describe, text, args)
}

// Describe is a shorthand for the corresponding package function.
func (f *Framework) Describe(text string, args ...interface{}) bool {
	return registerInSuite(ginkgo.Describe, text, args)
}

// Context is a wrapper around [ginkgo.Context] which supports framework With*
// labels as optional arguments in addition to those already supported by
// ginkgo itself, like [ginkgo.Label] and [gingko.Offset].
//
// Text and arguments may be mixed. The final text is a concatenation
// of the text arguments and special tags from the With functions.
func Context(text string, args ...interface{}) bool {
	return registerInSuite(ginkgo.Context, text, args)
}

// Context is a shorthand for the corresponding package function.
func (f *Framework) Context(text string, args ...interface{}) bool {
	return registerInSuite(ginkgo.Context, text, args)
}

// registerInSuite is the common implementation of all wrapper functions. It
// expects to be called through one intermediate wrapper.
func registerInSuite(ginkgoCall func(text string, args ...interface{}) bool, text string, args []interface{}) bool {
	var ginkgoArgs []interface{}
	var offset ginkgo.Offset
	var texts []string
	if text != "" {
		texts = append(texts, text)
	}

	addLabel := func(label string) {
		texts = append(texts, fmt.Sprintf("[%s]", label))
		ginkgoArgs = append(ginkgoArgs, ginkgo.Label(label))
	}

	haveEmptyStrings := false
	for _, arg := range args {
		switch arg := arg.(type) {
		case label:
			fullLabel := strings.Join(arg.parts, ": ")
			addLabel(fullLabel)
			if arg.extra != "" {
				addLabel(arg.extra)
			}
			if fullLabel == "Serial" {
				ginkgoArgs = append(ginkgoArgs, ginkgo.Serial)
			}
		case ginkgo.Offset:
			offset = arg
		case string:
			if arg == "" {
				haveEmptyStrings = true
			}
			texts = append(texts, arg)
		default:
			ginkgoArgs = append(ginkgoArgs, arg)
		}
	}
	offset += 2 // This function and its direct caller.

	// Now that we have the final offset, we can record bugs.
	if haveEmptyStrings {
		RecordBug(NewBug("empty strings as separators are unnecessary and need to be removed", int(offset)))
	}

	// Enforce that text snippets to not start or end with spaces because
	// those lead to double spaces when concatenating below.
	for _, text := range texts {
		if strings.HasPrefix(text, " ") || strings.HasSuffix(text, " ") {
			RecordBug(NewBug(fmt.Sprintf("trailing or leading spaces are unnecessary and need to be removed: %q", text), int(offset)))
		}
	}

	ginkgoArgs = append(ginkgoArgs, offset)
	text = strings.Join(texts, " ")
	return ginkgoCall(text, ginkgoArgs...)
}

// WithEnvironment specifies that a certain test or group of tests only works
// with a feature available. The return value must be passed as additional
// argument to [framework.It], [framework.Describe], [framework.Context].
//
// The feature must be listed in ValidFeatures.
func WithFeature(name Feature) interface{} {
	return withFeature(name)
}

// WithFeature is a shorthand for the corresponding package function.
func (f *Framework) WithFeature(name Feature) interface{} {
	return withFeature(name)
}

func withFeature(name Feature) interface{} {
	if !ValidFeatures.items.Has(name) {
		RecordBug(NewBug(fmt.Sprintf("WithFeature: unknown feature %q", name), 2))
	}
	return newLabel("Feature", string(name))
}

// WithFeatureGate specifies that a certain test or group of tests depends on a
// feature gate being enabled. The return value must be passed as additional
// argument to [framework.It], [framework.Describe], [framework.Context].
//
// The feature gate must be listed in
// [k8s.io/apiserver/pkg/util/feature.DefaultMutableFeatureGate]. Once a
// feature gate gets removed from there, the WithFeatureGate calls using it
// also need to be removed.
func WithFeatureGate(featureGate featuregate.Feature) interface{} {
	return withFeatureGate(featureGate)
}

// WithFeatureGate is a shorthand for the corresponding package function.
func (f *Framework) WithFeatureGate(featureGate featuregate.Feature) interface{} {
	return withFeatureGate(featureGate)
}

func withFeatureGate(featureGate featuregate.Feature) interface{} {
	spec, ok := utilfeature.DefaultMutableFeatureGate.GetAll()[featureGate]
	if !ok {
		RecordBug(NewBug(fmt.Sprintf("WithFeatureGate: the feature gate %q is unknown", featureGate), 2))
	}

	// We use mixed case (i.e. Beta instead of BETA). GA feature gates have no level string.
	var level string
	if spec.PreRelease != "" {
		level = string(spec.PreRelease)
		level = strings.ToUpper(level[0:1]) + strings.ToLower(level[1:])
	}

	l := newLabel("FeatureGate", string(featureGate))
	l.extra = level
	return l
}

// WithEnvironment specifies that a certain test or group of tests only works
// in a certain environment. The return value must be passed as additional
// argument to [framework.It], [framework.Describe], [framework.Context].
//
// The environment must be listed in ValidEnvironments.
func WithEnvironment(name Environment) interface{} {
	return withEnvironment(name)
}

// WithEnvironment is a shorthand for the corresponding package function.
func (f *Framework) WithEnvironment(name Environment) interface{} {
	return withEnvironment(name)
}

func withEnvironment(name Environment) interface{} {
	if !ValidEnvironments.items.Has(name) {
		RecordBug(NewBug(fmt.Sprintf("WithEnvironment: unknown environment %q", name), 2))
	}
	return newLabel("Environment", string(name))
}

// WithNodeFeature specifies that a certain test or group of tests only works
// if the node supports a certain feature. The return value must be passed as
// additional argument to [framework.It], [framework.Describe],
// [framework.Context].
//
// The environment must be listed in ValidNodeFeatures.
func WithNodeFeature(name NodeFeature) interface{} {
	return withNodeFeature(name)
}

// WithNodeFeature is a shorthand for the corresponding package function.
func (f *Framework) WithNodeFeature(name NodeFeature) interface{} {
	return withNodeFeature(name)
}

func withNodeFeature(name NodeFeature) interface{} {
	if !ValidNodeFeatures.items.Has(name) {
		RecordBug(NewBug(fmt.Sprintf("WithNodeFeature: unknown environment %q", name), 2))
	}
	return newLabel(string(name))
}

// WithConformace specifies that a certain test or group of tests must pass in
// all conformant Kubernetes clusters. The return value must be passed as
// additional argument to [framework.It], [framework.Describe],
// [framework.Context].
func WithConformance() interface{} {
	return withConformance()
}

// WithConformance is a shorthand for the corresponding package function.
func (f *Framework) WithConformance() interface{} {
	return withConformance()
}

func withConformance() interface{} {
	return newLabel("Conformance")
}

// WithNodeConformance specifies that a certain test or group of tests for node
// functionality that does not depend on runtime or Kubernetes distro specific
// behavior. The return value must be passed as additional argument to
// [framework.It], [framework.Describe], [framework.Context].
func WithNodeConformance() interface{} {
	return withNodeConformance()
}

// WithNodeConformance is a shorthand for the corresponding package function.
func (f *Framework) WithNodeConformance() interface{} {
	return withNodeConformance()
}

func withNodeConformance() interface{} {
	return newLabel("NodeConformance")
}

// WithDisruptive specifies that a certain test or group of tests temporarily
// affects the functionality of the Kubernetes cluster. The return value must
// be passed as additional argument to [framework.It], [framework.Describe],
// [framework.Context].
func WithDisruptive() interface{} {
	return withDisruptive()
}

// WithDisruptive is a shorthand for the corresponding package function.
func (f *Framework) WithDisruptive() interface{} {
	return withDisruptive()
}

func withDisruptive() interface{} {
	return newLabel("Disruptive")
}

// WithSerial specifies that a certain test or group of tests must not run in
// parallel with other tests. The return value must be passed as additional
// argument to [framework.It], [framework.Describe], [framework.Context].
//
// Starting with ginkgo v2, serial and parallel tests can be executed in the
// same invocation. Ginkgo itself will ensure that the serial tests run
// sequentially.
func WithSerial() interface{} {
	return withSerial()
}

// WithSerial is a shorthand for the corresponding package function.
func (f *Framework) WithSerial() interface{} {
	return withSerial()
}

func withSerial() interface{} {
	return newLabel("Serial")
}

// WithSlow specifies that a certain test or group of tests must not run in
// parallel with other tests. The return value must be passed as additional
// argument to [framework.It], [framework.Describe], [framework.Context].
func WithSlow() interface{} {
	return withSlow()
}

// WithSlow is a shorthand for the corresponding package function.
func (f *Framework) WithSlow() interface{} {
	return WithSlow()
}

func withSlow() interface{} {
	return newLabel("Slow")
}

// WithLabel is a wrapper around [ginkgo.Label]. Besides adding an arbitrary
// label to a test, it also injects the label in square brackets into the test
// name.
func WithLabel(label string) interface{} {
	return withLabel(label)
}

// WithLabel is a shorthand for the corresponding package function.
func (f *Framework) WithLabel(label string) interface{} {
	return withLabel(label)
}

func withLabel(label string) interface{} {
	return newLabel(label)
}

type label struct {
	// parts get concatenated with ": " to build the full label.
	parts []string
	// extra is an optional fully-formed extra label.
	extra string
}

func newLabel(parts ...string) label {
	return label{parts: parts}
}
