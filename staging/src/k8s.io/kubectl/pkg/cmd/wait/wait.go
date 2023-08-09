/*
Copyright 2018 The Kubernetes Authors.

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

package wait

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"time"

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/client-go/util/jsonpath"
	cmdget "k8s.io/kubectl/pkg/cmd/get"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/interrupt"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	waitLong = templates.LongDesc(i18n.T(`
		Experimental: Wait for a specific condition on one or many resources.

		The command takes multiple resources and waits until the specified condition
		is seen in the Status field of every given resource.

		Alternatively, the command can wait for the given set of resources to be deleted
		by providing the "delete" keyword as the value to the --for flag.

		A successful message will be printed to stdout indicating when the specified
        condition has been met. You can use -o option to change to output destination.`))

	waitExample = templates.Examples(i18n.T(`
		# Wait for the pod "busybox1" to contain the status condition of type "Ready"
		kubectl wait --for=condition=Ready pod/busybox1

		# The default value of status condition is true; you can wait for other targets after an equal delimiter (compared after Unicode simple case folding, which is a more general form of case-insensitivity):
		kubectl wait --for=condition=Ready=false pod/busybox1

		# Wait for the pod "busybox1" to contain the status phase to be "Running".
		kubectl wait --for=jsonpath='{.status.phase}'=Running pod/busybox1

		# Wait for the pod "busybox1" to be deleted, with a timeout of 60s, after having issued the "delete" command
		kubectl delete pod/busybox1
		kubectl wait --for=delete pod/busybox1 --timeout=60s`))
)

// errNoMatchingResources is returned when there is no resources matching a query.
var errNoMatchingResources = errors.New("no matching resources found")

// WaitFlags directly reflect the information that CLI is gathering via flags.  They will be converted to Options, which
// reflect the runtime requirements for the command.  This structure reduces the transformation to wiring and makes
// the logic itself easy to unit test
type WaitFlags struct {
	RESTClientGetter     genericclioptions.RESTClientGetter
	PrintFlags           *genericclioptions.PrintFlags
	ResourceBuilderFlags *genericclioptions.ResourceBuilderFlags

	Timeout      time.Duration
	ForCondition string

	genericclioptions.IOStreams
}

// NewWaitFlags returns a default WaitFlags
func NewWaitFlags(restClientGetter genericclioptions.RESTClientGetter, streams genericclioptions.IOStreams) *WaitFlags {
	return &WaitFlags{
		RESTClientGetter: restClientGetter,
		PrintFlags:       genericclioptions.NewPrintFlags("condition met"),
		ResourceBuilderFlags: genericclioptions.NewResourceBuilderFlags().
			WithLabelSelector("").
			WithFieldSelector("").
			WithAll(false).
			WithAllNamespaces(false).
			WithLocal(false).
			WithLatest(),

		Timeout: 30 * time.Second,

		IOStreams: streams,
	}
}

// NewCmdWait returns a cobra command for waiting
func NewCmdWait(restClientGetter genericclioptions.RESTClientGetter, streams genericclioptions.IOStreams) *cobra.Command {
	flags := NewWaitFlags(restClientGetter, streams)

	cmd := &cobra.Command{
		Use:     "wait ([-f FILENAME] | resource.group/resource.name | resource.group [(-l label | --all)]) [--for=delete|--for condition=available|--for=jsonpath='{}'=value]",
		Short:   i18n.T("Experimental: Wait for a specific condition on one or many resources"),
		Long:    waitLong,
		Example: waitExample,

		DisableFlagsInUseLine: true,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.RunWait())
		},
		SuggestFor: []string{"list", "ps"},
	}

	flags.AddFlags(cmd)

	return cmd
}

// AddFlags registers flags for a cli
func (flags *WaitFlags) AddFlags(cmd *cobra.Command) {
	flags.PrintFlags.AddFlags(cmd)
	flags.ResourceBuilderFlags.AddFlags(cmd.Flags())

	cmd.Flags().DurationVar(&flags.Timeout, "timeout", flags.Timeout, "The length of time to wait before giving up.  Zero means check once and don't wait, negative means wait for a week.")
	cmd.Flags().StringVar(&flags.ForCondition, "for", flags.ForCondition, "The condition to wait on: [delete|condition=condition-name[=condition-value]|jsonpath='{JSONPath expression}'=JSONPath Condition]. The default condition-value is true.  Condition values are compared after Unicode simple case folding, which is a more general form of case-insensitivity.")
}

// ToOptions converts from CLI inputs to runtime inputs
func (flags *WaitFlags) ToOptions(args []string) (*WaitOptions, error) {
	printer, err := flags.PrintFlags.ToPrinter()
	if err != nil {
		return nil, err
	}
	builder := flags.ResourceBuilderFlags.ToBuilder(flags.RESTClientGetter, args)
	clientConfig, err := flags.RESTClientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	dynamicClient, err := dynamic.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	conditionFn, err := conditionFuncFor(flags.ForCondition, flags.ErrOut)
	if err != nil {
		return nil, err
	}

	effectiveTimeout := flags.Timeout
	if effectiveTimeout < 0 {
		effectiveTimeout = 168 * time.Hour
	}

	o := &WaitOptions{
		ResourceFinder: builder,
		DynamicClient:  dynamicClient,
		Timeout:        effectiveTimeout,
		ForCondition:   flags.ForCondition,

		Printer:     printer,
		ConditionFn: conditionFn,
		IOStreams:   flags.IOStreams,
	}

	return o, nil
}

func conditionFuncFor(condition string, errOut io.Writer) (ConditionFunc, error) {
	if strings.ToLower(condition) == "delete" {
		return IsDeleted, nil
	}
	if strings.HasPrefix(condition, "condition=") {
		conditionName := condition[len("condition="):]
		conditionValue := "true"
		if equalsIndex := strings.Index(conditionName, "="); equalsIndex != -1 {
			conditionValue = conditionName[equalsIndex+1:]
			conditionName = conditionName[0:equalsIndex]
		}

		return ConditionalWait{
			conditionName:   conditionName,
			conditionStatus: conditionValue,
			errOut:          errOut,
		}.IsConditionMet, nil
	}
	if strings.HasPrefix(condition, "jsonpath=") {
		splitStr := strings.Split(condition, "=")
		if len(splitStr) != 3 {
			return nil, fmt.Errorf("jsonpath wait format must be --for=jsonpath='{.status.readyReplicas}'=3")
		}
		jsonPathExp, jsonPathCond, err := processJSONPathInput(splitStr[1], splitStr[2])
		if err != nil {
			return nil, err
		}
		j, err := newJSONPathParser(jsonPathExp)
		if err != nil {
			return nil, err
		}
		return JSONPathWait{
			jsonPathCondition: jsonPathCond,
			jsonPathParser:    j,
			errOut:            errOut,
		}.IsJSONPathConditionMet, nil
	}

	return nil, fmt.Errorf("unrecognized condition: %q", condition)
}

// newJSONPathParser will create a new JSONPath parser based on the jsonPathExpression
func newJSONPathParser(jsonPathExpression string) (*jsonpath.JSONPath, error) {
	j := jsonpath.New("wait").AllowMissingKeys(true)
	if jsonPathExpression == "" {
		return nil, errors.New("jsonpath expression cannot be empty")
	}
	if err := j.Parse(jsonPathExpression); err != nil {
		return nil, err
	}
	return j, nil
}

// processJSONPathInput will parses the user's JSONPath input and process the string
func processJSONPathInput(jsonPathExpression, jsonPathCond string) (string, string, error) {
	relaxedJSONPathExp, err := cmdget.RelaxedJSONPathExpression(jsonPathExpression)
	if err != nil {
		return "", "", err
	}
	if jsonPathCond == "" {
		return "", "", errors.New("jsonpath wait condition cannot be empty")
	}
	jsonPathCond = strings.Trim(jsonPathCond, `'"`)

	return relaxedJSONPathExp, jsonPathCond, nil
}

// ResourceLocation holds the location of a resource
type ResourceLocation struct {
	GroupResource schema.GroupResource
	Namespace     string
	Name          string
}

// UIDMap maps ResourceLocation with UID
type UIDMap map[ResourceLocation]types.UID

// WaitOptions is a set of options that allows you to wait.  This is the object reflects the runtime needs of a wait
// command, making the logic itself easy to unit test with our existing mocks.
type WaitOptions struct {
	ResourceFinder genericclioptions.ResourceFinder
	// UIDMap maps a resource location to a UID.  It is optional, but ConditionFuncs may choose to use it to make the result
	// more reliable.  For instance, delete can look for UID consistency during delegated calls.
	UIDMap        UIDMap
	DynamicClient dynamic.Interface
	Timeout       time.Duration
	ForCondition  string

	Printer     printers.ResourcePrinter
	ConditionFn ConditionFunc
	genericclioptions.IOStreams
}

// ConditionFunc is the interface for providing condition checks
type ConditionFunc func(ctx context.Context, info *resource.Info, o *WaitOptions) (finalObject runtime.Object, done bool, err error)

// RunWait runs the waiting logic
func (o *WaitOptions) RunWait() error {
	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), o.Timeout)
	defer cancel()

	visitCount := 0
	visitFunc := func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		visitCount++
		finalObject, success, err := o.ConditionFn(ctx, info, o)
		if success {
			o.Printer.PrintObj(finalObject, o.Out)
			return nil
		}
		if err == nil {
			return fmt.Errorf("%v unsatisified for unknown reason", finalObject)
		}
		return err
	}
	visitor := o.ResourceFinder.Do()
	isForDelete := strings.ToLower(o.ForCondition) == "delete"
	if visitor, ok := visitor.(*resource.Result); ok && isForDelete {
		visitor.IgnoreErrors(apierrors.IsNotFound)
	}

	err := visitor.Visit(visitFunc)
	if err != nil {
		return err
	}
	if visitCount == 0 && !isForDelete {
		return errNoMatchingResources
	}
	return err
}

// IsDeleted is a condition func for waiting for something to be deleted
func IsDeleted(ctx context.Context, info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	if len(info.Name) == 0 {
		return info.Object, false, fmt.Errorf("resource name must be provided")
	}

	gottenObj, initObjGetErr := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Get(context.Background(), info.Name, metav1.GetOptions{})
	if apierrors.IsNotFound(initObjGetErr) {
		return info.Object, true, nil
	}
	if initObjGetErr != nil {
		// TODO this could do something slightly fancier if we wish
		return info.Object, false, initObjGetErr
	}
	resourceLocation := ResourceLocation{
		GroupResource: info.Mapping.Resource.GroupResource(),
		Namespace:     gottenObj.GetNamespace(),
		Name:          gottenObj.GetName(),
	}
	if uid, ok := o.UIDMap[resourceLocation]; ok {
		if gottenObj.GetUID() != uid {
			return gottenObj, true, nil
		}
	}

	endTime := time.Now().Add(o.Timeout)
	timeout := time.Until(endTime)
	errWaitTimeoutWithName := extendErrWaitTimeout(wait.ErrWaitTimeout, info)
	if o.Timeout == 0 {
		// If timeout is zero check if the object exists once only
		if gottenObj == nil {
			return nil, true, nil
		}
		return gottenObj, false, fmt.Errorf("condition not met for %s", info.ObjectName())
	}
	if timeout < 0 {
		// we're out of time
		return info.Object, false, errWaitTimeoutWithName
	}

	fieldSelector := fields.OneTermEqualSelector("metadata.name", info.Name).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(context.TODO(), options)
		},
	}

	// this function is used to refresh the cache to prevent timeout waits on resources that have disappeared
	preconditionFunc := func(store cache.Store) (bool, error) {
		_, exists, err := store.Get(&metav1.ObjectMeta{Namespace: info.Namespace, Name: info.Name})
		if err != nil {
			return true, err
		}
		if !exists {
			// since we're looking for it to disappear we just return here if it no longer exists
			return true, nil
		}

		return false, nil
	}

	intrCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	intr := interrupt.New(nil, cancel)
	err := intr.Run(func() error {
		_, err := watchtools.UntilWithSync(intrCtx, lw, &unstructured.Unstructured{}, preconditionFunc, Wait{errOut: o.ErrOut}.IsDeleted)
		if errors.Is(err, context.DeadlineExceeded) {
			return errWaitTimeoutWithName
		}
		return err
	})
	if err != nil {
		if err == wait.ErrWaitTimeout {
			return gottenObj, false, errWaitTimeoutWithName
		}
		return gottenObj, false, err
	}

	return gottenObj, true, nil
}

// Wait has helper methods for handling watches, including error handling.
type Wait struct {
	errOut io.Writer
}

// IsDeleted returns true if the object is deleted. It prints any errors it encounters.
func (w Wait) IsDeleted(event watch.Event) (bool, error) {
	switch event.Type {
	case watch.Error:
		// keep waiting in the event we see an error - we expect the watch to be closed by
		// the server if the error is unrecoverable.
		err := apierrors.FromObject(event.Object)
		fmt.Fprintf(w.errOut, "error: An error occurred while waiting for the object to be deleted: %v", err)
		return false, nil
	case watch.Deleted:
		return true, nil
	default:
		return false, nil
	}
}

type isCondMetFunc func(event watch.Event) (bool, error)
type checkCondFunc func(obj *unstructured.Unstructured) (bool, error)

// getObjAndCheckCondition will make a List query to the API server to get the object and check if the condition is met using check function.
// If the condition is not met, it will make a Watch query to the server and pass in the condMet function
func getObjAndCheckCondition(ctx context.Context, info *resource.Info, o *WaitOptions, condMet isCondMetFunc, check checkCondFunc) (runtime.Object, bool, error) {
	if len(info.Name) == 0 {
		return info.Object, false, fmt.Errorf("resource name must be provided")
	}

	endTime := time.Now().Add(o.Timeout)
	timeout := time.Until(endTime)
	errWaitTimeoutWithName := extendErrWaitTimeout(wait.ErrWaitTimeout, info)
	if o.Timeout == 0 {
		// If timeout is zero we will fetch the object(s) once only and check
		gottenObj, initObjGetErr := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Get(context.Background(), info.Name, metav1.GetOptions{})
		if initObjGetErr != nil {
			return nil, false, initObjGetErr
		}
		if gottenObj == nil {
			return nil, false, fmt.Errorf("condition not met for %s", info.ObjectName())
		}
		conditionCheck, err := check(gottenObj)
		if err != nil {
			return gottenObj, false, err
		}
		if conditionCheck == false {
			return gottenObj, false, fmt.Errorf("condition not met for %s", info.ObjectName())
		}
		return gottenObj, true, nil
	}
	if timeout < 0 {
		// we're out of time
		return info.Object, false, errWaitTimeoutWithName
	}

	mapping := info.ResourceMapping() // used to pass back meaningful errors if object disappears
	fieldSelector := fields.OneTermEqualSelector("metadata.name", info.Name).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(context.TODO(), options)
		},
	}

	// this function is used to refresh the cache to prevent timeout waits on resources that have disappeared
	preconditionFunc := func(store cache.Store) (bool, error) {
		_, exists, err := store.Get(&metav1.ObjectMeta{Namespace: info.Namespace, Name: info.Name})
		if err != nil {
			return true, err
		}
		if !exists {
			return true, apierrors.NewNotFound(mapping.Resource.GroupResource(), info.Name)
		}

		return false, nil
	}

	intrCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var result runtime.Object
	intr := interrupt.New(nil, cancel)
	err := intr.Run(func() error {
		ev, err := watchtools.UntilWithSync(intrCtx, lw, &unstructured.Unstructured{}, preconditionFunc, watchtools.ConditionFunc(condMet))
		if ev != nil {
			result = ev.Object
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return errWaitTimeoutWithName
		}
		return err
	})
	if err != nil {
		if err == wait.ErrWaitTimeout {
			return result, false, errWaitTimeoutWithName
		}
		return result, false, err
	}

	return result, true, nil
}

// ConditionalWait hold information to check an API status condition
type ConditionalWait struct {
	conditionName   string
	conditionStatus string
	// errOut is written to if an error occurs
	errOut io.Writer
}

// IsConditionMet is a conditionfunc for waiting on an API condition to be met
func (w ConditionalWait) IsConditionMet(ctx context.Context, info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	return getObjAndCheckCondition(ctx, info, o, w.isConditionMet, w.checkCondition)
}

func (w ConditionalWait) checkCondition(obj *unstructured.Unstructured) (bool, error) {
	conditions, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if err != nil {
		return false, err
	}
	if !found {
		return false, nil
	}
	for _, conditionUncast := range conditions {
		condition := conditionUncast.(map[string]interface{})
		name, found, err := unstructured.NestedString(condition, "type")
		if !found || err != nil || !strings.EqualFold(name, w.conditionName) {
			continue
		}
		status, found, err := unstructured.NestedString(condition, "status")
		if !found || err != nil {
			continue
		}
		generation, found, _ := unstructured.NestedInt64(obj.Object, "metadata", "generation")
		if found {
			observedGeneration, found := getObservedGeneration(obj, condition)
			if found && observedGeneration < generation {
				return false, nil
			}
		}
		return strings.EqualFold(status, w.conditionStatus), nil
	}

	return false, nil
}

func (w ConditionalWait) isConditionMet(event watch.Event) (bool, error) {
	if event.Type == watch.Error {
		// keep waiting in the event we see an error - we expect the watch to be closed by
		// the server
		err := apierrors.FromObject(event.Object)
		fmt.Fprintf(w.errOut, "error: An error occurred while waiting for the condition to be satisfied: %v", err)
		return false, nil
	}
	if event.Type == watch.Deleted {
		// this will chain back out, result in another get and an return false back up the chain
		return false, nil
	}
	obj := event.Object.(*unstructured.Unstructured)
	return w.checkCondition(obj)
}

func extendErrWaitTimeout(err error, info *resource.Info) error {
	return fmt.Errorf("%s on %s/%s", err.Error(), info.Mapping.Resource.Resource, info.Name)
}

func getObservedGeneration(obj *unstructured.Unstructured, condition map[string]interface{}) (int64, bool) {
	conditionObservedGeneration, found, _ := unstructured.NestedInt64(condition, "observedGeneration")
	if found {
		return conditionObservedGeneration, true
	}
	statusObservedGeneration, found, _ := unstructured.NestedInt64(obj.Object, "status", "observedGeneration")
	return statusObservedGeneration, found
}

// JSONPathWait holds a JSONPath Parser which has the ability
// to check for the JSONPath condition and compare with the API server provided JSON output.
type JSONPathWait struct {
	jsonPathCondition string
	jsonPathParser    *jsonpath.JSONPath
	// errOut is written to if an error occurs
	errOut io.Writer
}

// IsJSONPathConditionMet fulfills the requirements of the interface ConditionFunc which provides condition check
func (j JSONPathWait) IsJSONPathConditionMet(ctx context.Context, info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	return getObjAndCheckCondition(ctx, info, o, j.isJSONPathConditionMet, j.checkCondition)
}

// isJSONPathConditionMet is a helper function of IsJSONPathConditionMet
// which check the watch event and check if a JSONPathWait condition is met
func (j JSONPathWait) isJSONPathConditionMet(event watch.Event) (bool, error) {
	if event.Type == watch.Error {
		// keep waiting in the event we see an error - we expect the watch to be closed by
		// the server
		err := apierrors.FromObject(event.Object)
		fmt.Fprintf(j.errOut, "error: An error occurred while waiting for the condition to be satisfied: %v", err)
		return false, nil
	}
	if event.Type == watch.Deleted {
		// this will chain back out, result in another get and an return false back up the chain
		return false, nil
	}
	// event runtime Object can be safely asserted to Unstructed
	// because we are working with dynamic client
	obj := event.Object.(*unstructured.Unstructured)
	return j.checkCondition(obj)
}

// checkCondition uses JSONPath parser to parse the JSON received from the API server
// and check if it matches the desired condition
func (j JSONPathWait) checkCondition(obj *unstructured.Unstructured) (bool, error) {
	queryObj := obj.UnstructuredContent()
	parseResults, err := j.jsonPathParser.FindResults(queryObj)
	if err != nil {
		return false, err
	}
	if len(parseResults) == 0 || len(parseResults[0]) == 0 {
		return false, nil
	}
	if err := verifyParsedJSONPath(parseResults); err != nil {
		return false, err
	}
	isConditionMet, err := compareResults(parseResults[0][0], j.jsonPathCondition)
	if err != nil {
		return false, err
	}
	return isConditionMet, nil
}

// verifyParsedJSONPath verifies the JSON received from the API server is valid.
// It will only accept a single JSON
func verifyParsedJSONPath(results [][]reflect.Value) error {
	if len(results) > 1 {
		return errors.New("given jsonpath expression matches more than one list")
	}
	if len(results[0]) > 1 {
		return errors.New("given jsonpath expression matches more than one value")
	}
	return nil
}

// compareResults will compare the reflect.Value from the result parsed by the
// JSONPath parser with the expected value given by the value
//
// Since this is coming from an unstructured this can only ever be a primitive,
// map[string]interface{}, or []interface{}.
// We do not support the last two and rely on fmt to handle conversion to string
// and compare the result with user input
func compareResults(r reflect.Value, expectedVal string) (bool, error) {
	switch r.Interface().(type) {
	case map[string]interface{}, []interface{}:
		return false, errors.New("jsonpath leads to a nested object or list which is not supported")
	}
	s := fmt.Sprintf("%v", r.Interface())
	return strings.TrimSpace(s) == strings.TrimSpace(expectedVal), nil
}
