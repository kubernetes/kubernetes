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
	"strings"
	"time"

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/printers"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	"k8s.io/client-go/dynamic"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

var (
	wait_long = templates.LongDesc(`
		Experimental: Wait for a specific condition on one or many resources.

		The command takes multiple resources and waits until the specified condition
		is seen in the Status field of every given resource.

		Alternatively, the command can wait for the given set of resources to be deleted
		by providing the "delete" keyword as the value to the --for flag.

		A successful message will be printed to stdout indicating when the specified
        condition has been met. One can use -o option to change to output destination.`)

	wait_example = templates.Examples(`
		# Wait for the pod "busybox1" to contain the status condition of type "Ready".
		kubectl wait --for=condition=Ready pod/busybox1

		# Wait for the pod "busybox1" to be deleted, with a timeout of 60s, after having issued the "delete" command.
		kubectl delete pod/busybox1
		kubectl wait --for=delete pod/busybox1 --timeout=60s`)
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
			WithAllNamespaces(false).
			WithLatest(),

		Timeout: 30 * time.Second,

		IOStreams: streams,
	}
}

// NewCmdWait returns a cobra command for waiting
func NewCmdWait(restClientGetter genericclioptions.RESTClientGetter, streams genericclioptions.IOStreams) *cobra.Command {
	flags := NewWaitFlags(restClientGetter, streams)

	cmd := &cobra.Command{
		Use: "wait resource.group/name [--for=delete|--for condition=available]",
		DisableFlagsInUseLine: true,
		Short:   "Experimental: Wait for a specific condition on one or many resources.",
		Long:    wait_long,
		Example: wait_example,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(args)
			cmdutil.CheckErr(err)
			err = o.RunWait()
			cmdutil.CheckErr(err)
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
	cmd.Flags().StringVar(&flags.ForCondition, "for", flags.ForCondition, "The condition to wait on: [delete|condition=condition-name].")
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
	conditionFn, err := conditionFuncFor(flags.ForCondition)
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

		Printer:     printer,
		ConditionFn: conditionFn,
		IOStreams:   flags.IOStreams,
	}

	return o, nil
}

func conditionFuncFor(condition string) (ConditionFunc, error) {
	if strings.ToLower(condition) == "delete" {
		return IsDeleted, nil
	}
	if strings.HasPrefix(condition, "condition=") {
		conditionName := condition[len("condition="):]
		return ConditionalWait{
			conditionName: conditionName,
			// TODO allow specifying a false
			conditionStatus: "true",
		}.IsConditionMet, nil
	}

	return nil, fmt.Errorf("unrecognized condition: %q", condition)
}

type ResourceLocation struct {
	GroupResource schema.GroupResource
	Namespace     string
	Name          string
}

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

	Printer     printers.ResourcePrinter
	ConditionFn ConditionFunc
	genericclioptions.IOStreams
}

// ConditionFunc is the interface for providing condition checks
type ConditionFunc func(info *resource.Info, o *WaitOptions) (finalObject runtime.Object, done bool, err error)

// RunWait runs the waiting logic
func (o *WaitOptions) RunWait() error {
	visitCount := 0
	err := o.ResourceFinder.Do().Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		visitCount++
		finalObject, success, err := o.ConditionFn(info, o)
		if success {
			o.Printer.PrintObj(finalObject, o.Out)
			return nil
		}
		if err == nil {
			return fmt.Errorf("%v unsatisified for unknown reason", finalObject)
		}
		return err
	})
	if err != nil {
		return err
	}
	if visitCount == 0 {
		return errNoMatchingResources
	}
	return err
}

// IsDeleted is a condition func for waiting for something to be deleted
func IsDeleted(info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	endTime := time.Now().Add(o.Timeout)
	for {
		gottenObj, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Get(info.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return info.Object, true, nil
		}
		if err != nil {
			// TODO this could do something slightly fancier if we wish
			return info.Object, false, err
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

		watchOptions := metav1.ListOptions{}
		watchOptions.FieldSelector = "metadata.name=" + info.Name
		watchOptions.ResourceVersion = gottenObj.GetResourceVersion()
		objWatch, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(watchOptions)
		if err != nil {
			return gottenObj, false, err
		}

		timeout := endTime.Sub(time.Now())
		if timeout < 0 {
			// we're out of time
			return gottenObj, false, wait.ErrWaitTimeout
		}

		ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), o.Timeout)
		watchEvent, err := watchtools.UntilWithoutRetry(ctx, objWatch, isDeleted)
		cancel()
		switch {
		case err == nil:
			return watchEvent.Object, true, nil
		case err == watchtools.ErrWatchClosed:
			continue
		case err == wait.ErrWaitTimeout:
			if watchEvent != nil {
				return watchEvent.Object, false, wait.ErrWaitTimeout
			}
			return gottenObj, false, wait.ErrWaitTimeout
		default:
			return gottenObj, false, err
		}
	}
}

func isDeleted(event watch.Event) (bool, error) {
	return event.Type == watch.Deleted, nil
}

// ConditionalWait hold information to check an API status condition
type ConditionalWait struct {
	conditionName   string
	conditionStatus string
}

// IsConditionMet is a conditionfunc for waiting on an API condition to be met
func (w ConditionalWait) IsConditionMet(info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	endTime := time.Now().Add(o.Timeout)
	for {
		resourceVersion := ""
		gottenObj, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Get(info.Name, metav1.GetOptions{})
		switch {
		case apierrors.IsNotFound(err):
			resourceVersion = "0"
		case err != nil:
			return info.Object, false, err
		default:
			conditionMet, err := w.checkCondition(gottenObj)
			if conditionMet {
				return gottenObj, true, nil
			}
			if err != nil {
				return gottenObj, false, err
			}
			resourceVersion = gottenObj.GetResourceVersion()
		}

		watchOptions := metav1.ListOptions{}
		watchOptions.FieldSelector = "metadata.name=" + info.Name
		watchOptions.ResourceVersion = resourceVersion
		objWatch, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(watchOptions)
		if err != nil {
			return gottenObj, false, err
		}

		timeout := endTime.Sub(time.Now())
		if timeout < 0 {
			// we're out of time
			return gottenObj, false, wait.ErrWaitTimeout
		}

		ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), o.Timeout)
		watchEvent, err := watchtools.UntilWithoutRetry(ctx, objWatch, w.isConditionMet)
		cancel()
		switch {
		case err == nil:
			return watchEvent.Object, true, nil
		case err == watchtools.ErrWatchClosed:
			continue
		case err == wait.ErrWaitTimeout:
			if watchEvent != nil {
				return watchEvent.Object, false, wait.ErrWaitTimeout
			}
			return gottenObj, false, wait.ErrWaitTimeout
		default:
			return gottenObj, false, err
		}
	}
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
		if !found || err != nil || strings.ToLower(name) != strings.ToLower(w.conditionName) {
			continue
		}
		status, found, err := unstructured.NestedString(condition, "status")
		if !found || err != nil {
			continue
		}
		return strings.ToLower(status) == strings.ToLower(w.conditionStatus), nil
	}

	return false, nil
}

func (w ConditionalWait) isConditionMet(event watch.Event) (bool, error) {
	if event.Type == watch.Deleted {
		// this will chain back out, result in another get and an return false back up the chain
		return false, nil
	}
	obj := event.Object.(*unstructured.Unstructured)
	return w.checkCondition(obj)
}
