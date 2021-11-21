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
	"time"

	"github.com/spf13/cobra"

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
	watchtools "k8s.io/client-go/tools/watch"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
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

		# The default value of status condition is true; you can set it to false
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
	cmd.Flags().StringVar(&flags.ForCondition, "for", flags.ForCondition, "The condition to wait on: [delete|condition=condition-name|jsonpath='{JSONPath expression}'=JSONPath Condition]. The default status value of condition-name is true, you can set false with condition=condition-name=false.")
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

	waiter, err := waiterFor(flags.ForCondition, flags.ErrOut)
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
		Printer:        printer,
		Waiter:         waiter,
		IOStreams:      flags.IOStreams,
	}

	return o, nil
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

	Printer printers.ResourcePrinter
	Waiter  Waiter
	genericclioptions.IOStreams
}

// RunWait runs the waiting logic
func (o *WaitOptions) RunWait() error {
	visitCount := 0

	err := o.ResourceFinder.Do().Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		visitCount++
		finalObject, success, err := o.Waiter.VisitResource(info, o)
		if success {
			o.Printer.PrintObj(finalObject, o.Out)
			return nil
		}
		if err == nil {
			return fmt.Errorf("%v unsatisified for unknown reason", finalObject)
		}
		return err
	})

	return o.Waiter.OnWaitLoopCompletion(visitCount, err)
}

type isCondMetFunc func(event watch.Event) (bool, error)
type checkCondFunc func(obj *unstructured.Unstructured) (bool, error)

// getObjAndCheckCondition will make a List query to the API server to get the object and check if the condition is met using check function.
// If the condition is not met, it will make a Watch query to the server and pass in the condMet function
func getObjAndCheckCondition(info *resource.Info, o *WaitOptions, condMet isCondMetFunc, check checkCondFunc) (runtime.Object, bool, error) {
	endTime := time.Now().Add(o.Timeout)
	for {
		if len(info.Name) == 0 {
			return info.Object, false, fmt.Errorf("resource name must be provided")
		}

		nameSelector := fields.OneTermEqualSelector("metadata.name", info.Name).String()

		var gottenObj *unstructured.Unstructured
		// List with a name field selector to get the current resourceVersion to watch from (not the object's resourceVersion)
		gottenObjList, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).List(context.TODO(), metav1.ListOptions{FieldSelector: nameSelector})

		resourceVersion := ""
		switch {
		case err != nil:
			return info.Object, false, err
		case len(gottenObjList.Items) != 1:
			resourceVersion = gottenObjList.GetResourceVersion()
		default:
			gottenObj = &gottenObjList.Items[0]
			conditionMet, err := check(gottenObj)
			if conditionMet {
				return gottenObj, true, nil
			}
			if err != nil {
				return gottenObj, false, err
			}
			resourceVersion = gottenObjList.GetResourceVersion()
		}

		watchOptions := metav1.ListOptions{}
		watchOptions.FieldSelector = nameSelector
		watchOptions.ResourceVersion = resourceVersion
		objWatch, err := o.DynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(context.TODO(), watchOptions)
		if err != nil {
			return gottenObj, false, err
		}

		timeout := endTime.Sub(time.Now())
		errWaitTimeoutWithName := extendErrWaitTimeout(wait.ErrWaitTimeout, info)
		if timeout < 0 {
			// we're out of time
			return gottenObj, false, errWaitTimeoutWithName
		}

		ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), o.Timeout)
		watchEvent, err := watchtools.UntilWithoutRetry(ctx, objWatch, watchtools.ConditionFunc(condMet))
		cancel()
		switch {
		case err == nil:
			return watchEvent.Object, true, nil
		case err == watchtools.ErrWatchClosed:
			continue
		case err == wait.ErrWaitTimeout:
			if watchEvent != nil {
				return watchEvent.Object, false, errWaitTimeoutWithName
			}
			return gottenObj, false, errWaitTimeoutWithName
		default:
			return gottenObj, false, err
		}
	}
}

func extendErrWaitTimeout(err error, info *resource.Info) error {
	return fmt.Errorf("%s on %s/%s", err.Error(), info.Mapping.Resource.Resource, info.Name)
}
