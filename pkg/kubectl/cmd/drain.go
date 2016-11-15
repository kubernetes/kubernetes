/*
Copyright 2015 The Kubernetes Authors.

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

package cmd

import (
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"time"

	"github.com/jonboulle/clockwork"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
)

type DrainOptions struct {
	client             *internalclientset.Clientset
	restClient         *restclient.RESTClient
	factory            cmdutil.Factory
	Force              bool
	GracePeriodSeconds int
	IgnoreDaemonsets   bool
	Timeout            time.Duration
	backOff            clockwork.Clock
	DeleteLocalData    bool
	mapper             meta.RESTMapper
	nodeInfo           *resource.Info
	out                io.Writer
	errOut             io.Writer
	typer              runtime.ObjectTyper
}

// Takes a pod and returns a bool indicating whether or not to operate on the
// pod, an optional warning message, and an optional fatal error.
type podFilter func(api.Pod) (include bool, w *warning, f *fatal)
type warning struct {
	string
}
type fatal struct {
	string
}

const (
	EvictionKind        = "Eviction"
	EvictionSubresource = "pods/eviction"

	kDaemonsetFatal      = "DaemonSet-managed pods (use --ignore-daemonsets to ignore)"
	kDaemonsetWarning    = "Ignoring DaemonSet-managed pods"
	kLocalStorageFatal   = "pods with local storage (use --delete-local-data to override)"
	kLocalStorageWarning = "Deleting pods with local storage"
	kUnmanagedFatal      = "pods not managed by ReplicationController, ReplicaSet, Job, or DaemonSet (use --force to override)"
	kUnmanagedWarning    = "Deleting pods not managed by ReplicationController, ReplicaSet, Job, or DaemonSet"
	kMaxNodeUpdateRetry  = 10
)

var (
	cordon_long = templates.LongDesc(`
		Mark node as unschedulable.`)

	cordon_example = templates.Examples(`
		# Mark node "foo" as unschedulable.
		kubectl cordon foo`)
)

func NewCmdCordon(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &DrainOptions{factory: f, out: out}

	cmd := &cobra.Command{
		Use:     "cordon NODE",
		Short:   "Mark node as unschedulable",
		Long:    cordon_long,
		Example: cordon_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.SetupDrain(cmd, args))
			cmdutil.CheckErr(options.RunCordonOrUncordon(true))
		},
	}
	return cmd
}

var (
	uncordon_long = templates.LongDesc(`
		Mark node as schedulable.`)

	uncordon_example = templates.Examples(`
		# Mark node "foo" as schedulable.
		$ kubectl uncordon foo`)
)

func NewCmdUncordon(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &DrainOptions{factory: f, out: out}

	cmd := &cobra.Command{
		Use:     "uncordon NODE",
		Short:   "Mark node as schedulable",
		Long:    uncordon_long,
		Example: uncordon_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.SetupDrain(cmd, args))
			cmdutil.CheckErr(options.RunCordonOrUncordon(false))
		},
	}
	return cmd
}

var (
	drain_long = templates.LongDesc(`
		Drain node in preparation for maintenance.

		The given node will be marked unschedulable to prevent new pods from arriving.
		The 'drain' deletes all pods except mirror pods (which cannot be deleted through
		the API server).  If there are DaemonSet-managed pods, drain will not proceed
		without --ignore-daemonsets, and regardless it will not delete any
		DaemonSet-managed pods, because those pods would be immediately replaced by the
		DaemonSet controller, which ignores unschedulable markings.  If there are any
		pods that are neither mirror pods nor managed by ReplicationController,
		ReplicaSet, DaemonSet or Job, then drain will not delete any pods unless you
		use --force.

		When you are ready to put the node back into service, use kubectl uncordon, which
		will make the node schedulable again.

		![Workflow](http://kubernetes.io/images/docs/kubectl_drain.svg)`)

	drain_example = templates.Examples(`
		# Drain node "foo", even if there are pods not managed by a ReplicationController, ReplicaSet, Job, or DaemonSet on it.
		$ kubectl drain foo --force

		# As above, but abort if there are pods not managed by a ReplicationController, ReplicaSet, Job, or DaemonSet, and use a grace period of 15 minutes.
		$ kubectl drain foo --grace-period=900`)
)

func NewCmdDrain(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	options := &DrainOptions{factory: f, out: out, errOut: errOut, backOff: clockwork.NewRealClock()}

	cmd := &cobra.Command{
		Use:     "drain NODE",
		Short:   "Drain node in preparation for maintenance",
		Long:    drain_long,
		Example: drain_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.SetupDrain(cmd, args))
			cmdutil.CheckErr(options.RunDrain())
		},
	}
	cmd.Flags().BoolVar(&options.Force, "force", false, "Continue even if there are pods not managed by a ReplicationController, ReplicaSet, Job, or DaemonSet.")
	cmd.Flags().BoolVar(&options.IgnoreDaemonsets, "ignore-daemonsets", false, "Ignore DaemonSet-managed pods.")
	cmd.Flags().BoolVar(&options.DeleteLocalData, "delete-local-data", false, "Continue even if there are pods using emptyDir (local data that will be deleted when the node is drained).")
	cmd.Flags().IntVar(&options.GracePeriodSeconds, "grace-period", -1, "Period of time in seconds given to each pod to terminate gracefully. If negative, the default value specified in the pod will be used.")
	cmd.Flags().DurationVar(&options.Timeout, "timeout", 0, "The length of time to wait before giving up on a delete, zero means determine a timeout from the size of the object")
	return cmd
}

// SetupDrain populates some fields from the factory, grabs command line
// arguments and looks up the node using Builder
func (o *DrainOptions) SetupDrain(cmd *cobra.Command, args []string) error {
	var err error
	if len(args) != 1 {
		return cmdutil.UsageError(cmd, fmt.Sprintf("USAGE: %s [flags]", cmd.Use))
	}

	if o.client, err = o.factory.ClientSet(); err != nil {
		return err
	}

	o.restClient, err = o.factory.RESTClient()
	if err != nil {
		return err
	}

	o.mapper, o.typer = o.factory.Object()

	cmdNamespace, _, err := o.factory.DefaultNamespace()
	if err != nil {
		return err
	}

	r := o.factory.NewBuilder().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		ResourceNames("node", args[0]).
		Do()

	if err = r.Err(); err != nil {
		return err
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		o.nodeInfo = info
		return nil
	})
}

// RunDrain runs the 'drain' command
func (o *DrainOptions) RunDrain() error {
	if err := o.RunCordonOrUncordon(true); err != nil {
		return err
	}

	err := o.deleteOrEvictPodsSimple()
	// TODO: update IsTooManyRequests() when the TooManyRequests(429) error returned from the API server has a non-empty Reason field
	for i := 1; i <= maxPatchRetry && apierrors.IsTooManyRequests(err); i++ {
		if i > triesBeforeBackOff {
			currBackOffPeriod := time.Duration(math.Exp2(float64(i-triesBeforeBackOff))) * backOffPeriod
			fmt.Fprintf(o.errOut, "Retry in %v\n", currBackOffPeriod)
			o.backOff.Sleep(currBackOffPeriod)
		}
		fmt.Fprintf(o.errOut, "Retrying\n")
		err = o.deleteOrEvictPodsSimple()
	}
	if err == nil {
		cmdutil.PrintSuccess(o.mapper, false, o.out, "node", o.nodeInfo.Name, false, "drained")
	}
	return err
}

func (o *DrainOptions) deleteOrEvictPodsSimple() error {
	pods, err := o.getPodsForDeletion()
	if err != nil {
		return err
	}
	if o.Timeout == 0 {
		o.Timeout = kubectl.Timeout + time.Duration(10*len(pods))*time.Second
	}
	err = o.deleteOrEvictPods(pods)
	if err != nil {
		pendingPods, newErr := o.getPodsForDeletion()
		if newErr != nil {
			return newErr
		}
		fmt.Fprintf(o.errOut, "There are pending pods when an error occurred: %v\n", err)
		for _, pendingPod := range pendingPods {
			fmt.Fprintf(o.errOut, "%s/%s\n", "pod", pendingPod.Name)
		}
	}
	return err
}

func (o *DrainOptions) getController(sr *api.SerializedReference) (interface{}, error) {
	switch sr.Reference.Kind {
	case "ReplicationController":
		return o.client.Core().ReplicationControllers(sr.Reference.Namespace).Get(sr.Reference.Name)
	case "DaemonSet":
		return o.client.Extensions().DaemonSets(sr.Reference.Namespace).Get(sr.Reference.Name)
	case "Job":
		return o.client.Batch().Jobs(sr.Reference.Namespace).Get(sr.Reference.Name)
	case "ReplicaSet":
		return o.client.Extensions().ReplicaSets(sr.Reference.Namespace).Get(sr.Reference.Name)
	case "StatefulSet":
		return o.client.Apps().StatefulSets(sr.Reference.Namespace).Get(sr.Reference.Name)
	}
	return nil, fmt.Errorf("Unknown controller kind %q", sr.Reference.Kind)
}

func (o *DrainOptions) getPodCreator(pod api.Pod) (*api.SerializedReference, error) {
	creatorRef, found := pod.ObjectMeta.Annotations[api.CreatedByAnnotation]
	if !found {
		return nil, nil
	}
	// Now verify that the specified creator actually exists.
	sr := &api.SerializedReference{}
	if err := runtime.DecodeInto(o.factory.Decoder(true), []byte(creatorRef), sr); err != nil {
		return nil, err
	}
	// We assume the only reason for an error is because the controller is
	// gone/missing, not for any other cause.  TODO(mml): something more
	// sophisticated than this
	_, err := o.getController(sr)
	if err != nil {
		return nil, err
	}
	return sr, nil
}

func (o *DrainOptions) unreplicatedFilter(pod api.Pod) (bool, *warning, *fatal) {
	// any finished pod can be removed
	if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
		return true, nil, nil
	}

	sr, err := o.getPodCreator(pod)
	if err != nil {
		return false, nil, &fatal{err.Error()}
	}
	if sr != nil {
		return true, nil, nil
	}
	if !o.Force {
		return false, nil, &fatal{kUnmanagedFatal}
	}
	return true, &warning{kUnmanagedWarning}, nil
}

func (o *DrainOptions) daemonsetFilter(pod api.Pod) (bool, *warning, *fatal) {
	// Note that we return false in all cases where the pod is DaemonSet managed,
	// regardless of flags.  We never delete them, the only question is whether
	// their presence constitutes an error.
	sr, err := o.getPodCreator(pod)
	if err != nil {
		return false, nil, &fatal{err.Error()}
	}
	if sr == nil || sr.Reference.Kind != "DaemonSet" {
		return true, nil, nil
	}
	if _, err := o.client.Extensions().DaemonSets(sr.Reference.Namespace).Get(sr.Reference.Name); err != nil {
		return false, nil, &fatal{err.Error()}
	}
	if !o.IgnoreDaemonsets {
		return false, nil, &fatal{kDaemonsetFatal}
	}
	return false, &warning{kDaemonsetWarning}, nil
}

func mirrorPodFilter(pod api.Pod) (bool, *warning, *fatal) {
	if _, found := pod.ObjectMeta.Annotations[types.ConfigMirrorAnnotationKey]; found {
		return false, nil, nil
	}
	return true, nil, nil
}

func hasLocalStorage(pod api.Pod) bool {
	for _, volume := range pod.Spec.Volumes {
		if volume.EmptyDir != nil {
			return true
		}
	}

	return false
}

func (o *DrainOptions) localStorageFilter(pod api.Pod) (bool, *warning, *fatal) {
	if !hasLocalStorage(pod) {
		return true, nil, nil
	}
	if !o.DeleteLocalData {
		return false, nil, &fatal{kLocalStorageFatal}
	}
	return true, &warning{kLocalStorageWarning}, nil
}

// Map of status message to a list of pod names having that status.
type podStatuses map[string][]string

func (ps podStatuses) Message() string {
	msgs := []string{}

	for key, pods := range ps {
		msgs = append(msgs, fmt.Sprintf("%s: %s", key, strings.Join(pods, ", ")))
	}
	return strings.Join(msgs, "; ")
}

// getPodsForDeletion returns all the pods we're going to delete.  If there are
// any pods preventing us from deleting, we return that list in an error.
func (o *DrainOptions) getPodsForDeletion() (pods []api.Pod, err error) {
	podList, err := o.client.Core().Pods(api.NamespaceAll).List(api.ListOptions{
		FieldSelector: fields.SelectorFromSet(fields.Set{"spec.nodeName": o.nodeInfo.Name})})
	if err != nil {
		return pods, err
	}

	ws := podStatuses{}
	fs := podStatuses{}

	for _, pod := range podList.Items {
		podOk := true
		for _, filt := range []podFilter{mirrorPodFilter, o.localStorageFilter, o.unreplicatedFilter, o.daemonsetFilter} {
			filterOk, w, f := filt(pod)

			podOk = podOk && filterOk
			if w != nil {
				ws[w.string] = append(ws[w.string], pod.Name)
			}
			if f != nil {
				fs[f.string] = append(fs[f.string], pod.Name)
			}
		}
		if podOk {
			pods = append(pods, pod)
		}
	}

	if len(fs) > 0 {
		return []api.Pod{}, errors.New(fs.Message())
	}
	if len(ws) > 0 {
		fmt.Fprintf(o.errOut, "WARNING: %s\n", ws.Message())
	}
	return pods, nil
}

func (o *DrainOptions) deletePod(pod api.Pod) error {
	deleteOptions := &api.DeleteOptions{}
	if o.GracePeriodSeconds >= 0 {
		gracePeriodSeconds := int64(o.GracePeriodSeconds)
		deleteOptions.GracePeriodSeconds = &gracePeriodSeconds
	}
	return o.client.Core().Pods(pod.Namespace).Delete(pod.Name, deleteOptions)
}

func (o *DrainOptions) evictPod(pod api.Pod, policyGroupVersion string) error {
	deleteOptions := &api.DeleteOptions{}
	if o.GracePeriodSeconds >= 0 {
		gracePeriodSeconds := int64(o.GracePeriodSeconds)
		deleteOptions.GracePeriodSeconds = &gracePeriodSeconds
	}
	eviction := &policy.Eviction{
		TypeMeta: unversioned.TypeMeta{
			APIVersion: policyGroupVersion,
			Kind:       EvictionKind,
		},
		ObjectMeta: api.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		DeleteOptions: deleteOptions,
	}
	// Remember to change change the URL manipulation func when Evction's version change
	return o.client.Policy().Evictions(eviction.Namespace).Evict(eviction)
}

// deleteOrEvictPods deletes or evicts the pods on the api server
func (o *DrainOptions) deleteOrEvictPods(pods []api.Pod) error {
	if len(pods) == 0 {
		return nil
	}

	policyGroupVersion, err := SupportEviction(o.client)
	if err != nil {
		return err
	}

	for _, pod := range pods {
		if len(policyGroupVersion) > 0 {
			err = o.evictPod(pod, policyGroupVersion)
		} else {
			err = o.deletePod(pod)
		}
		if err != nil {
			return err
		}
	}

	getPodFn := func(namespace, name string) (*api.Pod, error) {
		return o.client.Core().Pods(namespace).Get(name)
	}
	_, err = o.waitForDelete(pods, kubectl.Interval, o.Timeout, getPodFn)
	return err
}

func (o *DrainOptions) waitForDelete(pods []api.Pod, interval, timeout time.Duration, getPodFn func(string, string) (*api.Pod, error)) ([]api.Pod, error) {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		pendingPods := []api.Pod{}
		for i, pod := range pods {
			p, err := getPodFn(pod.Namespace, pod.Name)
			if apierrors.IsNotFound(err) || (p != nil && p.ObjectMeta.UID != pod.ObjectMeta.UID) {
				cmdutil.PrintSuccess(o.mapper, false, o.out, "pod", pod.Name, false, "deleted")
				continue
			} else if err != nil {
				return false, err
			} else {
				pendingPods = append(pendingPods, pods[i])
			}
		}
		pods = pendingPods
		if len(pendingPods) > 0 {
			return false, nil
		}
		return true, nil
	})
	return pods, err
}

// SupportEviction uses Discovery API to find out if the server support eviction subresource
// If support, it will return its groupVersion; Otherwise, it will return ""
func SupportEviction(clientset *internalclientset.Clientset) (string, error) {
	discoveryClient := clientset.Discovery()
	groupList, err := discoveryClient.ServerGroups()
	if err != nil {
		return "", err
	}
	foundPolicyGroup := false
	var policyGroupVersion string
	for _, group := range groupList.Groups {
		if group.Name == "policy" {
			foundPolicyGroup = true
			policyGroupVersion = group.PreferredVersion.GroupVersion
			break
		}
	}
	if !foundPolicyGroup {
		return "", nil
	}
	resourceList, err := discoveryClient.ServerResourcesForGroupVersion("v1")
	if err != nil {
		return "", err
	}
	for _, resource := range resourceList.APIResources {
		if resource.Name == EvictionSubresource && resource.Kind == EvictionKind {
			return policyGroupVersion, nil
		}
	}
	return "", nil
}

// RunCordonOrUncordon runs either Cordon or Uncordon.  The desired value for
// "Unschedulable" is passed as the first arg.
func (o *DrainOptions) RunCordonOrUncordon(desired bool) error {
	cmdNamespace, _, err := o.factory.DefaultNamespace()
	if err != nil {
		return err
	}

	if o.nodeInfo.Mapping.GroupVersionKind.Kind == "Node" {
		unsched := reflect.ValueOf(o.nodeInfo.Object).Elem().FieldByName("Spec").FieldByName("Unschedulable")
		if unsched.Bool() == desired {
			cmdutil.PrintSuccess(o.mapper, false, o.out, o.nodeInfo.Mapping.Resource, o.nodeInfo.Name, false, already(desired))
		} else {
			helper := resource.NewHelper(o.restClient, o.nodeInfo.Mapping)
			unsched.SetBool(desired)
			var err error
			for i := 0; i < kMaxNodeUpdateRetry; i++ {
				// We don't care about what previous versions may exist, we always want
				// to overwrite, and Replace always sets current ResourceVersion if version is "".
				helper.Versioner.SetResourceVersion(o.nodeInfo.Object, "")
				_, err = helper.Replace(cmdNamespace, o.nodeInfo.Name, true, o.nodeInfo.Object)
				if err != nil {
					if !apierrors.IsConflict(err) {
						return err
					}
				} else {
					break
				}
				// It's a race, no need to sleep
			}
			if err != nil {
				return err
			}
			cmdutil.PrintSuccess(o.mapper, false, o.out, o.nodeInfo.Mapping.Resource, o.nodeInfo.Name, false, changed(desired))
		}
	} else {
		cmdutil.PrintSuccess(o.mapper, false, o.out, o.nodeInfo.Mapping.Resource, o.nodeInfo.Name, false, "skipped")
	}

	return nil
}

// already() and changed() return suitable strings for {un,}cordoning

func already(desired bool) string {
	if desired {
		return "already cordoned"
	}
	return "already uncordoned"
}

func changed(desired bool) string {
	if desired {
		return "cordoned"
	}
	return "uncordoned"
}
