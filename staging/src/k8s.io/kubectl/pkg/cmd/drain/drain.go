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

package drain

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/drain"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type DrainCmdOptions struct {
	PrintFlags *genericclioptions.PrintFlags
	ToPrinter  func(string) (printers.ResourcePrinterFunc, error)

	Namespace string

	drainer   *drain.Helper
	nodeInfos []*resource.Info

	genericclioptions.IOStreams
}

var (
	cordonLong = templates.LongDesc(i18n.T(`
		Mark node as unschedulable.`))

	cordonExample = templates.Examples(i18n.T(`
		# Mark node "foo" as unschedulable.
		kubectl cordon foo`))
)

func NewCmdCordon(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewDrainCmdOptions(f, ioStreams)

	cmd := &cobra.Command{
		Use:                   "cordon NODE",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Mark node as unschedulable"),
		Long:                  cordonLong,
		Example:               cordonExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.RunCordonOrUncordon(true))
		},
	}
	cmd.Flags().StringVarP(&o.drainer.Selector, "selector", "l", o.drainer.Selector, "Selector (label query) to filter on")
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

var (
	uncordonLong = templates.LongDesc(i18n.T(`
		Mark node as schedulable.`))

	uncordonExample = templates.Examples(i18n.T(`
		# Mark node "foo" as schedulable.
		$ kubectl uncordon foo`))
)

func NewCmdUncordon(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewDrainCmdOptions(f, ioStreams)

	cmd := &cobra.Command{
		Use:                   "uncordon NODE",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Mark node as schedulable"),
		Long:                  uncordonLong,
		Example:               uncordonExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.RunCordonOrUncordon(false))
		},
	}
	cmd.Flags().StringVarP(&o.drainer.Selector, "selector", "l", o.drainer.Selector, "Selector (label query) to filter on")
	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

var (
	drainLong = templates.LongDesc(i18n.T(`
		Drain node in preparation for maintenance.

		The given node will be marked unschedulable to prevent new pods from arriving.
		'drain' evicts the pods if the APIServer supports
		[eviction](http://kubernetes.io/docs/admin/disruptions/). Otherwise, it will use normal
		DELETE to delete the pods.
		The 'drain' evicts or deletes all pods except mirror pods (which cannot be deleted through
		the API server).  If there are DaemonSet-managed pods, drain will not proceed
		without --ignore-daemonsets, and regardless it will not delete any
		DaemonSet-managed pods, because those pods would be immediately replaced by the
		DaemonSet controller, which ignores unschedulable markings.  If there are any
		pods that are neither mirror pods nor managed by ReplicationController,
		ReplicaSet, DaemonSet, StatefulSet or Job, then drain will not delete any pods unless you
		use --force.  --force will also allow deletion to proceed if the managing resource of one
		or more pods is missing.

		'drain' waits for graceful termination. You should not operate on the machine until
		the command completes.

		When you are ready to put the node back into service, use kubectl uncordon, which
		will make the node schedulable again.

		![Workflow](http://kubernetes.io/images/docs/kubectl_drain.svg)`))

	drainExample = templates.Examples(i18n.T(`
		# Drain node "foo", even if there are pods not managed by a ReplicationController, ReplicaSet, Job, DaemonSet or StatefulSet on it.
		$ kubectl drain foo --force

		# As above, but abort if there are pods not managed by a ReplicationController, ReplicaSet, Job, DaemonSet or StatefulSet, and use a grace period of 15 minutes.
		$ kubectl drain foo --grace-period=900`))
)

func NewDrainCmdOptions(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *DrainCmdOptions {
	return &DrainCmdOptions{
		PrintFlags: genericclioptions.NewPrintFlags("drained").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
		drainer: &drain.Helper{
			GracePeriodSeconds: -1,
			ErrOut:             ioStreams.ErrOut,
		},
	}
}

func NewCmdDrain(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewDrainCmdOptions(f, ioStreams)

	cmd := &cobra.Command{
		Use:                   "drain NODE",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Drain node in preparation for maintenance"),
		Long:                  drainLong,
		Example:               drainExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.RunDrain())
		},
	}
	cmd.Flags().BoolVar(&o.drainer.Force, "force", o.drainer.Force, "Continue even if there are pods not managed by a ReplicationController, ReplicaSet, Job, DaemonSet or StatefulSet.")
	cmd.Flags().BoolVar(&o.drainer.IgnoreAllDaemonSets, "ignore-daemonsets", o.drainer.IgnoreAllDaemonSets, "Ignore DaemonSet-managed pods.")
	cmd.Flags().BoolVar(&o.drainer.DeleteLocalData, "delete-local-data", o.drainer.DeleteLocalData, "Continue even if there are pods using emptyDir (local data that will be deleted when the node is drained).")
	cmd.Flags().IntVar(&o.drainer.GracePeriodSeconds, "grace-period", o.drainer.GracePeriodSeconds, "Period of time in seconds given to each pod to terminate gracefully. If negative, the default value specified in the pod will be used.")
	cmd.Flags().DurationVar(&o.drainer.Timeout, "timeout", o.drainer.Timeout, "The length of time to wait before giving up, zero means infinite")
	cmd.Flags().StringVarP(&o.drainer.Selector, "selector", "l", o.drainer.Selector, "Selector (label query) to filter on")
	cmd.Flags().StringVarP(&o.drainer.PodSelector, "pod-selector", "", o.drainer.PodSelector, "Label selector to filter pods on the node")

	cmdutil.AddDryRunFlag(cmd)
	return cmd
}

// Complete populates some fields from the factory, grabs command line
// arguments and looks up the node using Builder
func (o *DrainCmdOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	if len(args) == 0 && !cmd.Flags().Changed("selector") {
		return cmdutil.UsageErrorf(cmd, fmt.Sprintf("USAGE: %s [flags]", cmd.Use))
	}
	if len(args) > 0 && len(o.drainer.Selector) > 0 {
		return cmdutil.UsageErrorf(cmd, "error: cannot specify both a node name and a --selector option")
	}

	o.drainer.DryRun = cmdutil.GetDryRunFlag(cmd)

	if o.drainer.Client, err = f.KubernetesClientSet(); err != nil {
		return err
	}

	if len(o.drainer.PodSelector) > 0 {
		if _, err := labels.Parse(o.drainer.PodSelector); err != nil {
			return errors.New("--pod-selector=<pod_selector> must be a valid label selector")
		}
	}

	o.nodeInfos = []*resource.Info{}

	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.ToPrinter = func(operation string) (printers.ResourcePrinterFunc, error) {
		o.PrintFlags.NamePrintFlags.Operation = operation
		if o.drainer.DryRun {
			o.PrintFlags.Complete("%s (dry run)")
		}

		printer, err := o.PrintFlags.ToPrinter()
		if err != nil {
			return nil, err
		}

		return printer.PrintObj, nil
	}

	builder := f.NewBuilder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		NamespaceParam(o.Namespace).DefaultNamespace().
		ResourceNames("nodes", args...).
		SingleResourceType().
		Flatten()

	if len(o.drainer.Selector) > 0 {
		builder = builder.LabelSelectorParam(o.drainer.Selector).
			ResourceTypes("nodes")
	}

	r := builder.Do()

	if err = r.Err(); err != nil {
		return err
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		if info.Mapping.Resource.GroupResource() != (schema.GroupResource{Group: "", Resource: "nodes"}) {
			return fmt.Errorf("error: expected resource of type node, got %q", info.Mapping.Resource)
		}

		o.nodeInfos = append(o.nodeInfos, info)
		return nil
	})
}

// RunDrain runs the 'drain' command
func (o *DrainCmdOptions) RunDrain() error {
	if err := o.RunCordonOrUncordon(true); err != nil {
		return err
	}

	printObj, err := o.ToPrinter("drained")
	if err != nil {
		return err
	}

	drainedNodes := sets.NewString()
	var fatal error

	for _, info := range o.nodeInfos {
		var err error
		if !o.drainer.DryRun {
			err = o.deleteOrEvictPodsSimple(info)
		}
		if err == nil || o.drainer.DryRun {
			drainedNodes.Insert(info.Name)
			printObj(info.Object, o.Out)
		} else {
			fmt.Fprintf(o.ErrOut, "error: unable to drain node %q, aborting command...\n\n", info.Name)
			remainingNodes := []string{}
			fatal = err
			for _, remainingInfo := range o.nodeInfos {
				if drainedNodes.Has(remainingInfo.Name) {
					continue
				}
				remainingNodes = append(remainingNodes, remainingInfo.Name)
			}

			if len(remainingNodes) > 0 {
				fmt.Fprintf(o.ErrOut, "There are pending nodes to be drained:\n")
				for _, nodeName := range remainingNodes {
					fmt.Fprintf(o.ErrOut, " %s\n", nodeName)
				}
			}
			break
		}
	}

	return fatal
}

func (o *DrainCmdOptions) deleteOrEvictPodsSimple(nodeInfo *resource.Info) error {
	list, errs := o.drainer.GetPodsForDeletion(nodeInfo.Name)
	if errs != nil {
		return utilerrors.NewAggregate(errs)
	}
	if warnings := list.Warnings(); warnings != "" {
		fmt.Fprintf(o.ErrOut, "WARNING: %s\n", warnings)
	}

	if err := o.deleteOrEvictPods(list.Pods()); err != nil {
		pendingList, newErrs := o.drainer.GetPodsForDeletion(nodeInfo.Name)

		fmt.Fprintf(o.ErrOut, "There are pending pods in node %q when an error occurred: %v\n", nodeInfo.Name, err)
		for _, pendingPod := range pendingList.Pods() {
			fmt.Fprintf(o.ErrOut, "%s/%s\n", "pod", pendingPod.Name)
		}
		if newErrs != nil {
			fmt.Fprintf(o.ErrOut, "following errors also occurred:\n%s", utilerrors.NewAggregate(newErrs))
		}
		return err
	}
	return nil
}

// deleteOrEvictPods deletes or evicts the pods on the api server
func (o *DrainCmdOptions) deleteOrEvictPods(pods []corev1.Pod) error {
	if len(pods) == 0 {
		return nil
	}

	policyGroupVersion, err := drain.CheckEvictionSupport(o.drainer.Client)
	if err != nil {
		return err
	}

	getPodFn := func(namespace, name string) (*corev1.Pod, error) {
		return o.drainer.Client.CoreV1().Pods(namespace).Get(name, metav1.GetOptions{})
	}

	if len(policyGroupVersion) > 0 {
		return o.evictPods(pods, policyGroupVersion, getPodFn)
	} else {
		return o.deletePods(pods, getPodFn)
	}
}

func (o *DrainCmdOptions) evictPods(pods []corev1.Pod, policyGroupVersion string, getPodFn func(namespace, name string) (*corev1.Pod, error)) error {
	returnCh := make(chan error, 1)

	for _, pod := range pods {
		go func(pod corev1.Pod, returnCh chan error) {
			for {
				fmt.Fprintf(o.Out, "evicting pod %q\n", pod.Name)
				err := o.drainer.EvictPod(pod, policyGroupVersion)
				if err == nil {
					break
				} else if apierrors.IsNotFound(err) {
					returnCh <- nil
					return
				} else if apierrors.IsTooManyRequests(err) {
					fmt.Fprintf(o.ErrOut, "error when evicting pod %q (will retry after 5s): %v\n", pod.Name, err)
					time.Sleep(5 * time.Second)
				} else {
					returnCh <- fmt.Errorf("error when evicting pod %q: %v", pod.Name, err)
					return
				}
			}
			_, err := o.waitForDelete([]corev1.Pod{pod}, 1*time.Second, time.Duration(math.MaxInt64), true, getPodFn)
			if err == nil {
				returnCh <- nil
			} else {
				returnCh <- fmt.Errorf("error when waiting for pod %q terminating: %v", pod.Name, err)
			}
		}(pod, returnCh)
	}

	doneCount := 0
	var errors []error

	// 0 timeout means infinite, we use MaxInt64 to represent it.
	var globalTimeout time.Duration
	if o.drainer.Timeout == 0 {
		globalTimeout = time.Duration(math.MaxInt64)
	} else {
		globalTimeout = o.drainer.Timeout
	}
	globalTimeoutCh := time.After(globalTimeout)
	numPods := len(pods)
	for doneCount < numPods {
		select {
		case err := <-returnCh:
			doneCount++
			if err != nil {
				errors = append(errors, err)
			}
		case <-globalTimeoutCh:
			return fmt.Errorf("drain did not complete within %v", globalTimeout)
		}
	}
	return utilerrors.NewAggregate(errors)
}

func (o *DrainCmdOptions) deletePods(pods []corev1.Pod, getPodFn func(namespace, name string) (*corev1.Pod, error)) error {
	// 0 timeout means infinite, we use MaxInt64 to represent it.
	var globalTimeout time.Duration
	if o.drainer.Timeout == 0 {
		globalTimeout = time.Duration(math.MaxInt64)
	} else {
		globalTimeout = o.drainer.Timeout
	}
	for _, pod := range pods {
		err := o.drainer.DeletePod(pod)
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	_, err := o.waitForDelete(pods, 1*time.Second, globalTimeout, false, getPodFn)
	return err
}

func (o *DrainCmdOptions) waitForDelete(pods []corev1.Pod, interval, timeout time.Duration, usingEviction bool, getPodFn func(string, string) (*corev1.Pod, error)) ([]corev1.Pod, error) {
	var verbStr string
	if usingEviction {
		verbStr = "evicted"
	} else {
		verbStr = "deleted"
	}
	printObj, err := o.ToPrinter(verbStr)
	if err != nil {
		return pods, err
	}

	err = wait.PollImmediate(interval, timeout, func() (bool, error) {
		pendingPods := []corev1.Pod{}
		for i, pod := range pods {
			p, err := getPodFn(pod.Namespace, pod.Name)
			if apierrors.IsNotFound(err) || (p != nil && p.ObjectMeta.UID != pod.ObjectMeta.UID) {
				printObj(&pod, o.Out)
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

// RunCordonOrUncordon runs either Cordon or Uncordon.  The desired value for
// "Unschedulable" is passed as the first arg.
func (o *DrainCmdOptions) RunCordonOrUncordon(desired bool) error {
	cordonOrUncordon := "cordon"
	if !desired {
		cordonOrUncordon = "un" + cordonOrUncordon
	}

	for _, nodeInfo := range o.nodeInfos {

		printError := func(err error) {
			fmt.Fprintf(o.ErrOut, "error: unable to %s node %q: %v\n", cordonOrUncordon, nodeInfo.Name, err)
		}

		gvk := nodeInfo.ResourceMapping().GroupVersionKind
		if gvk.Kind == "Node" {
			c, err := drain.NewCordonHelperFromRuntimeObject(nodeInfo.Object, scheme.Scheme, gvk)
			if err != nil {
				printError(err)
				continue
			}

			if updateRequired := c.UpdateIfRequired(desired); !updateRequired {
				printObj, err := o.ToPrinter(already(desired))
				if err != nil {
					fmt.Fprintf(o.ErrOut, "error: %v\n", err)
					continue
				}
				printObj(nodeInfo.Object, o.Out)
			} else {
				if !o.drainer.DryRun {
					err, patchErr := c.PatchOrReplace(o.drainer.Client)
					if patchErr != nil {
						printError(patchErr)
					}
					if err != nil {
						printError(err)
						continue
					}
				}
				printObj, err := o.ToPrinter(changed(desired))
				if err != nil {
					fmt.Fprintf(o.ErrOut, "%v\n", err)
					continue
				}
				printObj(nodeInfo.Object, o.Out)
			}
		} else {
			printObj, err := o.ToPrinter("skipped")
			if err != nil {
				fmt.Fprintf(o.ErrOut, "%v\n", err)
				continue
			}
			printObj(nodeInfo.Object, o.Out)
		}
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
