/*
Copyright 2020 The Kubernetes Authors.

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

package debug

import (
	"context"
	"fmt"
	"time"

	"github.com/docker/distribution/reference"
	"github.com/spf13/cobra"
	"k8s.io/klog/v2"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubectl/pkg/cmd/attach"
	"k8s.io/kubectl/pkg/cmd/exec"
	"k8s.io/kubectl/pkg/cmd/logs"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/interrupt"
	"k8s.io/kubectl/pkg/util/templates"
	"k8s.io/utils/pointer"
)

var (
	debugLong = templates.LongDesc(i18n.T(`
		Debug cluster resources using interactive debugging containers.

		'debug' provides automation for common debugging tasks for cluster objects identified by
		resource and name. Pods will be used by default if resource is not specified.

		The action taken by 'debug' varies depending on what resource is specified. Supported
		actions include:

		* Workload: Create a copy of an existing pod with certain attributes changed,
	                for example changing the image tag to a new version.
		* Workload: Add an ephemeral container to an already running pod, for example to add
		            debugging utilities without restarting the pod.
		* Node: Create a new pod that runs in the node's host namespaces and can access
		        the node's filesystem.

		Alpha disclaimer: command line flags may change`))

	debugExample = templates.Examples(i18n.T(`
		# Create an interactive debugging session in pod mypod and immediately attach to it.
		# (requires the EphemeralContainers feature to be enabled in the cluster)
		kubectl alpha debug mypod -it --image=busybox

		# Create a debug container named debugger using a custom automated debugging image.
		# (requires the EphemeralContainers feature to be enabled in the cluster)
		kubectl alpha debug --image=myproj/debug-tools -c debugger mypod

		# Create a debug container as a copy of the original Pod and attach to it
		kubectl alpha debug mypod -it --image=busybox --copy-to=my-debugger

		# Create a copy of mypod named my-debugger with my-container's image changed to busybox
		kubectl alpha debug mypod --image=busybox --container=my-container --copy-to=my-debugger -- sleep 1d

		# Create an interactive debugging session on a node and immediately attach to it.
		# The container will run in the host namespaces and the host's filesystem will be mounted at /host
		kubectl alpha debug node/mynode -it --image=busybox
`))
)

var nameSuffixFunc = utilrand.String

// DebugOptions holds the options for an invocation of kubectl debug.
type DebugOptions struct {
	Args           []string
	ArgsOnly       bool
	Attach         bool
	Container      string
	CopyTo         string
	Replace        bool
	Env            []corev1.EnvVar
	Image          string
	Interactive    bool
	Namespace      string
	TargetNames    []string
	PullPolicy     corev1.PullPolicy
	Quiet          bool
	SameNode       bool
	ShareProcesses bool
	Target         string
	TTY            bool

	shareProcessedChanged bool

	builder   *resource.Builder
	podClient corev1client.PodsGetter

	genericclioptions.IOStreams
}

// NewDebugOptions returns a DebugOptions initialized with default values.
func NewDebugOptions(streams genericclioptions.IOStreams) *DebugOptions {
	return &DebugOptions{
		Args:           []string{},
		IOStreams:      streams,
		TargetNames:    []string{},
		ShareProcesses: true,
	}
}

// NewCmdDebug returns a cobra command that runs kubectl debug.
func NewCmdDebug(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewDebugOptions(streams)

	cmd := &cobra.Command{
		Use:                   "debug NAME --image=image [ -- COMMAND [args...] ]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Attach a debug container to a running pod"),
		Long:                  debugLong,
		Example:               debugExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate(cmd))
			cmdutil.CheckErr(o.Run(f, cmd))
		},
	}

	addDebugFlags(cmd, o)
	return cmd
}

func addDebugFlags(cmd *cobra.Command, opt *DebugOptions) {
	cmd.Flags().BoolVar(&opt.ArgsOnly, "arguments-only", opt.ArgsOnly, i18n.T("If specified, everything after -- will be passed to the new container as Args instead of Command."))
	cmd.Flags().BoolVar(&opt.Attach, "attach", opt.Attach, i18n.T("If true, wait for the container to start running, and then attach as if 'kubectl attach ...' were called.  Default false, unless '-i/--stdin' is set, in which case the default is true."))
	cmd.Flags().StringVarP(&opt.Container, "container", "c", opt.Container, i18n.T("Container name to use for debug container."))
	cmd.Flags().StringVar(&opt.CopyTo, "copy-to", opt.CopyTo, i18n.T("Create a copy of the target Pod with this name."))
	cmd.Flags().BoolVar(&opt.Replace, "replace", opt.Replace, i18n.T("When used with '--copy-to', delete the original Pod"))
	cmd.Flags().StringToString("env", nil, i18n.T("Environment variables to set in the container."))
	cmd.Flags().StringVar(&opt.Image, "image", opt.Image, i18n.T("Container image to use for debug container."))
	cmd.MarkFlagRequired("image")
	cmd.Flags().String("image-pull-policy", string(corev1.PullIfNotPresent), i18n.T("The image pull policy for the container."))
	cmd.Flags().BoolVarP(&opt.Interactive, "stdin", "i", opt.Interactive, i18n.T("Keep stdin open on the container(s) in the pod, even if nothing is attached."))
	cmd.Flags().BoolVar(&opt.Quiet, "quiet", opt.Quiet, i18n.T("If true, suppress informational messages."))
	cmd.Flags().BoolVar(&opt.SameNode, "same-node", opt.SameNode, i18n.T("When used with '--copy-to', schedule the copy of target Pod on the same node."))
	cmd.Flags().BoolVar(&opt.ShareProcesses, "share-processes", opt.ShareProcesses, i18n.T("When used with '--copy-to', enable process namespace sharing in the copy."))
	cmd.Flags().StringVar(&opt.Target, "target", "", i18n.T("When debugging a pod, target processes in this container name."))
	cmd.Flags().BoolVarP(&opt.TTY, "tty", "t", opt.TTY, i18n.T("Allocate a TTY for the debugging container."))
}

// Complete finishes run-time initialization of debug.DebugOptions.
func (o *DebugOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	o.builder = f.NewBuilder()
	o.PullPolicy = corev1.PullPolicy(cmdutil.GetFlagString(cmd, "image-pull-policy"))

	// Arguments
	argsLen := cmd.ArgsLenAtDash()
	o.TargetNames = args
	// If there is a dash and there are args after the dash, extract the args.
	if argsLen >= 0 && len(args) > argsLen {
		o.TargetNames, o.Args = args[:argsLen], args[argsLen:]
	}

	// Attach
	attachFlag := cmd.Flags().Lookup("attach")
	if !attachFlag.Changed && o.Interactive {
		o.Attach = true
	}

	// Environment
	envStrings, err := cmd.Flags().GetStringToString("env")
	if err != nil {
		return fmt.Errorf("internal error getting env flag: %v", err)
	}
	for k, v := range envStrings {
		o.Env = append(o.Env, corev1.EnvVar{Name: k, Value: v})
	}

	// Namespace
	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	// Clientset
	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return fmt.Errorf("internal error getting clientset: %v", err)
	}
	o.podClient = clientset.CoreV1()

	// Share processes
	o.shareProcessedChanged = cmd.Flags().Changed("share-processes")

	return nil
}

// Validate checks that the provided debug options are specified.
func (o *DebugOptions) Validate(cmd *cobra.Command) error {
	// Image
	if len(o.Image) == 0 {
		return fmt.Errorf("--image is required")
	}
	if !reference.ReferenceRegexp.MatchString(o.Image) {
		return fmt.Errorf("Invalid image name %q: %v", o.Image, reference.ErrReferenceInvalidFormat)
	}

	// Name
	if len(o.TargetNames) == 0 {
		return fmt.Errorf("NAME is required for debug")
	}

	// Pull Policy
	switch o.PullPolicy {
	case corev1.PullAlways, corev1.PullIfNotPresent, corev1.PullNever, "":
		// continue
	default:
		return fmt.Errorf("invalid image pull policy: %s", o.PullPolicy)
	}

	// Target
	if len(o.Target) > 0 && len(o.CopyTo) > 0 {
		return fmt.Errorf("--target is incompatible with --copy-to. Use --share-processes instead.")
	}

	// TTY
	if o.TTY && !o.Interactive {
		return fmt.Errorf("-i/--stdin is required for containers with -t/--tty=true")
	}

	return nil
}

// Run executes a kubectl debug.
func (o *DebugOptions) Run(f cmdutil.Factory, cmd *cobra.Command) error {
	ctx := context.Background()

	r := o.builder.
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		NamespaceParam(o.Namespace).DefaultNamespace().ResourceNames("pods", o.TargetNames...).
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			// TODO(verb): configurable early return
			return err
		}

		var (
			debugPod      *corev1.Pod
			containerName string
			visitErr      error
		)
		switch obj := info.Object.(type) {
		case *corev1.Node:
			debugPod, containerName, visitErr = o.visitNode(ctx, obj)
		case *corev1.Pod:
			debugPod, containerName, visitErr = o.visitPod(ctx, obj)
		default:
			visitErr = fmt.Errorf("%q not supported by debug", info.Mapping.GroupVersionKind)
		}
		if visitErr != nil {
			return visitErr
		}

		if o.Attach {
			opts := &attach.AttachOptions{
				StreamOptions: exec.StreamOptions{
					IOStreams: o.IOStreams,
					Stdin:     o.Interactive,
					TTY:       o.TTY,
					Quiet:     o.Quiet,
				},
				// TODO(verb): kubectl prints an incorrect "Session ended" message for debug containers.
				CommandName: cmd.Parent().CommandPath() + " attach",

				Attach: &attach.DefaultRemoteAttach{},
			}
			config, err := f.ToRESTConfig()
			if err != nil {
				return err
			}
			opts.Config = config
			opts.AttachFunc = attach.DefaultAttachFunc

			if err := handleAttachPod(ctx, f, o.podClient, debugPod.Namespace, debugPod.Name, containerName, opts); err != nil {
				return err
			}
		}

		return nil
	})

	return err
}

// visitNode handles debugging for node targets by creating a privileged pod running in the host namespaces.
// Returns an already created pod and container name for subsequent attach, if applicable.
func (o *DebugOptions) visitNode(ctx context.Context, node *corev1.Node) (*corev1.Pod, string, error) {
	pods := o.podClient.Pods(o.Namespace)
	newPod, err := pods.Create(ctx, o.generateNodeDebugPod(node.Name), metav1.CreateOptions{})
	if err != nil {
		return nil, "", err
	}

	return newPod, newPod.Spec.Containers[0].Name, nil
}

// visitPod handles debugging for pod targets by (depending on options):
//   1. Creating an ephemeral debug container in an existing pod, OR
//   2. Making a copy of pod with certain attributes changed
// visitPod returns a pod and debug container name for subsequent attach, if applicable.
func (o *DebugOptions) visitPod(ctx context.Context, pod *corev1.Pod) (*corev1.Pod, string, error) {
	if len(o.CopyTo) > 0 {
		return o.debugByCopy(ctx, pod)
	}
	return o.debugByEphemeralContainer(ctx, pod)
}

// debugByEphemeralContainer runs an EphemeralContainer in the target Pod for use as a debug container
func (o *DebugOptions) debugByEphemeralContainer(ctx context.Context, pod *corev1.Pod) (*corev1.Pod, string, error) {
	pods := o.podClient.Pods(pod.Namespace)
	ec, err := pods.GetEphemeralContainers(ctx, pod.Name, metav1.GetOptions{})
	if err != nil {
		// The pod has already been fetched at this point, so a NotFound error indicates the ephemeralcontainers subresource wasn't found.
		if serr, ok := err.(*errors.StatusError); ok && serr.Status().Reason == metav1.StatusReasonNotFound {
			return nil, "", fmt.Errorf("ephemeral containers are disabled for this cluster (error from server: %q).", err)
		}
		return nil, "", err
	}
	klog.V(2).Infof("existing ephemeral containers: %v", ec.EphemeralContainers)

	debugContainer := o.generateDebugContainer(pod)
	klog.V(2).Infof("new ephemeral container: %#v", debugContainer)
	ec.EphemeralContainers = append(ec.EphemeralContainers, *debugContainer)
	_, err = pods.UpdateEphemeralContainers(ctx, pod.Name, ec, metav1.UpdateOptions{})
	if err != nil {
		return nil, "", fmt.Errorf("error updating ephemeral containers: %v", err)
	}

	return pod, debugContainer.Name, nil
}

// debugByCopy runs a copy of the target Pod with a debug container added or an original container modified
func (o *DebugOptions) debugByCopy(ctx context.Context, pod *corev1.Pod) (*corev1.Pod, string, error) {
	copied, dc := o.generatePodCopyWithDebugContainer(pod)
	copied, err := o.podClient.Pods(copied.Namespace).Create(ctx, copied, metav1.CreateOptions{})
	if err != nil {
		return nil, "", err
	}
	if o.Replace {
		err := o.podClient.Pods(pod.Namespace).Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
		if err != nil {
			return nil, "", err
		}
	}
	return copied, dc, nil
}

// generateDebugContainer returns an EphemeralContainer suitable for use as a debug container
// in the given pod.
func (o *DebugOptions) generateDebugContainer(pod *corev1.Pod) *corev1.EphemeralContainer {
	name := o.computeDebugContainerName(pod)

	ec := &corev1.EphemeralContainer{
		EphemeralContainerCommon: corev1.EphemeralContainerCommon{
			Name:                     name,
			Env:                      o.Env,
			Image:                    o.Image,
			ImagePullPolicy:          o.PullPolicy,
			Stdin:                    o.Interactive,
			TerminationMessagePolicy: corev1.TerminationMessageReadFile,
			TTY:                      o.TTY,
		},
		TargetContainerName: o.Target,
	}

	if o.ArgsOnly {
		ec.Args = o.Args
	} else {
		ec.Command = o.Args
	}

	return ec
}

// generateNodeDebugPod generates a debugging pod that schedules on the specified node.
// The generated pod will run in the host PID, Network & IPC namespaces, and it will have the node's filesystem mounted at /host.
func (o *DebugOptions) generateNodeDebugPod(node string) *corev1.Pod {
	cn := "debugger"
	// Setting a user-specified container name doesn't make much difference when there's only one container,
	// but the argument exists for pod debugging so it might be confusing if it didn't work here.
	if len(o.Container) > 0 {
		cn = o.Container
	}

	// The name of the debugging pod is based on the target node, and it's not configurable to
	// limit the number of command line flags. There may be a collision on the name, but this
	// should be rare enough that it's not worth the API round trip to check.
	pn := fmt.Sprintf("node-debugger-%s-%s", node, nameSuffixFunc(5))
	if !o.Quiet {
		fmt.Fprintf(o.Out, "Creating debugging pod %s with container %s on node %s.\n", pn, cn, node)
	}

	p := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: pn,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:                     cn,
					Env:                      o.Env,
					Image:                    o.Image,
					ImagePullPolicy:          o.PullPolicy,
					Stdin:                    o.Interactive,
					TerminationMessagePolicy: corev1.TerminationMessageReadFile,
					TTY:                      o.TTY,
					VolumeMounts: []corev1.VolumeMount{
						{
							MountPath: "/host",
							Name:      "host-root",
						},
					},
				},
			},
			HostIPC:       true,
			HostNetwork:   true,
			HostPID:       true,
			NodeName:      node,
			RestartPolicy: corev1.RestartPolicyNever,
			Volumes: []corev1.Volume{
				{
					Name: "host-root",
					VolumeSource: corev1.VolumeSource{
						HostPath: &corev1.HostPathVolumeSource{Path: "/"},
					},
				},
			},
		},
	}

	if o.ArgsOnly {
		p.Spec.Containers[0].Args = o.Args
	} else {
		p.Spec.Containers[0].Command = o.Args
	}

	return p
}

// generatePodCopy takes a Pod and returns a copy and the debug container name of that copy
func (o *DebugOptions) generatePodCopyWithDebugContainer(pod *corev1.Pod) (*corev1.Pod, string) {
	copied := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        o.CopyTo,
			Namespace:   pod.Namespace,
			Annotations: pod.Annotations,
		},
		Spec: *pod.Spec.DeepCopy(),
	}
	// set EphemeralContainers to nil so that the copy of pod can be created
	copied.Spec.EphemeralContainers = nil
	// change ShareProcessNamespace configuration only when commanded explicitly
	if o.shareProcessedChanged {
		copied.Spec.ShareProcessNamespace = pointer.BoolPtr(o.ShareProcesses)
	}
	if !o.SameNode {
		copied.Spec.NodeName = ""
	}

	containerByName := containerNameToRef(copied)
	c, containerExists := containerByName[o.Container]
	// Add a new container if the specified container does not exist
	if !containerExists {
		name := o.computeDebugContainerName(copied)
		c = &corev1.Container{Name: name}
		// envs are customizable when adding new container
		c.Env = o.Env
	}
	c.Image = o.Image
	c.ImagePullPolicy = o.PullPolicy
	c.Stdin = o.Interactive
	c.TerminationMessagePolicy = corev1.TerminationMessageReadFile
	c.TTY = o.TTY
	if o.ArgsOnly {
		c.Args = o.Args
	} else {
		c.Command = o.Args
		c.Args = nil
	}
	if !containerExists {
		copied.Spec.Containers = append(copied.Spec.Containers, *c)
	}
	return copied, c.Name
}

func (o *DebugOptions) computeDebugContainerName(pod *corev1.Pod) string {
	if len(o.Container) > 0 {
		return o.Container
	}
	name := o.Container
	if len(name) == 0 {
		cn, containerByName := "", containerNameToRef(pod)
		for len(cn) == 0 || (containerByName[cn] != nil) {
			cn = fmt.Sprintf("debugger-%s", nameSuffixFunc(5))
		}
		if !o.Quiet {
			fmt.Fprintf(o.Out, "Defaulting debug container name to %s.\n", cn)
		}
		name = cn
	}
	return name
}

func containerNameToRef(pod *corev1.Pod) map[string]*corev1.Container {
	names := map[string]*corev1.Container{}
	for i := range pod.Spec.Containers {
		ref := &pod.Spec.Containers[i]
		names[ref.Name] = ref
	}
	for i := range pod.Spec.InitContainers {
		ref := &pod.Spec.InitContainers[i]
		names[ref.Name] = ref
	}
	for i := range pod.Spec.EphemeralContainers {
		ref := (*corev1.Container)(&pod.Spec.EphemeralContainers[i].EphemeralContainerCommon)
		names[ref.Name] = ref
	}
	return names
}

// waitForContainer watches the given pod until the container is running
func waitForContainer(ctx context.Context, podClient corev1client.PodsGetter, ns, podName, containerName string) (*corev1.Pod, error) {
	// TODO: expose the timeout
	ctx, cancel := watchtools.ContextWithOptionalTimeout(ctx, 0*time.Second)
	defer cancel()

	fieldSelector := fields.OneTermEqualSelector("metadata.name", podName).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return podClient.Pods(ns).List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return podClient.Pods(ns).Watch(ctx, options)
		},
	}

	intr := interrupt.New(nil, cancel)
	var result *corev1.Pod
	err := intr.Run(func() error {
		ev, err := watchtools.UntilWithSync(ctx, lw, &corev1.Pod{}, nil, func(ev watch.Event) (bool, error) {
			switch ev.Type {
			case watch.Deleted:
				return false, errors.NewNotFound(schema.GroupResource{Resource: "pods"}, "")
			}

			p, ok := ev.Object.(*corev1.Pod)
			if !ok {
				return false, fmt.Errorf("watch did not return a pod: %v", ev.Object)
			}

			s := getContainerStatusByName(p, containerName)
			if s == nil {
				return false, nil
			}
			klog.V(2).Infof("debug container status is %v", s)
			if s.State.Running != nil || s.State.Terminated != nil {
				return true, nil
			}
			return false, nil
		})
		if ev != nil {
			result = ev.Object.(*corev1.Pod)
		}
		return err
	})

	return result, err
}

func handleAttachPod(ctx context.Context, f cmdutil.Factory, podClient corev1client.PodsGetter, ns, podName, containerName string, opts *attach.AttachOptions) error {
	pod, err := waitForContainer(ctx, podClient, ns, podName, containerName)
	if err != nil {
		return err
	}

	opts.Namespace = ns
	opts.Pod = pod
	opts.PodName = podName
	opts.ContainerName = containerName
	if opts.AttachFunc == nil {
		opts.AttachFunc = attach.DefaultAttachFunc
	}

	status := getContainerStatusByName(pod, containerName)
	if status == nil {
		// impossible path
		return fmt.Errorf("Error get container status of %s: %+v", containerName, err)
	}
	if status.State.Terminated != nil {
		klog.V(1).Info("Ephemeral container terminated, falling back to logs")
		return logOpts(f, pod, opts)
	}

	if err := opts.Run(); err != nil {
		fmt.Fprintf(opts.ErrOut, "Error attaching, falling back to logs: %v\n", err)
		return logOpts(f, pod, opts)
	}
	return nil
}

func getContainerStatusByName(pod *corev1.Pod, containerName string) *corev1.ContainerStatus {
	allContainerStatus := [][]corev1.ContainerStatus{pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses, pod.Status.EphemeralContainerStatuses}
	for _, statusSlice := range allContainerStatus {
		for i := range statusSlice {
			if statusSlice[i].Name == containerName {
				return &statusSlice[i]
			}
		}
	}
	return nil
}

// logOpts logs output from opts to the pods log.
func logOpts(restClientGetter genericclioptions.RESTClientGetter, pod *corev1.Pod, opts *attach.AttachOptions) error {
	ctrName, err := opts.GetContainerName(pod)
	if err != nil {
		return err
	}

	requests, err := polymorphichelpers.LogsForObjectFn(restClientGetter, pod, &corev1.PodLogOptions{Container: ctrName}, opts.GetPodTimeout, false)
	if err != nil {
		return err
	}
	for _, request := range requests {
		if err := logs.DefaultConsumeRequest(request, opts.Out); err != nil {
			return err
		}
	}

	return nil
}
