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
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/cmd/attach"
	"k8s.io/kubectl/pkg/cmd/exec"
	"k8s.io/kubectl/pkg/cmd/logs"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/interrupt"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	debugLong = templates.LongDesc(i18n.T(`Tools for debugging Kubernetes resources`))

	debugExample = templates.Examples(i18n.T(`
		# Create an interactive debugging session in pod mypod and immediately attach to it.
		# (requires the EphemeralContainers feature to be enabled in the cluster)
		kubectl alpha debug mypod -i --image=busybox

		# Create a debug container named debugger using a custom automated debugging image.
		# (requires the EphemeralContainers feature to be enabled in the cluster)
		kubectl alpha debug --image=myproj/debug-tools -c debugger mypod`))
)

var nameSuffixFunc = utilrand.String

// DebugOptions holds the options for an invocation of kubectl debug.
type DebugOptions struct {
	Args        []string
	ArgsOnly    bool
	Attach      bool
	Container   string
	Env         []corev1.EnvVar
	Image       string
	Interactive bool
	Namespace   string
	TargetNames []string
	PullPolicy  corev1.PullPolicy
	Quiet       bool
	Target      string
	TTY         bool

	builder   *resource.Builder
	podClient corev1client.PodsGetter

	genericclioptions.IOStreams
}

// NewDebugOptions returns a DebugOptions initialized with default values.
func NewDebugOptions(streams genericclioptions.IOStreams) *DebugOptions {
	return &DebugOptions{
		Args:        []string{},
		IOStreams:   streams,
		TargetNames: []string{},
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
	cmd.Flags().BoolVar(&opt.Attach, "attach", opt.Attach, i18n.T("If true, wait for the Pod to start running, and then attach to the Pod as if 'kubectl attach ...' were called.  Default false, unless '-i/--stdin' is set, in which case the default is true."))
	cmd.Flags().StringVarP(&opt.Container, "container", "c", opt.Container, i18n.T("Container name to use for debug container."))
	cmd.Flags().StringToString("env", nil, i18n.T("Environment variables to set in the container."))
	cmd.Flags().StringVar(&opt.Image, "image", opt.Image, i18n.T("Container image to use for debug container."))
	cmd.MarkFlagRequired("image")
	cmd.Flags().String("image-pull-policy", string(corev1.PullIfNotPresent), i18n.T("The image pull policy for the container."))
	cmd.Flags().BoolVarP(&opt.Interactive, "stdin", "i", opt.Interactive, i18n.T("Keep stdin open on the container(s) in the pod, even if nothing is attached."))
	cmd.Flags().BoolVar(&opt.Quiet, "quiet", opt.Quiet, i18n.T("If true, suppress prompt messages."))
	cmd.Flags().StringVar(&opt.Target, "target", "", i18n.T("Target processes in this container name."))
	cmd.Flags().BoolVarP(&opt.TTY, "tty", "t", opt.TTY, i18n.T("Allocated a TTY for each container in the pod."))
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

// visitPod handles debugging for pod targets by (depending on options):
//   1. Creating an ephemeral debug container in an existing pod, OR
//   2. Making a copy of pod with certain attributes changed (NOT YET IMPLEMENTED)
// visitPod returns a pod and debug container name for subsequent attach, if applicable.
func (o *DebugOptions) visitPod(ctx context.Context, pod *corev1.Pod) (*corev1.Pod, string, error) {
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

func containerNames(pod *corev1.Pod) map[string]bool {
	names := map[string]bool{}
	for _, c := range pod.Spec.Containers {
		names[c.Name] = true
	}
	for _, c := range pod.Spec.InitContainers {
		names[c.Name] = true
	}
	for _, c := range pod.Spec.EphemeralContainers {
		names[c.Name] = true
	}
	return names
}

// generateDebugContainer returns an EphemeralContainer suitable for use as a debug container
// in the given pod.
func (o *DebugOptions) generateDebugContainer(pod *corev1.Pod) *corev1.EphemeralContainer {
	name := o.Container
	if len(name) == 0 {
		cn, existing := "", containerNames(pod)
		for len(cn) == 0 || existing[cn] {
			cn = fmt.Sprintf("debugger-%s", nameSuffixFunc(5))
		}
		if !o.Quiet {
			fmt.Fprintf(o.ErrOut, "Defaulting debug container name to %s.\n", cn)
		}
		name = cn
	}

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

// waitForEphemeralContainer watches the given pod until the ephemeralContainer is running
func waitForEphemeralContainer(ctx context.Context, podClient corev1client.PodsGetter, ns, podName, ephemeralContainerName string) (*corev1.Pod, error) {
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

			for _, s := range p.Status.EphemeralContainerStatuses {
				if s.Name != ephemeralContainerName {
					continue
				}

				klog.V(2).Infof("debug container status is %v", s)
				if s.State.Running != nil || s.State.Terminated != nil {
					return true, nil
				}
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

// TODO(verb): handle other types of containers
func handleAttachPod(ctx context.Context, f cmdutil.Factory, podClient corev1client.PodsGetter, ns, podName, ephemeralContainerName string, opts *attach.AttachOptions) error {
	pod, err := waitForEphemeralContainer(ctx, podClient, ns, podName, ephemeralContainerName)
	if err != nil {
		return err
	}

	opts.Namespace = ns
	opts.Pod = pod
	opts.PodName = podName
	opts.ContainerName = ephemeralContainerName
	if opts.AttachFunc == nil {
		opts.AttachFunc = attach.DefaultAttachFunc
	}

	var status *corev1.ContainerStatus
	for i := range pod.Status.EphemeralContainerStatuses {
		if pod.Status.EphemeralContainerStatuses[i].Name == ephemeralContainerName {
			status = &pod.Status.EphemeralContainerStatuses[i]
		}
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
