/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io"
	"os"
	"time"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"github.com/docker/distribution/reference"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	batchv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	uexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/watch"
)

var (
	run_long = dedent.Dedent(`
		Create and run a particular image, possibly replicated.
		Creates a deployment or job to manage the created container(s).`)
	run_example = dedent.Dedent(`
		# Start a single instance of nginx.
		kubectl run nginx --image=nginx

		# Start a single instance of hazelcast and let the container expose port 5701 .
		kubectl run hazelcast --image=hazelcast --port=5701

		# Start a single instance of hazelcast and set environment variables "DNS_DOMAIN=cluster" and "POD_NAMESPACE=default" in the container.
		kubectl run hazelcast --image=hazelcast --env="DNS_DOMAIN=cluster" --env="POD_NAMESPACE=default"

		# Start a replicated instance of nginx.
		kubectl run nginx --image=nginx --replicas=5

		# Dry run. Print the corresponding API objects without creating them.
		kubectl run nginx --image=nginx --dry-run

		# Start a single instance of nginx, but overload the spec of the deployment with a partial set of values parsed from JSON.
		kubectl run nginx --image=nginx --overrides='{ "apiVersion": "v1", "spec": { ... } }'

		# Start a pod of busybox and keep it in the foreground, don't restart it if it exits.
		kubectl run -i -t busybox --image=busybox --restart=Never

		# Start the nginx container using the default command, but use custom arguments (arg1 .. argN) for that command.
		kubectl run nginx --image=nginx -- <arg1> <arg2> ... <argN>

		# Start the nginx container using a different command and custom arguments.
		kubectl run nginx --image=nginx --command -- <cmd> <arg1> ... <argN>

		# Start the perl container to compute π to 2000 places and print it out.
		kubectl run pi --image=perl --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(2000)'

		# Start the scheduled job to compute π to 2000 places and print it out every 5 minutes.
		kubectl run pi --schedule="0/5 * * * ?" --image=perl --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(2000)'`)
)

func NewCmdRun(f *cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use: "run NAME --image=image [--env=\"key=value\"] [--port=port] [--replicas=replicas] [--dry-run=bool] [--overrides=inline-json] [--command] -- [COMMAND] [args...]",
		// run-container is deprecated
		Aliases: []string{"run-container"},
		Short:   "Run a particular image on the cluster",
		Long:    run_long,
		Example: run_example,
		Run: func(cmd *cobra.Command, args []string) {
			argsLenAtDash := cmd.ArgsLenAtDash()
			err := Run(f, cmdIn, cmdOut, cmdErr, cmd, args, argsLenAtDash)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddPrinterFlags(cmd)
	addRunFlags(cmd)
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

func addRunFlags(cmd *cobra.Command) {
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().String("generator", "", "The name of the API generator to use.  Default is 'deployment/v1beta1' if --restart=Always, 'job/v1' for OnFailure and 'run-pod/v1' for Never.  This will happen only for cluster version at least 1.3, for 1.2 we will fallback to 'deployment/v1beta1' for --restart=Always, 'job/v1' for others, for olders we will fallback to 'run/v1' for --restart=Always, 'run-pod/v1' for others.")
	cmd.Flags().String("image", "", "The image for the container to run.")
	cmd.MarkFlagRequired("image")
	cmd.Flags().String("image-pull-policy", "", "The image pull policy for the container. If left empty, this value will not be specified by the client and defaulted by the server")
	cmd.Flags().IntP("replicas", "r", 1, "Number of replicas to create for this container. Default is 1.")
	cmd.Flags().Bool("rm", false, "If true, delete resources created in this command for attached containers.")
	cmd.Flags().String("overrides", "", "An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.")
	cmd.Flags().StringSlice("env", []string{}, "Environment variables to set in the container")
	cmd.Flags().Int("port", -1, "The port that this container exposes.  If --expose is true, this is also the port used by the service that is created.")
	cmd.Flags().Int("hostport", -1, "The host port mapping for the container port. To demonstrate a single-machine container.")
	cmd.Flags().StringP("labels", "l", "", "Labels to apply to the pod(s).")
	cmd.Flags().BoolP("stdin", "i", false, "Keep stdin open on the container(s) in the pod, even if nothing is attached.")
	cmd.Flags().BoolP("tty", "t", false, "Allocated a TTY for each container in the pod.")
	cmd.Flags().Bool("attach", false, "If true, wait for the Pod to start running, and then attach to the Pod as if 'kubectl attach ...' were called.  Default false, unless '-i/--stdin' is set, in which case the default is true. With '--restart=Never' the exit code of the container process is returned.")
	cmd.Flags().Bool("leave-stdin-open", false, "If the pod is started in interactive mode or with stdin, leave stdin open after the first attach completes. By default, stdin will be closed after the first attach completes.")
	cmd.Flags().String("restart", "Always", "The restart policy for this Pod.  Legal values [Always, OnFailure, Never].  If set to 'Always' a deployment is created for this pod, if set to 'OnFailure', a job is created for this pod, if set to 'Never', a regular pod is created. For the latter two --replicas must be 1.  Default 'Always'")
	cmd.Flags().Bool("command", false, "If true and extra arguments are present, use them as the 'command' field in the container, rather than the 'args' field which is the default.")
	cmd.Flags().String("requests", "", "The resource requirement requests for this container.  For example, 'cpu=100m,memory=256Mi'.  Note that server side components may assign requests depending on the server configuration, such as limit ranges.")
	cmd.Flags().String("limits", "", "The resource requirement limits for this container.  For example, 'cpu=200m,memory=512Mi'.  Note that server side components may assign limits depending on the server configuration, such as limit ranges.")
	cmd.Flags().Bool("expose", false, "If true, a public, external service is created for the container(s) which are run")
	cmd.Flags().String("service-generator", "service/v2", "The name of the generator to use for creating a service.  Only used if --expose is true")
	cmd.Flags().String("service-overrides", "", "An inline JSON override for the generated service object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field.  Only used if --expose is true.")
	cmd.Flags().Bool("quiet", false, "If true, suppress prompt messages.")
	cmd.Flags().String("schedule", "", "A schedule in the Cron format the job should be run with.")
}

func Run(f *cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer, cmd *cobra.Command, args []string, argsLenAtDash int) error {
	if len(os.Args) > 1 && os.Args[1] == "run-container" {
		printDeprecationWarning("run", "run-container")
	}

	// Let kubectl run follow rules for `--`, see #13004 issue
	if len(args) == 0 || argsLenAtDash == 0 {
		return cmdutil.UsageError(cmd, "NAME is required for run")
	}

	// validate image name
	imageName := cmdutil.GetFlagString(cmd, "image")
	validImageRef := reference.ReferenceRegexp.MatchString(imageName)
	if !validImageRef {
		return fmt.Errorf("Invalid image name %q: %v", imageName, reference.ErrReferenceInvalidFormat)
	}

	interactive := cmdutil.GetFlagBool(cmd, "stdin")
	tty := cmdutil.GetFlagBool(cmd, "tty")
	if tty && !interactive {
		return cmdutil.UsageError(cmd, "-i/--stdin is required for containers with -t/--tty=true")
	}
	replicas := cmdutil.GetFlagInt(cmd, "replicas")
	if interactive && replicas != 1 {
		return cmdutil.UsageError(cmd, fmt.Sprintf("-i/--stdin requires that replicas is 1, found %d", replicas))
	}

	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	restartPolicy, err := getRestartPolicy(cmd, interactive)
	if err != nil {
		return err
	}
	if restartPolicy != api.RestartPolicyAlways && replicas != 1 {
		return cmdutil.UsageError(cmd, fmt.Sprintf("--restart=%s requires that --replicas=1, found %d", restartPolicy, replicas))
	}

	if err := verifyImagePullPolicy(cmd); err != nil {
		return err
	}

	generatorName := cmdutil.GetFlagString(cmd, "generator")
	schedule := cmdutil.GetFlagString(cmd, "schedule")
	if len(schedule) != 0 && len(generatorName) == 0 {
		generatorName = "scheduledjob/v2alpha1"
	}
	if len(generatorName) == 0 {
		client, err := f.Client()
		if err != nil {
			return err
		}
		resourcesList, err := client.Discovery().ServerResources()
		// ServerResources ignores errors for old servers do not expose discovery
		if err != nil {
			return fmt.Errorf("failed to discover supported resources: %v", err)
		}
		switch restartPolicy {
		case api.RestartPolicyAlways:
			if contains(resourcesList, v1beta1.SchemeGroupVersion.WithResource("deployments")) {
				generatorName = "deployment/v1beta1"
			} else {
				generatorName = "run/v1"
			}
		case api.RestartPolicyOnFailure:
			if contains(resourcesList, batchv1.SchemeGroupVersion.WithResource("jobs")) {
				generatorName = "job/v1"
			} else if contains(resourcesList, v1beta1.SchemeGroupVersion.WithResource("jobs")) {
				generatorName = "job/v1beta1"
			} else {
				generatorName = "run-pod/v1"
			}
		case api.RestartPolicyNever:
			generatorName = "run-pod/v1"
		}
	}
	generators := f.Generators("run")
	generator, found := generators[generatorName]
	if !found {
		return cmdutil.UsageError(cmd, fmt.Sprintf("generator %q not found.", generatorName))
	}
	names := generator.ParamNames()
	params := kubectl.MakeParams(cmd, names)
	params["name"] = args[0]
	if len(args) > 1 {
		params["args"] = args[1:]
	}

	params["env"] = cmdutil.GetFlagStringSlice(cmd, "env")

	obj, _, mapper, mapping, err := createGeneratedObject(f, cmd, generator, names, params, cmdutil.GetFlagString(cmd, "overrides"), namespace)
	if err != nil {
		return err
	}

	if cmdutil.GetFlagBool(cmd, "expose") {
		serviceGenerator := cmdutil.GetFlagString(cmd, "service-generator")
		if len(serviceGenerator) == 0 {
			return cmdutil.UsageError(cmd, fmt.Sprintf("No service generator specified"))
		}
		if err := generateService(f, cmd, args, serviceGenerator, params, namespace, cmdOut); err != nil {
			return err
		}
	}

	attachFlag := cmd.Flags().Lookup("attach")
	attach := cmdutil.GetFlagBool(cmd, "attach")

	if !attachFlag.Changed && interactive {
		attach = true
	}

	remove := cmdutil.GetFlagBool(cmd, "rm")
	if !attach && remove {
		return cmdutil.UsageError(cmd, "--rm should only be used for attached containers")
	}

	if attach {
		quiet := cmdutil.GetFlagBool(cmd, "quiet")
		opts := &AttachOptions{
			StreamOptions: StreamOptions{
				In:    cmdIn,
				Out:   cmdOut,
				Err:   cmdErr,
				Stdin: interactive,
				TTY:   tty,
				Quiet: quiet,
			},

			CommandName: cmd.Parent().CommandPath() + " attach",

			Attach: &DefaultRemoteAttach{},
		}
		config, err := f.ClientConfig()
		if err != nil {
			return err
		}
		opts.Config = config

		client, err := f.Client()
		if err != nil {
			return err
		}
		opts.Client = client

		attachablePod, err := f.AttachablePodForObject(obj)
		if err != nil {
			return err
		}
		err = handleAttachPod(f, client, attachablePod.Namespace, attachablePod.Name, opts, quiet)
		if err != nil {
			return err
		}

		var pod *api.Pod
		leaveStdinOpen := cmdutil.GetFlagBool(cmd, "leave-stdin-open")
		waitForExitCode := !leaveStdinOpen && restartPolicy == api.RestartPolicyNever
		if waitForExitCode {
			pod, err = waitForPodTerminated(client, attachablePod.Namespace, attachablePod.Name, opts.Out, quiet)
			if err != nil {
				return err
			}
		}

		if remove {
			namespace, err = mapping.MetadataAccessor.Namespace(obj)
			if err != nil {
				return err
			}
			var name string
			name, err = mapping.MetadataAccessor.Name(obj)
			if err != nil {
				return err
			}
			_, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
			r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
				ContinueOnError().
				NamespaceParam(namespace).DefaultNamespace().
				ResourceNames(mapping.Resource, name).
				Flatten().
				Do()
			err = ReapResult(r, f, cmdOut, true, true, 0, -1, false, mapper, quiet)
			if err != nil {
				return err
			}
		}

		// after removal is done, return successfully if we are not interested in the exit code
		if !waitForExitCode {
			return nil
		}

		switch pod.Status.Phase {
		case api.PodSucceeded:
			return nil
		case api.PodFailed:
			unknownRcErr := fmt.Errorf("pod %s/%s failed with unknown exit code", pod.Namespace, pod.Name)
			if len(pod.Status.ContainerStatuses) == 0 || pod.Status.ContainerStatuses[0].State.Terminated == nil {
				return unknownRcErr
			}
			// assume here that we have at most one status because kubectl-run only creates one container per pod
			rc := pod.Status.ContainerStatuses[0].State.Terminated.ExitCode
			if rc == 0 {
				return unknownRcErr
			}
			return uexec.CodeExitError{
				Err:  fmt.Errorf("pod %s/%s terminated", pod.Namespace, pod.Name),
				Code: int(rc),
			}
		default:
			return fmt.Errorf("pod %s/%s left in phase %s", pod.Namespace, pod.Name, pod.Status.Phase)
		}
	}

	outputFormat := cmdutil.GetFlagString(cmd, "output")
	if outputFormat != "" || cmdutil.GetDryRunFlag(cmd) {
		return f.PrintObject(cmd, mapper, obj, cmdOut)
	}
	cmdutil.PrintSuccess(mapper, false, cmdOut, mapping.Resource, args[0], "created")
	return nil
}

// TODO turn this into reusable method checking available resources
func contains(resourcesList map[string]*unversioned.APIResourceList, resource unversioned.GroupVersionResource) bool {
	if resourcesList == nil {
		return false
	}
	resourcesGroup, ok := resourcesList[resource.GroupVersion().String()]
	if !ok {
		return false
	}
	for _, item := range resourcesGroup.APIResources {
		if resource.Resource == item.Name {
			return true
		}
	}
	return false
}

// waitForPod watches the given pod until the exitCondition is true. Each two seconds
// the tick function is called e.g. for progress output.
func waitForPod(c *client.Client, ns, name string, exitCondition func(*api.Pod) bool, tick func(*api.Pod)) (*api.Pod, error) {
	pod, err := c.Pods(ns).Get(name)
	if err != nil {
		return nil, err
	}
	if exitCondition(pod) {
		return pod, nil
	}

	tick(pod)

	w, err := c.Pods(ns).Watch(api.SingleObject(api.ObjectMeta{Name: pod.Name, ResourceVersion: pod.ResourceVersion}))
	if err != nil {
		return nil, err
	}

	t := time.NewTicker(2 * time.Second)
	defer t.Stop()
	go func() {
		for range t.C {
			tick(pod)
		}
	}()

	err = nil
	result := pod
	kubectl.WatchLoop(w, func(ev watch.Event) error {
		switch ev.Type {
		case watch.Added, watch.Modified:
			pod = ev.Object.(*api.Pod)
			if exitCondition(pod) {
				result = pod
				w.Stop()
			}
		case watch.Deleted:
			w.Stop()
		case watch.Error:
			result = nil
			err = fmt.Errorf("failed to watch pod %s/%s", ns, name)
			w.Stop()
		}
		return nil
	})

	return result, err
}

func waitForPodRunning(c *client.Client, ns, name string, out io.Writer, quiet bool) (*api.Pod, error) {
	exitCondition := func(pod *api.Pod) bool {
		switch pod.Status.Phase {
		case api.PodRunning:
			for _, status := range pod.Status.ContainerStatuses {
				if !status.Ready {
					return false
				}
			}
			return true
		case api.PodSucceeded, api.PodFailed:
			return true
		default:
			return false
		}
	}
	return waitForPod(c, ns, name, exitCondition, func(pod *api.Pod) {
		if !quiet {
			fmt.Fprintf(out, "Waiting for pod %s/%s to be running, status is %s, pod ready: false\n", pod.Namespace, pod.Name, pod.Status.Phase)
		}
	})
}

func waitForPodTerminated(c *client.Client, ns, name string, out io.Writer, quiet bool) (*api.Pod, error) {
	exitCondition := func(pod *api.Pod) bool {
		return pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed
	}
	return waitForPod(c, ns, name, exitCondition, func(pod *api.Pod) {
		if !quiet {
			fmt.Fprintf(out, "Waiting for pod %s/%s to terminate, status is %s\n", pod.Namespace, pod.Name, pod.Status.Phase)
		}
	})
}

func handleAttachPod(f *cmdutil.Factory, c *client.Client, ns, name string, opts *AttachOptions, quiet bool) error {
	pod, err := waitForPodRunning(c, ns, name, opts.Out, quiet)
	if err != nil {
		return err
	}
	ctrName, err := opts.GetContainerName(pod)
	if err != nil {
		return err
	}
	if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
		req, err := f.LogsForObject(pod, &api.PodLogOptions{Container: ctrName})
		if err != nil {
			return err
		}
		readCloser, err := req.Stream()
		if err != nil {
			return err
		}
		defer readCloser.Close()
		_, err = io.Copy(opts.Out, readCloser)
		return err
	}
	opts.Client = c
	opts.PodName = name
	opts.Namespace = ns
	// TODO: opts.Run sets opts.Err to nil, we need to find a better way
	stderr := opts.Err
	if err := opts.Run(); err != nil {
		fmt.Fprintf(stderr, "Error attaching, falling back to logs: %v\n", err)
		req, err := f.LogsForObject(pod, &api.PodLogOptions{Container: ctrName})
		if err != nil {
			return err
		}
		readCloser, err := req.Stream()
		if err != nil {
			return err
		}
		defer readCloser.Close()
		_, err = io.Copy(opts.Out, readCloser)
		return err
	}
	return nil
}

func getRestartPolicy(cmd *cobra.Command, interactive bool) (api.RestartPolicy, error) {
	restart := cmdutil.GetFlagString(cmd, "restart")
	if len(restart) == 0 {
		if interactive {
			return api.RestartPolicyOnFailure, nil
		} else {
			return api.RestartPolicyAlways, nil
		}
	}
	switch api.RestartPolicy(restart) {
	case api.RestartPolicyAlways:
		return api.RestartPolicyAlways, nil
	case api.RestartPolicyOnFailure:
		return api.RestartPolicyOnFailure, nil
	case api.RestartPolicyNever:
		return api.RestartPolicyNever, nil
	default:
		return "", cmdutil.UsageError(cmd, fmt.Sprintf("invalid restart policy: %s", restart))
	}
}

func verifyImagePullPolicy(cmd *cobra.Command) error {
	pullPolicy := cmdutil.GetFlagString(cmd, "image-pull-policy")
	switch api.PullPolicy(pullPolicy) {
	case api.PullAlways, api.PullIfNotPresent, api.PullNever:
		return nil
	case "":
		return nil
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("invalid image pull policy: %s", pullPolicy))
	}
}

func generateService(f *cmdutil.Factory, cmd *cobra.Command, args []string, serviceGenerator string, paramsIn map[string]interface{}, namespace string, out io.Writer) error {
	generators := f.Generators("expose")
	generator, found := generators[serviceGenerator]
	if !found {
		return fmt.Errorf("missing service generator: %s", serviceGenerator)
	}
	names := generator.ParamNames()

	port := cmdutil.GetFlagInt(cmd, "port")
	if port < 1 {
		return fmt.Errorf("--port must be a positive integer when exposing a service")
	}

	params := map[string]interface{}{}
	for key, value := range paramsIn {
		_, isString := value.(string)
		if isString {
			params[key] = value
		}
	}

	name, found := params["name"]
	if !found || len(name.(string)) == 0 {
		return fmt.Errorf("name is a required parameter")
	}
	selector, found := params["labels"]
	if !found || len(selector.(string)) == 0 {
		selector = fmt.Sprintf("run=%s", name.(string))
	}
	params["selector"] = selector

	if defaultName, found := params["default-name"]; !found || len(defaultName.(string)) == 0 {
		params["default-name"] = name
	}

	obj, _, mapper, mapping, err := createGeneratedObject(f, cmd, generator, names, params, cmdutil.GetFlagString(cmd, "service-overrides"), namespace)
	if err != nil {
		return err
	}

	if cmdutil.GetFlagString(cmd, "output") != "" || cmdutil.GetDryRunFlag(cmd) {
		return f.PrintObject(cmd, mapper, obj, out)
	}
	cmdutil.PrintSuccess(mapper, false, out, mapping.Resource, args[0], "created")

	return nil
}

func createGeneratedObject(f *cmdutil.Factory, cmd *cobra.Command, generator kubectl.Generator, names []kubectl.GeneratorParam, params map[string]interface{}, overrides, namespace string) (runtime.Object, string, meta.RESTMapper, *meta.RESTMapping, error) {
	err := kubectl.ValidateParams(names, params)
	if err != nil {
		return nil, "", nil, nil, err
	}

	// TODO: Validate flag usage against selected generator. More tricky since --expose was added.
	obj, err := generator.Generate(params)
	if err != nil {
		return nil, "", nil, nil, err
	}

	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	groupVersionKinds, _, err := typer.ObjectKinds(obj)
	if err != nil {
		return nil, "", nil, nil, err
	}
	groupVersionKind := groupVersionKinds[0]

	if len(overrides) > 0 {
		codec := runtime.NewCodec(f.JSONEncoder(), f.Decoder(true))
		obj, err = cmdutil.Merge(codec, obj, overrides, groupVersionKind.Kind)
		if err != nil {
			return nil, "", nil, nil, err
		}
	}

	mapping, err := mapper.RESTMapping(groupVersionKind.GroupKind(), groupVersionKind.Version)
	if err != nil {
		return nil, "", nil, nil, err
	}
	client, err := f.ClientForMapping(mapping)
	if err != nil {
		return nil, "", nil, nil, err
	}

	annotations, err := mapping.MetadataAccessor.Annotations(obj)
	if err != nil {
		return nil, "", nil, nil, err
	}
	if cmdutil.GetRecordFlag(cmd) || len(annotations[kubectl.ChangeCauseAnnotation]) > 0 {
		if err := cmdutil.RecordChangeCause(obj, f.Command()); err != nil {
			return nil, "", nil, nil, err
		}
	}
	if !cmdutil.GetDryRunFlag(cmd) {
		resourceMapper := &resource.Mapper{
			ObjectTyper:  typer,
			RESTMapper:   mapper,
			ClientMapper: resource.ClientMapperFunc(f.ClientForMapping),
			Decoder:      f.Decoder(true),
		}
		info, err := resourceMapper.InfoForObject(obj, nil)
		if err != nil {
			return nil, "", nil, nil, err
		}

		if err := kubectl.CreateOrUpdateAnnotation(cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag), info, f.JSONEncoder()); err != nil {
			return nil, "", nil, nil, err
		}

		obj, err = resource.NewHelper(client, mapping).Create(namespace, false, info.Object)
		if err != nil {
			return nil, "", nil, nil, err
		}
	}
	return obj, groupVersionKind.Kind, mapper, mapping, err
}
