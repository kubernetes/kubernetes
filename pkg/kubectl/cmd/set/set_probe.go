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

package set

import (
	"fmt"
	"io"
	"net"
	"net/url"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

var (
	probeResources = `
	pod (po), replicationcontroller (rc), deployment (deploy), daemonset (ds), replicaset (rs), job (Job), cronjob (Cronjob)`

	probeLong = templates.LongDesc(`
		Set or remove a liveness or readiness probe from a pod or pod template

		Each container in a pod may define one or more probes that are used for general health
		checking. A liveness probe is checked periodically to ensure the container is still healthy:
		if the probe fails, the container is restarted. Readiness probes set or clear the ready
		flag for each container, which controls whether the container's ports are included in the list
		of endpoints for a service and whether a deployment can proceed. A readiness check should
		indicate when your container is ready to accept incoming traffic or begin handling work.
		Setting both liveness and readiness probes for each container is highly recommended.

		The three probe types are:

		1. Open a TCP socket on the pod IP
		2. Perform an HTTP GET against a URL on a container that must return 200 OK
		3. Run a command in the container that must return exit code 0

		Containers that take a variable amount of time to start should set generous
		initial-delay-seconds values, otherwise as your application evolves you may suddenly begin
		to fail.
		Possible resources include (case insensitive):
			` + probeResources)

	probeExample = templates.Examples(`
	  # Clear both readiness and liveness probes off all containers
	  kubectl set probe deploy/registry --remove --readiness --liveness
	  
	  # Set an exec action as a liveness probe to run 'echo ok'
	  kubectl set probe deploy/registry --liveness -- echo ok

	  # Set probe of a local file without talking to the server
	  kubectl set probe -f ./deploy.yaml -c=perl --liveness --local -o yaml -- echo ok

	  # Set a readiness probe to try to open a TCP socket on 3306
	  kubectl set probe deploy/mysql --readiness --open-tcp=3306

	  # Set an HTTP readiness probe for port 8080 and path /healthz over HTTP on the pod IP
	  kubectl set probe deploy/webapp --readiness --get-url=http://:8080/healthz

	  # Set an HTTP readiness probe over HTTPS on 127.0.0.1 for a hostNetwork pod
	  kubectl set probe deploy/router --readiness --get-url=https://127.0.0.1:1936/stats

	  # Set only the initial-delay-seconds field on all deployments
	  kubectl set probe deploy --all --readiness --initial-delay-seconds=30`)
)

type ProbeOptions struct {
	Out io.Writer
	Err io.Writer
	resource.FilenameOptions
	ContainerSelector      string
	Selector               string
	All                    bool
	Output                 string
	Builder                *resource.Builder
	Infos                  []*resource.Info
	Encoder                runtime.Encoder
	ShortOutput            bool
	DryRun                 bool
	Record                 bool
	Cmd                    *cobra.Command
	ChangeCause            string
	Mapper                 meta.RESTMapper
	PrintSuccess           func(mapper meta.RESTMapper, shortOutput bool, out io.Writer, resource, name string, dryRun bool, operation string)
	PrintObject            func(cmd *cobra.Command, isLocal bool, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
	UpdatePodSpecForObject func(obj runtime.Object, fn func(*v1.PodSpec) error) (bool, error)
	Resources              []string

	Readiness bool
	Liveness  bool
	Remove    bool
	Local     bool

	OpenTCPSocket string
	HTTPGet       string
	Command       []string

	FlagSet       func(string) bool
	HTTPGetAction *v1.HTTPGetAction

	// Length of time before health checking is activated.  In seconds.
	InitialDelaySeconds *int
	// Length of time before health checking times out.  In seconds.
	TimeoutSeconds *int
	// How often (in seconds) to perform the probe.
	PeriodSeconds *int
	// Minimum consecutive successes for the probe to be considered successful after having failed.
	// Must be 1 for liveness.
	SuccessThreshold *int
	// Minimum consecutive failures for the probe to be considered failed after having succeeded.
	FailureThreshold *int
}

// NewCmdProbe implements the set probe command
func NewCmdProbe(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	options := &ProbeOptions{
		Out: out,
		Err: errOut,
	}
	cmd := &cobra.Command{
		Use:     "probe RESOURCE/NAME --readiness|--liveness [options] (--get-url=URL|--open-tcp=PORT|-- CMD)",
		Short:   "Update a probe on a pod template",
		Long:    probeLong,
		Example: probeExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run(f))
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().StringVarP(&options.ContainerSelector, "containers", "c", "*", "The names of containers in the selected pod templates to change - may use wildcards")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", options.Selector, "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&options.All, "all", options.All, "If true, select all resources in the namespace of the specified resource types")

	cmd.Flags().BoolVar(&options.Remove, "remove", options.Remove, "If true, remove the specified probe(s).")
	cmd.Flags().BoolVar(&options.Readiness, "readiness", options.Readiness, "Set or remove a readiness probe to indicate when this container should receive traffic")
	cmd.Flags().BoolVar(&options.Liveness, "liveness", options.Liveness, "Set or remove a liveness probe to verify this container is running")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set probe will NOT contact api-server but run locally.")

	cmd.Flags().StringVar(&options.OpenTCPSocket, "open-tcp", options.OpenTCPSocket, "A port number or port name to attempt to open via TCP.")
	cmd.Flags().StringVar(&options.HTTPGet, "get-url", options.HTTPGet, "A URL to perform an HTTP GET on (you can omit the host, have a string port, or omit the scheme.")
	options.InitialDelaySeconds = cmd.Flags().Int("initial-delay-seconds", 0, "The time in seconds to wait before the probe begins checking")
	options.SuccessThreshold = cmd.Flags().Int("success-threshold", 0, "The number of successes required before the probe is considered successful")
	options.FailureThreshold = cmd.Flags().Int("failure-threshold", 0, "The number of failures before the probe is considered to have failed")
	options.PeriodSeconds = cmd.Flags().Int("period-seconds", 0, "The time in seconds between attempts")
	options.TimeoutSeconds = cmd.Flags().Int("timeout-seconds", 0, "The time in seconds to wait before considering the probe to have failed")

	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddRecordFlag(cmd)
	cmdutil.AddIncludeUninitializedFlag(cmd)
	cmd.MarkFlagFilename("filename", "yaml", "yml", "json")

	return cmd
}

func (o *ProbeOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.Resources, o.Command = getResourcesAndCommand(cmd, args)
	o.Mapper, _ = f.Object()
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Record = cmdutil.GetRecordFlag(cmd)
	o.ChangeCause = f.Command(cmd, false)
	o.Encoder = f.JSONEncoder()
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.PrintObject = f.PrintObject
	o.PrintSuccess = f.PrintSuccess
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.Cmd = cmd

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	includeUninitialized := cmdutil.ShouldIncludeUninitialized(cmd, false)
	builder := f.NewBuilder().
		Internal().
		LocalParam(o.Local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		IncludeUninitialized(includeUninitialized).
		Flatten()

	if !o.Local {
		builder = builder.
			LabelSelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, o.Resources...).
			Latest()
	}

	o.Infos, err = builder.Do().Infos()
	if err != nil {
		return err
	}

	if !cmd.Flags().Lookup("initial-delay-seconds").Changed {
		o.InitialDelaySeconds = nil
	}
	if !cmd.Flags().Lookup("timeout-seconds").Changed {
		o.TimeoutSeconds = nil
	}
	if !cmd.Flags().Lookup("period-seconds").Changed {
		o.PeriodSeconds = nil
	}
	if !cmd.Flags().Lookup("success-threshold").Changed {
		o.SuccessThreshold = nil
	}
	if !cmd.Flags().Lookup("failure-threshold").Changed {
		o.FailureThreshold = nil
	}

	if len(o.HTTPGet) > 0 {
		url, err := url.Parse(o.HTTPGet)
		if err != nil {
			return fmt.Errorf("--get-url could not be parsed as a valid URL: %v", err)
		}
		var host, port string
		if strings.Contains(url.Host, ":") {
			if host, port, err = net.SplitHostPort(url.Host); err != nil {
				return fmt.Errorf("--get-url did not have a valid port specification: %v", err)
			}
		}
		if host == "localhost" {
			host = ""
		}
		o.HTTPGetAction = &v1.HTTPGetAction{
			Scheme: v1.URIScheme(strings.ToUpper(url.Scheme)),
			Host:   host,
			Port:   intOrString(port),
			Path:   url.Path,
		}
	}

	return nil
}

func (o *ProbeOptions) Validate() error {
	errors := []error{}
	if len(o.Resources) < 1 && cmdutil.IsFilenameSliceEmpty(o.Filenames) {
		errors = append(errors, fmt.Errorf("one or more resources must be specified as <resource> <name> or <resource>/<name>"))
	}
	if !o.Readiness && !o.Liveness {
		errors = append(errors, fmt.Errorf("you must specify one of --readiness or --liveness or both"))
	}
	count := 0
	if o.Command != nil {
		count++
	}
	if len(o.OpenTCPSocket) > 0 {
		count++
	}
	if len(o.HTTPGet) > 0 {
		count++
	}

	switch {
	case o.Remove && count != 0:
		errors = append(errors, fmt.Errorf("--remove may not be used with any flag except --readiness or --liveness"))
	case count > 1:
		errors = append(errors, fmt.Errorf("you may only set one of --get-url, --open-tcp, or command"))
	case len(o.OpenTCPSocket) > 0 && intOrString(o.OpenTCPSocket).IntVal > 65535:
		errors = append(errors, fmt.Errorf("--open-tcp must be a port number between 1 and 65535 or an IANA port name"))
	}
	if o.FailureThreshold != nil && *o.FailureThreshold < 1 {
		errors = append(errors, fmt.Errorf("--failure-threshold may not be less than one"))
	}
	if o.SuccessThreshold != nil && *o.SuccessThreshold < 1 {
		errors = append(errors, fmt.Errorf("--success-threshold may not be less than one"))
	}
	if o.InitialDelaySeconds != nil && *o.InitialDelaySeconds < 0 {
		errors = append(errors, fmt.Errorf("--initial-delay-seconds may not be negative"))
	}
	if o.TimeoutSeconds != nil && *o.TimeoutSeconds < 0 {
		errors = append(errors, fmt.Errorf("--timeout-seconds may not be negative"))
	}
	if o.PeriodSeconds != nil && *o.PeriodSeconds < 0 {
		errors = append(errors, fmt.Errorf("--period-seconds may not be negative"))
	}
	return utilerrors.NewAggregate(errors)
}

func (o *ProbeOptions) Run(f cmdutil.Factory) error {
	allErrs := []error{}
	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		transformed := false
		info.Object = info.AsVersioned()
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *v1.PodSpec) error {
			containers, _ := selectContainers(spec.Containers, o.ContainerSelector)
			if len(containers) == 0 {
				if _, err := fmt.Fprintf(o.Err, "warning: %s/%s does not have any containers matching %q\n", info.Mapping.Resource, info.Name, o.ContainerSelector); err != nil {
					return err
				}
				return nil
			}
			// perform updates
			transformed = true
			for _, container := range containers {
				o.updateContainer(container)
			}
			return nil
		})
		if transformed && err == nil {
			return runtime.Encode(o.Encoder, info.Object)
		}
		return nil, err
	})

	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			fmt.Fprintf(o.Err, "error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err)
			continue
		}

		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			if _, err := fmt.Fprintf(o.Out, "%s %q was not changed\n", info.Mapping.Resource, info.Name); err != nil {
				return err
			}
			continue
		}
		if o.PrintObject != nil && (o.Local || o.DryRun) {
			if err := o.PrintObject(o.Cmd, o.Local, o.Mapper, patch.Info.AsVersioned(), o.Out); err != nil {
				return err
			}
			continue
		}
		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch probe update to pod template: %v\n", err))

			// if no port was specified, inform that one must be provided
			if len(o.HTTPGet) > 0 && len(o.HTTPGetAction.Port.String()) == 0 {
				fmt.Fprintf(o.Err, "\nA port must be specified as part of a url (http://127.0.0.1:3306).\nSee 'oc set probe -h' for help and examples.\n")
			}
			continue
		}

		// record this change (for rollout history)
		if o.Record || cmdutil.ContainsChangeCause(info) {
			if patch, patchType, err := cmdutil.ChangeResourcePatch(info, o.ChangeCause); err == nil {
				if obj, err = resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, patchType, patch); err != nil {
					fmt.Fprintf(o.Err, "WARNING: changes to %s/%s can't be recorded: %v\n", info.Mapping.Resource, info.Name, err)
				}
			}
		}

		info.Refresh(obj, true)

		if len(o.Output) > 0 && o.PrintObject != nil {
			if err := o.PrintObject(o.Cmd, o.Local, o.Mapper, patch.Info.AsVersioned(), o.Out); err != nil {
				return err
			}
			continue
		}
		o.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, o.DryRun, "image updated")
	}

	return utilerrors.NewAggregate(allErrs)
}

func (o *ProbeOptions) updateContainer(container *v1.Container) {
	if o.Remove {
		if o.Readiness {
			container.ReadinessProbe = nil
		}
		if o.Liveness {
			container.LivenessProbe = nil
		}
		return
	}
	if o.Readiness {
		if container.ReadinessProbe == nil {
			container.ReadinessProbe = &v1.Probe{}
		}
		o.updateProbe(container.ReadinessProbe)
	}
	if o.Liveness {
		if container.LivenessProbe == nil {
			container.LivenessProbe = &v1.Probe{}
		}
		o.updateProbe(container.LivenessProbe)
	}
}

// updateProbe updates only those fields with flags set by the user
func (o *ProbeOptions) updateProbe(probe *v1.Probe) {
	switch {
	case o.Command != nil:
		probe.Handler = v1.Handler{Exec: &v1.ExecAction{Command: o.Command}}
	case o.HTTPGetAction != nil:
		probe.Handler = v1.Handler{HTTPGet: o.HTTPGetAction}
	case len(o.OpenTCPSocket) > 0:
		probe.Handler = v1.Handler{TCPSocket: &v1.TCPSocketAction{Port: intOrString(o.OpenTCPSocket)}}
	}
	if o.InitialDelaySeconds != nil {
		probe.InitialDelaySeconds = int32(*o.InitialDelaySeconds)
	}
	if o.SuccessThreshold != nil {
		probe.SuccessThreshold = int32(*o.SuccessThreshold)
	}
	if o.FailureThreshold != nil {
		probe.FailureThreshold = int32(*o.FailureThreshold)
	}
	if o.TimeoutSeconds != nil {
		probe.TimeoutSeconds = int32(*o.TimeoutSeconds)
	}
	if o.PeriodSeconds != nil {
		probe.PeriodSeconds = int32(*o.PeriodSeconds)
	}
}

func intOrString(s string) intstr.IntOrString {
	if i, err := strconv.Atoi(s); err == nil {
		return intstr.FromInt(i)
	}
	return intstr.FromString(s)
}

func getResourcesAndCommand(cmd *cobra.Command, args []string) (resources []string, commands []string) {
	resources = args
	if i := cmd.ArgsLenAtDash(); i != -1 {
		resources = args[:i]
		commands := args[i:]
		return resources, commands
	}
	return resources, nil
}
