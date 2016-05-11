/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

const (
	probeLong = `
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
to fail.`

	probeExample = `  # Clear both readiness and liveness probes off all containers
  %[1]s probe dc/registry --remove --readiness --liveness

  # Set an exec action as a liveness probe to run 'echo ok'
  %[1]s probe dc/registry --liveness -- echo ok

  # Set a readiness probe to try to open a TCP socket on 3306
  %[1]s probe rc/mysql --readiness --open-tcp=3306

  # Set an HTTP readiness probe for port 8080 and path /healthz over HTTP on the pod IP
  %[1]s probe dc/webapp --readiness --get-url=http://:8080/healthz

  # Set an HTTP readiness probe over HTTPS on 127.0.0.1 for a hostNetwork pod
  %[1]s probe dc/router --readiness --get-url=https://127.0.0.1:1936/stats

  # Set only the initial-delay-seconds field on all deployments
  %[1]s probe dc --all --readiness --initial-delay-seconds=30`
)

type ProbeOptions struct {
	Out io.Writer
	Err io.Writer

	Filenames         []string
	ContainerSelector string
	Selector          string
	All               bool

	Builder *resource.Builder
	Infos   []*resource.Info

	Encoder runtime.Encoder

	ShortOutput bool
	Mapper      meta.RESTMapper

	PrintObject            func(runtime.Object) error
	UpdatePodSpecForObject func(runtime.Object, func(spec *api.PodSpec) error) (bool, error)

	Readiness bool
	Liveness  bool
	Remove    bool

	OpenTCPSocket string
	HTTPGet       string
	Command       []string

	FlagSet       func(string) bool
	HTTPGetAction *api.HTTPGetAction

	// Length of time before health checking is activated.  In seconds.
	InitialDelaySeconds *int32
	// Length of time before health checking times out.  In seconds.
	TimeoutSeconds *int32
	// How often (in seconds) to perform the probe.
	PeriodSeconds *int32
	// Minimum consecutive successes for the probe to be considered successful after having failed.
	// Must be 1 for liveness.
	SuccessThreshold *int32
	// Minimum consecutive failures for the probe to be considered failed after having succeeded.
	FailureThreshold *int32
}

// NewCmdProbe implements the set probe command
func NewCmdProbe(fullName string, f *cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	options := &ProbeOptions{
		Out: out,
		Err: errOut,

		ContainerSelector: "*",
	}
	cmd := &cobra.Command{
		Use:     "probe RESOURCE/NAME --readiness|--liveness (--get-url=URL|--open-tcp=PORT|-- CMD)",
		Short:   "Update a probe on a pod template",
		Long:    probeLong,
		Example: fmt.Sprintf(probeExample, fullName),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().StringVarP(&options.ContainerSelector, "containers", "c", options.ContainerSelector, "The names of containers in the selected pod templates to change - may use wildcards")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", options.Selector, "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&options.All, "all", options.All, "Select all resources in the namespace of the specified resource types")
	cmd.Flags().StringSliceVarP(&options.Filenames, "filename", "f", options.Filenames, "Filename, directory, or URL to file to use to edit the resource.")

	cmd.Flags().BoolVar(&options.Remove, "remove", options.Remove, "If true, remove the specified probe(s).")
	cmd.Flags().BoolVar(&options.Readiness, "readiness", options.Readiness, "Set or remove a readiness probe to indicate when this container should receive traffic")
	cmd.Flags().BoolVar(&options.Liveness, "liveness", options.Liveness, "Set or remove a liveness probe to verify this container is running")

	cmd.Flags().StringVar(&options.OpenTCPSocket, "open-tcp", options.OpenTCPSocket, "A port number or port name to attempt to open via TCP.")
	cmd.Flags().StringVar(&options.HTTPGet, "get-url", options.HTTPGet, "A URL to perform an HTTP GET on (you can omit the host, have a string port, or omit the scheme.")
	options.InitialDelaySeconds = cmd.Flags().Int32("initial-delay-seconds", 0, "The time in seconds to wait before the probe begins checking")
	options.SuccessThreshold = cmd.Flags().Int32("success-threshold", 0, "The number of successes required before the probe is considered successful")
	options.FailureThreshold = cmd.Flags().Int32("failure-threshold", 0, "The number of failures before the probe is considered to have failed")
	options.PeriodSeconds = cmd.Flags().Int32("period-seconds", 0, "The time in seconds between attempts")
	options.TimeoutSeconds = cmd.Flags().Int32("timeout-seconds", 0, "The time in seconds to wait before considering the probe to have failed")

	cmd.MarkFlagFilename("filename", "yaml", "yml", "json")

	return cmd
}

func (o *ProbeOptions) Complete(f *cmdutil.Factory, cmd *cobra.Command, args []string) error {
	resources := args
	if i := cmd.ArgsLenAtDash(); i != -1 {
		resources = args[:i]
		o.Command = args[i:]
	}
	if len(o.Filenames) == 0 && len(args) < 1 {
		return cmdutil.UsageError(cmd, "one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}

	cmdNamespace, explicit, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	mapper, typer := f.Object(false)
	o.Builder = resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(explicit, false, o.Filenames...).
		SelectorParam(o.Selector).
		ResourceTypeOrNameArgs(o.All, resources...).
		Flatten()

	output := cmdutil.GetFlagString(cmd, "output")
	if len(output) != 0 {
		o.PrintObject = func(obj runtime.Object) error { return f.PrintObject(cmd, mapper, obj, o.Out) }
	}

	o.Encoder = f.JSONEncoder()
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"
	o.Mapper = mapper

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
		o.HTTPGetAction = &api.HTTPGetAction{
			Scheme: api.URIScheme(strings.ToUpper(url.Scheme)),
			Host:   host,
			Port:   intOrString(port),
			Path:   url.Path,
		}
	}

	return nil
}

func (o *ProbeOptions) Validate() error {
	if !o.Readiness && !o.Liveness {
		return fmt.Errorf("you must specify one of --readiness or --liveness or both")
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
		return fmt.Errorf("--remove may not be used with any flag except --readiness or --liveness")
	case count > 1:
		return fmt.Errorf("you may only set one of --get-url, --open-tcp, or command")
	case len(o.OpenTCPSocket) > 0 && intOrString(o.OpenTCPSocket).IntVal > 65535:
		return fmt.Errorf("--open-tcp must be a port number between 1 and 65535 or an IANA port name")
	}
	if o.FailureThreshold != nil && *o.FailureThreshold < 1 {
		return fmt.Errorf("--failure-threshold may not be less than one")
	}
	if o.SuccessThreshold != nil && *o.SuccessThreshold < 1 {
		return fmt.Errorf("--success-threshold may not be less than one")
	}
	if o.InitialDelaySeconds != nil && *o.InitialDelaySeconds < 0 {
		return fmt.Errorf("--initial-delay-seconds may not be negative")
	}
	if o.TimeoutSeconds != nil && *o.TimeoutSeconds < 0 {
		return fmt.Errorf("--timeout-seconds may not be negative")
	}
	if o.PeriodSeconds != nil && *o.PeriodSeconds < 0 {
		return fmt.Errorf("--period-seconds may not be negative")
	}
	return nil
}

func (o *ProbeOptions) Run() error {
	infos := o.Infos
	singular := len(o.Infos) <= 1
	if o.Builder != nil {
		loaded, err := o.Builder.Do().IntoSingular(&singular).Infos()
		if err != nil {
			return err
		}
		infos = loaded
	}

	patches := CalculatePatches(infos, o.Encoder, func(info *resource.Info) (bool, error) {
		transformed := false
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *api.PodSpec) error {
			containers, _ := selectContainers(spec.Containers, o.ContainerSelector)
			if len(containers) == 0 {
				fmt.Fprintf(o.Err, "warning: %s/%s does not have any containers matching %q\n", info.Mapping.Resource, info.Name, o.ContainerSelector)
				return nil
			}
			// perform updates
			transformed = true
			for _, container := range containers {
				o.updateContainer(container)
			}
			return nil
		})
		return transformed, err
	})
	if singular && len(patches) == 0 {
		return fmt.Errorf("%s/%s is not a pod or does not have a pod template", infos[0].Mapping.Resource, infos[0].Name)
	}

	if o.PrintObject != nil {
		var infos []*resource.Info
		for _, patch := range patches {
			info := patch.Info
			if patch.Err != nil {
				fmt.Fprintf(o.Err, "error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err)
				continue
			}
			infos = append(infos, info)
		}
		object, err := resource.AsVersionedObject(infos, !singular, "", nil)
		if err != nil {
			return err
		}
		return o.PrintObject(object)
	}

	failed := false
	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			fmt.Fprintf(o.Err, "error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err)
			continue
		}

		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			fmt.Fprintf(o.Err, "info: %s %q was not changed\n", info.Mapping.Resource, info.Name)
			continue
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, api.StrategicMergePatchType, patch.Patch)
		if err != nil {
			handlePodUpdateError(o.Err, err, "probes")
			failed = true
			continue
		}

		info.Refresh(obj, true)
		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, "updated")
	}
	if failed {
		return cmdutil.ErrExit
	}
	return nil
}

func (o *ProbeOptions) updateContainer(container *api.Container) {
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
			container.ReadinessProbe = &api.Probe{}
		}
		o.updateProbe(container.ReadinessProbe)
	}
	if o.Liveness {
		if container.LivenessProbe == nil {
			container.LivenessProbe = &api.Probe{}
		}
		o.updateProbe(container.LivenessProbe)
	}
}

// updateProbe updates only those fields with flags set by the user
func (o *ProbeOptions) updateProbe(probe *api.Probe) {
	switch {
	case o.Command != nil:
		probe.Handler = api.Handler{Exec: &api.ExecAction{Command: o.Command}}
	case o.HTTPGetAction != nil:
		probe.Handler = api.Handler{HTTPGet: o.HTTPGetAction}
	case len(o.OpenTCPSocket) > 0:
		probe.Handler = api.Handler{TCPSocket: &api.TCPSocketAction{Port: intOrString(o.OpenTCPSocket)}}
	}
	if o.InitialDelaySeconds != nil {
		probe.InitialDelaySeconds = *o.InitialDelaySeconds
	}
	if o.SuccessThreshold != nil {
		probe.SuccessThreshold = *o.SuccessThreshold
	}
	if o.FailureThreshold != nil {
		probe.FailureThreshold = *o.FailureThreshold
	}
	if o.TimeoutSeconds != nil {
		probe.TimeoutSeconds = *o.TimeoutSeconds
	}
	if o.PeriodSeconds != nil {
		probe.PeriodSeconds = *o.PeriodSeconds
	}
}

func intOrString(s string) intstr.IntOrString {
	if i, err := strconv.Atoi(s); err == nil {
		return intstr.FromInt(i)
	}
	return intstr.FromString(s)
}
