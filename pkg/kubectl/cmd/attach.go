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
	"net/url"

	"github.com/golang/glog"
	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	remotecommandserver "k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/term"
)

var (
	attach_example = dedent.Dedent(`
		# Get output from running pod 123456-7890, using the first container by default
		kubectl attach 123456-7890

		# Get output from ruby-container from pod 123456-7890
		kubectl attach 123456-7890 -c ruby-container

		# Switch to raw terminal mode, sends stdin to 'bash' in ruby-container from pod 123456-7890
		# and sends stdout/stderr from 'bash' back to the client
		kubectl attach 123456-7890 -c ruby-container -i -t`)
)

func NewCmdAttach(f *cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer) *cobra.Command {
	options := &AttachOptions{
		StreamOptions: StreamOptions{
			In:  cmdIn,
			Out: cmdOut,
			Err: cmdErr,
		},

		Attach: &DefaultRemoteAttach{},
	}
	cmd := &cobra.Command{
		Use:     "attach POD -c CONTAINER",
		Short:   "Attach to a running container",
		Long:    "Attach to a process that is already running inside an existing container.",
		Example: attach_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}
	// TODO support UID
	cmd.Flags().StringVarP(&options.ContainerName, "container", "c", "", "Container name. If omitted, the first container in the pod will be chosen")
	cmd.Flags().BoolVarP(&options.Stdin, "stdin", "i", false, "Pass stdin to the container")
	cmd.Flags().BoolVarP(&options.TTY, "tty", "t", false, "Stdin is a TTY")
	return cmd
}

// RemoteAttach defines the interface accepted by the Attach command - provided for test stubbing
type RemoteAttach interface {
	Attach(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue term.TerminalSizeQueue) error
}

// DefaultRemoteAttach is the standard implementation of attaching
type DefaultRemoteAttach struct{}

func (*DefaultRemoteAttach) Attach(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue term.TerminalSizeQueue) error {
	exec, err := remotecommand.NewExecutor(config, method, url)
	if err != nil {
		return err
	}
	return exec.Stream(remotecommand.StreamOptions{
		SupportedProtocols: remotecommandserver.SupportedStreamingProtocols,
		Stdin:              stdin,
		Stdout:             stdout,
		Stderr:             stderr,
		Tty:                tty,
		TerminalSizeQueue:  terminalSizeQueue,
	})
}

// AttachOptions declare the arguments accepted by the Exec command
type AttachOptions struct {
	StreamOptions

	CommandName string

	Pod *api.Pod

	Attach RemoteAttach
	Client *client.Client
	Config *restclient.Config
}

// Complete verifies command line arguments and loads data from the command environment
func (p *AttachOptions) Complete(f *cmdutil.Factory, cmd *cobra.Command, argsIn []string) error {
	if len(argsIn) == 0 {
		return cmdutil.UsageError(cmd, "POD is required for attach")
	}
	if len(argsIn) > 1 {
		return cmdutil.UsageError(cmd, fmt.Sprintf("expected a single argument: POD, saw %d: %s", len(argsIn), argsIn))
	}
	p.PodName = argsIn[0]

	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	p.Namespace = namespace

	config, err := f.ClientConfig()
	if err != nil {
		return err
	}
	p.Config = config

	client, err := f.Client()
	if err != nil {
		return err
	}
	p.Client = client

	if p.CommandName == "" {
		p.CommandName = cmd.CommandPath()
	}

	return nil
}

// Validate checks that the provided attach options are specified.
func (p *AttachOptions) Validate() error {
	allErrs := []error{}
	if len(p.PodName) == 0 {
		allErrs = append(allErrs, fmt.Errorf("pod name must be specified"))
	}
	if p.Out == nil || p.Err == nil {
		allErrs = append(allErrs, fmt.Errorf("both output and error output must be provided"))
	}
	if p.Attach == nil || p.Client == nil || p.Config == nil {
		allErrs = append(allErrs, fmt.Errorf("client, client config, and attach must be provided"))
	}
	return utilerrors.NewAggregate(allErrs)
}

// Run executes a validated remote execution against a pod.
func (p *AttachOptions) Run() error {
	if p.Pod == nil {
		pod, err := p.Client.Pods(p.Namespace).Get(p.PodName)
		if err != nil {
			return err
		}

		if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
			return fmt.Errorf("cannot attach a container in a completed pod; current phase is %s", pod.Status.Phase)
		}

		p.Pod = pod
		// TODO: convert this to a clean "wait" behavior
	}
	pod := p.Pod

	// check for TTY
	containerToAttach, err := p.containerToAttachTo(pod)
	if err != nil {
		return fmt.Errorf("cannot attach to the container: %v", err)
	}
	if p.TTY && !containerToAttach.TTY {
		p.TTY = false
		if p.Err != nil {
			fmt.Fprintf(p.Err, "Unable to use a TTY - container %s did not allocate one\n", containerToAttach.Name)
		}
	} else if !p.TTY && containerToAttach.TTY {
		// the container was launched with a TTY, so we have to force a TTY here, otherwise you'll get
		// an error "Unrecognized input header"
		p.TTY = true
	}

	// ensure we can recover the terminal while attached
	t := p.setupTTY()

	// save p.Err so we can print the command prompt message below
	stderr := p.Err

	var sizeQueue term.TerminalSizeQueue
	if t.Raw {
		if size := t.GetSize(); size != nil {
			// fake resizing +1 and then back to normal so that attach-detach-reattach will result in the
			// screen being redrawn
			sizePlusOne := *size
			sizePlusOne.Width++
			sizePlusOne.Height++

			// this call spawns a goroutine to monitor/update the terminal size
			sizeQueue = t.MonitorSize(&sizePlusOne, size)
		}

		// unset p.Err if it was previously set because both stdout and stderr go over p.Out when tty is
		// true
		p.Err = nil
	}

	fn := func() error {

		if !p.Quiet && stderr != nil {
			fmt.Fprintln(stderr, "If you don't see a command prompt, try pressing enter.")
		}

		// TODO: consider abstracting into a client invocation or client helper
		req := p.Client.RESTClient.Post().
			Resource("pods").
			Name(pod.Name).
			Namespace(pod.Namespace).
			SubResource("attach")
		req.VersionedParams(&api.PodAttachOptions{
			Container: containerToAttach.Name,
			Stdin:     p.Stdin,
			Stdout:    p.Out != nil,
			Stderr:    p.Err != nil,
			TTY:       t.Raw,
		}, api.ParameterCodec)

		return p.Attach.Attach("POST", req.URL(), p.Config, p.In, p.Out, p.Err, t.Raw, sizeQueue)
	}

	if err := t.Safe(fn); err != nil {
		return err
	}

	if p.Stdin && t.Raw && pod.Spec.RestartPolicy == api.RestartPolicyAlways {
		fmt.Fprintf(p.Out, "Session ended, resume using '%s %s -c %s -i -t' command when the pod is running\n", p.CommandName, pod.Name, containerToAttach.Name)
	}
	return nil
}

// containerToAttach returns a reference to the container to attach to, given
// by name or the first container if name is empty.
func (p *AttachOptions) containerToAttachTo(pod *api.Pod) (*api.Container, error) {
	if len(p.ContainerName) > 0 {
		for i := range pod.Spec.Containers {
			if pod.Spec.Containers[i].Name == p.ContainerName {
				return &pod.Spec.Containers[i], nil
			}
		}
		for i := range pod.Spec.InitContainers {
			if pod.Spec.InitContainers[i].Name == p.ContainerName {
				return &pod.Spec.InitContainers[i], nil
			}
		}
		return nil, fmt.Errorf("container not found (%s)", p.ContainerName)
	}

	glog.V(4).Infof("defaulting container name to %s", pod.Spec.Containers[0].Name)
	return &pod.Spec.Containers[0], nil
}

// GetContainerName returns the name of the container to attach to, with a fallback.
func (p *AttachOptions) GetContainerName(pod *api.Pod) (string, error) {
	c, err := p.containerToAttachTo(pod)
	if err != nil {
		return "", err
	}
	return c.Name, nil
}
