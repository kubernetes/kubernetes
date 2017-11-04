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
	"errors"
	"fmt"
	"io"
	"net/url"
	"time"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	attachExample = templates.Examples(i18n.T(`
		# Get output from running pod 123456-7890, using the first container by default
		kubectl attach 123456-7890

		# Get output from ruby-container from pod 123456-7890
		kubectl attach 123456-7890 -c ruby-container

		# Switch to raw terminal mode, sends stdin to 'bash' in ruby-container from pod 123456-7890
		# and sends stdout/stderr from 'bash' back to the client
		kubectl attach 123456-7890 -c ruby-container -i -t

		# Get output from the first pod of a ReplicaSet named nginx
		kubectl attach rs/nginx
		`))
)

const (
	defaultPodAttachTimeout = 60 * time.Second
	defaultPodLogsTimeout   = 20 * time.Second
)

func NewCmdAttach(f cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer) *cobra.Command {
	options := &AttachOptions{
		StreamOptions: StreamOptions{
			In:  cmdIn,
			Out: cmdOut,
			Err: cmdErr,
		},

		Attach: &DefaultRemoteAttach{},
	}
	cmd := &cobra.Command{
		Use:     "attach (POD | TYPE/NAME) -c CONTAINER",
		Short:   i18n.T("Attach to a running container"),
		Long:    "Attach to a process that is already running inside an existing container.",
		Example: attachExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run())
		},
	}
	cmdutil.AddPodRunningTimeoutFlag(cmd, defaultPodAttachTimeout)
	cmd.Flags().StringVarP(&options.ContainerName, "container", "c", "", "Container name. If omitted, the first container in the pod will be chosen")
	cmd.Flags().BoolVarP(&options.Stdin, "stdin", "i", false, "Pass stdin to the container")
	cmd.Flags().BoolVarP(&options.TTY, "tty", "t", false, "Stdin is a TTY")
	return cmd
}

// RemoteAttach defines the interface accepted by the Attach command - provided for test stubbing
type RemoteAttach interface {
	Attach(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue remotecommand.TerminalSizeQueue) error
}

// DefaultRemoteAttach is the standard implementation of attaching
type DefaultRemoteAttach struct{}

func (*DefaultRemoteAttach) Attach(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue remotecommand.TerminalSizeQueue) error {
	exec, err := remotecommand.NewSPDYExecutor(config, method, url)
	if err != nil {
		return err
	}
	return exec.Stream(remotecommand.StreamOptions{
		Stdin:             stdin,
		Stdout:            stdout,
		Stderr:            stderr,
		Tty:               tty,
		TerminalSizeQueue: terminalSizeQueue,
	})
}

// AttachOptions declare the arguments accepted by the Exec command
type AttachOptions struct {
	StreamOptions

	CommandName string

	Pod *api.Pod

	Attach        RemoteAttach
	PodClient     coreclient.PodsGetter
	GetPodTimeout time.Duration
	Config        *restclient.Config
}

// Complete verifies command line arguments and loads data from the command environment
func (p *AttachOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, argsIn []string) error {
	if len(argsIn) == 0 {
		return cmdutil.UsageErrorf(cmd, "at least 1 argument is required for attach")
	}
	if len(argsIn) > 2 {
		return cmdutil.UsageErrorf(cmd, "expected POD, TYPE/NAME, or TYPE NAME, (at most 2 arguments) saw %d: %v", len(argsIn), argsIn)
	}

	namespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	p.GetPodTimeout, err = cmdutil.GetPodRunningTimeoutFlag(cmd)
	if err != nil {
		return cmdutil.UsageErrorf(cmd, err.Error())
	}

	builder := f.NewBuilder().
		NamespaceParam(namespace).DefaultNamespace()

	switch len(argsIn) {
	case 1:
		builder.ResourceNames("pods", argsIn[0])
	case 2:
		builder.ResourceNames(argsIn[0], argsIn[1])
	}

	obj, err := builder.Do().Object()
	if err != nil {
		return err
	}

	attachablePod, err := f.AttachablePodForObject(obj, p.GetPodTimeout)
	if err != nil {
		return err
	}

	p.PodName = attachablePod.Name
	p.Namespace = namespace

	config, err := f.ClientConfig()
	if err != nil {
		return err
	}
	p.Config = config

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}
	p.PodClient = clientset.Core()

	if p.CommandName == "" {
		p.CommandName = cmd.CommandPath()
	}

	return nil
}

// Validate checks that the provided attach options are specified.
func (p *AttachOptions) Validate() error {
	allErrs := []error{}
	if len(p.PodName) == 0 {
		allErrs = append(allErrs, errors.New("pod name must be specified"))
	}
	if p.Out == nil || p.Err == nil {
		allErrs = append(allErrs, errors.New("both output and error output must be provided"))
	}
	if p.Attach == nil || p.PodClient == nil || p.Config == nil {
		allErrs = append(allErrs, errors.New("client, client config, and attach must be provided"))
	}
	return utilerrors.NewAggregate(allErrs)
}

// Run executes a validated remote execution against a pod.
func (p *AttachOptions) Run() error {
	if p.Pod == nil {
		pod, err := p.PodClient.Pods(p.Namespace).Get(p.PodName, metav1.GetOptions{})
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

	var sizeQueue remotecommand.TerminalSizeQueue
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

		restClient, err := restclient.RESTClientFor(p.Config)
		if err != nil {
			return err
		}
		// TODO: consider abstracting into a client invocation or client helper
		req := restClient.Post().
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
		}, legacyscheme.ParameterCodec)

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
