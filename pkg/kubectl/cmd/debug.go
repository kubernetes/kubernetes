/*
Copyright 2017 The Kubernetes Authors.

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

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	remotecommandconsts "k8s.io/apimachinery/pkg/util/remotecommand"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/kubernetes/pkg/api"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

var (
	debug_example = templates.Examples(`
		# Switch to raw terminal mode, sends stdin to 'bash' in a new container in pod 123456-7890
		# named 'debug' from image 'debian and sends stdout/stderr from 'bash' back to the client
		kubectl debug 123456-7890 -c debug -m debian -i -t -- bash -il`)
)

const (
	debugUsageStr             = "expected 'debug POD_NAME [--] [COMMAND [ARG1 ... ARGN]]'.\nPOD_NAME is a required argument for the debug command.\nCOMMAND may be required if the container does not provide an ENTRYPOINT"
	debugDefaultContainerName = "debug"
	debugDefaultImageName     = "debian"
)

func NewCmdDebug(f cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer) *cobra.Command {
	options := &DebugOptions{
		StreamOptions: StreamOptions{
			In:  cmdIn,
			Out: cmdOut,
			Err: cmdErr,
		},

		Debugger: &DefaultRemoteDebugger{},
	}
	cmd := &cobra.Command{
		Use:     "debug POD [-c CONTAINER] -- COMMAND [args...]",
		Short:   i18n.T("Run a debug container in a pod"),
		Long:    "Run a Debug Container inside an existing pod.",
		Example: debug_example,
		Run: func(cmd *cobra.Command, args []string) {
			argsLenAtDash := cmd.ArgsLenAtDash()
			cmdutil.CheckErr(options.Complete(f, cmd, args, argsLenAtDash))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunDebug())
		},
	}
	cmd.Flags().StringVarP(&options.ContainerName, "container", "c", "", fmt.Sprintf("Container name. [default=%s]", debugDefaultContainerName))
	cmd.Flags().StringVarP(&options.ImageName, "image", "m", "", fmt.Sprintf("Image name. [default=%s]", debugDefaultImageName))
	cmd.Flags().BoolVarP(&options.Stdin, "stdin", "i", false, "Pass stdin to the container")
	cmd.Flags().BoolVarP(&options.TTY, "tty", "t", false, "Stdin is a TTY")
	return cmd
}

// RemoteDebugger defines the interface accepted by the Debug command - provided for test stubbing
type RemoteDebugger interface {
	Debug(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue remotecommand.TerminalSizeQueue) error
}

// DefaultRemoteDebugger is the standard implementation of remote command debugging
type DefaultRemoteDebugger struct{}

func (*DefaultRemoteDebugger) Debug(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue remotecommand.TerminalSizeQueue) error {
	debug, err := remotecommand.NewExecutor(config, method, url)
	if err != nil {
		return err
	}
	return debug.Stream(remotecommand.StreamOptions{
		SupportedProtocols: remotecommandconsts.SupportedStreamingProtocols,
		Stdin:              stdin,
		Stdout:             stdout,
		Stderr:             stderr,
		Tty:                tty,
		TerminalSizeQueue:  terminalSizeQueue,
	})
}

// DebugOptions declare the arguments accepted by the Debug command
type DebugOptions struct {
	StreamOptions
	ImageName string

	Command []string

	FullCmdName string

	Debugger  RemoteDebugger
	PodClient coreclient.PodsGetter
	Config    *restclient.Config
}

// Complete verifies command line arguments and loads data from the command environment
func (p *DebugOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, argsIn []string, argsLenAtDash int) error {
	if len(argsIn) == 0 || argsLenAtDash == 0 {
		return cmdutil.UsageError(cmd, debugUsageStr)
	}
	p.PodName = argsIn[0]
	p.Command = argsIn[1:]

	if len(p.ContainerName) == 0 {
		p.ContainerName = debugDefaultContainerName
	}

	if len(p.ImageName) == 0 {
		p.ImageName = debugDefaultImageName
	}

	cmdParent := cmd.Parent()
	if cmdParent != nil {
		p.FullCmdName = cmdParent.CommandPath()
	}

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

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}
	p.PodClient = clientset.Core()

	return nil
}

// Validate checks that the provided debug options are specified.
func (p *DebugOptions) Validate() error {
	if len(p.PodName) == 0 {
		return fmt.Errorf("pod name must be specified")
	}
	if p.Out == nil || p.Err == nil {
		return fmt.Errorf("both output and error output must be provided")
	}
	if p.Debugger == nil || p.PodClient == nil || p.Config == nil {
		return fmt.Errorf("client, client config, and debugger must be provided")
	}
	return nil
}

// RunDebug debugs a validated remote debugging against a pod.
func (p *DebugOptions) RunDebug() error {
	pod, err := p.PodClient.Pods(p.Namespace).Get(p.PodName, metav1.GetOptions{})
	if err != nil {
		return err
	}

	if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
		return fmt.Errorf("cannot debug into a container in a completed pod; current phase is %s", pod.Status.Phase)
	}

	// ensure we can recover the terminal while attached
	t := p.setupTTY()

	var sizeQueue remotecommand.TerminalSizeQueue
	if t.Raw {
		// this call spawns a goroutine to monitor/update the terminal size
		sizeQueue = t.MonitorSize(t.GetSize())

		// unset p.Err if it was previously set because both stdout and stderr go over p.Out when tty is
		// true
		p.Err = nil
	}

	fn := func() error {
		restClient, err := restclient.RESTClientFor(p.Config)
		if err != nil {
			return err
		}

		req := restClient.Post().
			Resource("pods").
			Name(pod.Name).
			Namespace(pod.Namespace).
			SubResource("debug").
			Param("image", p.ImageName) // TODO(verb): replace with versioned param when available
		req.VersionedParams(&api.PodExecOptions{
			Container: p.ContainerName,
			Command:   p.Command,
			Stdin:     p.Stdin,
			Stdout:    p.Out != nil,
			Stderr:    p.Err != nil,
			TTY:       t.Raw,
		}, api.ParameterCodec)

		return p.Debugger.Debug("POST", req.URL(), p.Config, p.In, p.Out, p.Err, t.Raw, sizeQueue)
	}

	return t.Safe(fn)
}
