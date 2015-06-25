/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/remotecommand"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/docker/docker/pkg/term"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

const (
	exec_example = `// get output from running 'date' from pod 123456-7890, using the first container by default
$ kubectl exec 123456-7890 date
	
// get output from running 'date' in ruby-container from pod 123456-7890
$ kubectl exec 123456-7890 -c ruby-container date

// switch to raw terminal mode, sends stdin to 'bash' in ruby-container from pod 123456-780
// and sends stdout/stderr from 'bash' back to the client
$ kubectl exec 123456-7890 -c ruby-container -i -t -- bash -il`
)

func NewCmdExec(f *cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer) *cobra.Command {
	params := &execParams{}
	cmd := &cobra.Command{
		Use:     "exec POD -c CONTAINER -- COMMAND [args...]",
		Short:   "Execute a command in a container.",
		Long:    "Execute a command in a container.",
		Example: exec_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunExec(f, cmd, cmdIn, cmdOut, cmdErr, params, args, &defaultRemoteExecutor{})
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().StringVarP(&params.podName, "pod", "p", "", "Pod name")
	// TODO support UID
	cmd.Flags().StringVarP(&params.containerName, "container", "c", "", "Container name")
	cmd.Flags().BoolVarP(&params.stdin, "stdin", "i", false, "Pass stdin to the container")
	cmd.Flags().BoolVarP(&params.tty, "tty", "t", false, "Stdin is a TTY")
	return cmd
}

type remoteExecutor interface {
	Execute(req *client.Request, config *client.Config, command []string, stdin io.Reader, stdout, stderr io.Writer, tty bool) error
}

type defaultRemoteExecutor struct{}

func (*defaultRemoteExecutor) Execute(req *client.Request, config *client.Config, command []string, stdin io.Reader, stdout, stderr io.Writer, tty bool) error {
	executor := remotecommand.New(req, config, command, stdin, stdout, stderr, tty)
	return executor.Execute()
}

type execParams struct {
	podName       string
	containerName string
	stdin         bool
	tty           bool
}

func extractPodAndContainer(cmd *cobra.Command, argsIn []string, p *execParams) (podName string, containerName string, args []string, err error) {
	if len(p.podName) == 0 && len(argsIn) == 0 {
		return "", "", nil, cmdutil.UsageError(cmd, "POD is required for exec")
	}
	if len(p.podName) != 0 {
		printDeprecationWarning("exec POD", "-p POD")
		podName = p.podName
		if len(argsIn) < 1 {
			return "", "", nil, cmdutil.UsageError(cmd, "COMMAND is required for exec")
		}
		args = argsIn
	} else {
		podName = argsIn[0]
		args = argsIn[1:]
		if len(args) < 1 {
			return "", "", nil, cmdutil.UsageError(cmd, "COMMAND is required for exec")
		}
	}
	return podName, p.containerName, args, nil
}

func RunExec(f *cmdutil.Factory, cmd *cobra.Command, cmdIn io.Reader, cmdOut, cmdErr io.Writer, p *execParams, argsIn []string, re remoteExecutor) error {
	podName, containerName, args, err := extractPodAndContainer(cmd, argsIn, p)
	namespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	client, err := f.Client()
	if err != nil {
		return err
	}

	pod, err := client.Pods(namespace).Get(podName)
	if err != nil {
		return err
	}

	if pod.Status.Phase != api.PodRunning {
		glog.Fatalf("Unable to execute command because pod %s is not running. Current status=%v", podName, pod.Status.Phase)
	}

	if len(containerName) == 0 {
		glog.V(4).Infof("defaulting container name to %s", pod.Spec.Containers[0].Name)
		containerName = pod.Spec.Containers[0].Name
	}

	var stdin io.Reader
	tty := p.tty
	if p.stdin {
		stdin = cmdIn
		if tty {
			if file, ok := cmdIn.(*os.File); ok {
				inFd := file.Fd()
				if term.IsTerminal(inFd) {
					oldState, err := term.SetRawTerminal(inFd)
					if err != nil {
						glog.Fatal(err)
					}
					// this handles a clean exit, where the command finished
					defer term.RestoreTerminal(inFd, oldState)

					// SIGINT is handled by term.SetRawTerminal (it runs a goroutine that listens
					// for SIGINT and restores the terminal before exiting)

					// this handles SIGTERM
					sigChan := make(chan os.Signal, 1)
					signal.Notify(sigChan, syscall.SIGTERM)
					go func() {
						<-sigChan
						term.RestoreTerminal(inFd, oldState)
						os.Exit(0)
					}()
				} else {
					glog.Warning("Stdin is not a terminal")
				}
			} else {
				tty = false
				glog.Warning("Unable to use a TTY")
			}
		}
	}

	config, err := f.ClientConfig()
	if err != nil {
		return err
	}

	req := client.RESTClient.Get().
		Resource("pods").
		Name(pod.Name).
		Namespace(namespace).
		SubResource("exec").
		Param("container", containerName)

	return re.Execute(req, config, args, stdin, cmdOut, cmdErr, tty)
}
