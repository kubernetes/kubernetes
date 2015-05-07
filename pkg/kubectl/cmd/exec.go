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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/remotecommand"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/docker/docker/pkg/term"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

const (
	exec_example = `// get output from running 'date' in ruby-container from pod 123456-7890
$ kubectl exec -p 123456-7890 -c ruby-container date

//switch to raw terminal mode, sends stdin to 'bash' in ruby-container from pod 123456-780 and sends stdout/stderr from 'bash' back to the client
$ kubectl exec -p 123456-7890 -c ruby-container -i -t -- bash -il`
)

func NewCmdExec(f *cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "exec -p POD -c CONTAINER -- COMMAND [args...]",
		Short:   "Execute a command in a container.",
		Long:    "Execute a command in a container.",
		Example: exec_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunExec(f, cmdIn, cmdOut, cmdErr, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().StringP("pod", "p", "", "Pod name")
	cmd.MarkFlagRequired("pod")
	// TODO support UID
	cmd.Flags().StringP("container", "c", "", "Container name")
	cmd.MarkFlagRequired("container")
	cmd.Flags().BoolP("stdin", "i", false, "Pass stdin to the container")
	cmd.Flags().BoolP("tty", "t", false, "Stdin is a TTY")
	return cmd
}

func RunExec(f *cmdutil.Factory, cmdIn io.Reader, cmdOut, cmdErr io.Writer, cmd *cobra.Command, args []string) error {
	podName := cmdutil.GetFlagString(cmd, "pod")
	if len(podName) == 0 {
		return cmdutil.UsageError(cmd, "POD is required for exec")
	}

	if len(args) < 1 {
		return cmdutil.UsageError(cmd, "COMMAND is required for exec")
	}

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
		glog.Fatalf("Unable to execute command because pod is not running. Current status=%v", pod.Status.Phase)
	}

	containerName := cmdutil.GetFlagString(cmd, "container")
	if len(containerName) == 0 {
		containerName = pod.Spec.Containers[0].Name
	}

	var stdin io.Reader
	tty := cmdutil.GetFlagBool(cmd, "tty")
	if cmdutil.GetFlagBool(cmd, "stdin") {
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
		Prefix("proxy").
		Resource("nodes").
		Name(pod.Spec.Host).
		Suffix("exec", namespace, podName, containerName)

	e := remotecommand.New(req, config, args, stdin, cmdOut, cmdErr, tty)
	return e.Execute()
}
