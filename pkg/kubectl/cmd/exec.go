/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/docker/docker/pkg/term"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

func (f *Factory) NewCmdExec(cmdIn io.Reader, cmdOut, cmdErr io.Writer) *cobra.Command {
	flags := &struct {
		pod       string
		container string
		stdin     bool
		tty       bool
	}{}

	cmd := &cobra.Command{
		Use:   "exec -p <pod> -c <container> -- <command> [<args...>]",
		Short: "Execute a command in a container.",
		Long: `Execute a command in a container.
Examples:
  $ kubectl exec -p 123456-7890 -c ruby-container date
  <returns output from running 'date' in ruby-container from pod 123456-7890>

  $ kubectl exec -p 123456-7890 -c ruby-container -i -t -- bash -il
  <switches to raw terminal mode, sends stdin to 'bash' in ruby-container from
   pod 123456-780 and sends stdout/stderr from 'bash' back to the client`,
		Run: func(cmd *cobra.Command, args []string) {
			if len(flags.pod) == 0 {
				usageError(cmd, "<pod> is required for exec")
			}

			if len(args) < 1 {
				usageError(cmd, "<command> is required for exec")
			}

			namespace, err := f.DefaultNamespace(cmd)
			checkErr(err)

			client, err := f.Client(cmd)
			checkErr(err)

			pod, err := client.Pods(namespace).Get(flags.pod)
			checkErr(err)

			if pod.Status.Phase != api.PodRunning {
				glog.Fatalf("Unable to execute command because pod is not running. Current status=%v", pod.Status.Phase)
			}

			if len(flags.container) == 0 {
				flags.container = pod.Spec.Containers[0].Name
			}

			var stdin io.Reader
			if util.GetFlagBool(cmd, "stdin") {
				stdin = cmdIn
				if flags.tty {
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
						flags.tty = false
						glog.Warning("Unable to use a TTY")
					}
				}
			}

			config, err := f.ClientConfig(cmd)
			checkErr(err)

			req := client.RESTClient.Get().
				Prefix("proxy").
				Resource("minions").
				Name(pod.Status.Host).
				Suffix("exec", namespace, flags.pod, flags.container)

			e := remotecommand.New(req, config, args, stdin, cmdOut, cmdErr, flags.tty)
			err = e.Execute()
			checkErr(err)
		},
	}
	cmd.Flags().StringVarP(&flags.pod, "pod", "p", "", "Pod name")
	// TODO support UID
	cmd.Flags().StringVarP(&flags.container, "container", "c", "", "Container name")
	cmd.Flags().BoolVarP(&flags.stdin, "stdin", "i", false, "Pass stdin to the container")
	cmd.Flags().BoolVarP(&flags.tty, "tty", "t", false, "Stdin is a TTY")
	return cmd
}
