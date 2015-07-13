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
	"os"
	"os/signal"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/portforward"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const (
	portforward_example = `
// listens on ports 5000 and 6000 locally, forwarding data to/from ports 5000 and 6000 in the pod
$ kubectl port-forward mypod 5000 6000

// listens on port 8888 locally, forwarding to 5000 in the pod
$ kubectl port-forward mypod 8888:5000

// listens on a random port locally, forwarding to 5000 in the pod
$ kubectl port-forward mypod :5000

// listens on a random port locally, forwarding to 5000 in the pod
$ kubectl port-forward  mypod 0:5000`
)

func NewCmdPortForward(f *cmdutil.Factory) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "port-forward POD [LOCAL_PORT:]REMOTE_PORT [...[LOCAL_PORT_N:]REMOTE_PORT_N]",
		Short:   "Forward one or more local ports to a pod.",
		Long:    "Forward one or more local ports to a pod.",
		Example: portforward_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunPortForward(f, cmd, args, &defaultPortForwarder{})
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().StringP("pod", "p", "", "Pod name")
	// TODO support UID
	return cmd
}

type portForwarder interface {
	ForwardPorts(req *client.Request, config *client.Config, ports []string, stopChan <-chan struct{}) error
}

type defaultPortForwarder struct{}

func (*defaultPortForwarder) ForwardPorts(req *client.Request, config *client.Config, ports []string, stopChan <-chan struct{}) error {
	fw, err := portforward.New(req, config, ports, stopChan)
	if err != nil {
		return err
	}
	return fw.ForwardPorts()
}

func RunPortForward(f *cmdutil.Factory, cmd *cobra.Command, args []string, fw portForwarder) error {
	podName := cmdutil.GetFlagString(cmd, "pod")
	if len(podName) == 0 && len(args) == 0 {
		return cmdutil.UsageError(cmd, "POD is required for port-forward")
	}

	if len(podName) != 0 {
		printDeprecationWarning("port-forward POD", "-p POD")
	} else {
		podName = args[0]
		args = args[1:]
	}

	if len(args) < 1 {
		return cmdutil.UsageError(cmd, "at least 1 PORT is required for port-forward")
	}

	namespace, _, err := f.DefaultNamespace()
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

	config, err := f.ClientConfig()
	if err != nil {
		return err
	}

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt)
	defer signal.Stop(signals)

	stopCh := make(chan struct{}, 1)
	go func() {
		<-signals
		close(stopCh)
	}()

	req := client.RESTClient.Post().
		Resource("pods").
		Namespace(namespace).
		Name(pod.Name).
		SubResource("portforward")

	return fw.ForwardPorts(req, config, args, stopCh)
}
