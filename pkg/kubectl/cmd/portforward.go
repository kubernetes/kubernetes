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
	"net/url"
	"os"
	"os/signal"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/portforward"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const (
	portforward_example = `
# Listen on ports 5000 and 6000 locally, forwarding data to/from ports 5000 and 6000 in the pod
kubectl port-forward mypod 5000 6000

# Listen on port 8888 locally, forwarding to 5000 in the pod
kubectl port-forward mypod 8888:5000

# Listen on a random port locally, forwarding to 5000 in the pod
kubectl port-forward mypod :5000

# Listen on a random port locally, forwarding to 5000 in the pod
kubectl port-forward  mypod 0:5000`
)

func NewCmdPortForward(f *cmdutil.Factory, cmdOut, cmdErr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "port-forward POD [LOCAL_PORT:]REMOTE_PORT [...[LOCAL_PORT_N:]REMOTE_PORT_N]",
		Short:   "Forward one or more local ports to a pod.",
		Long:    "Forward one or more local ports to a pod.",
		Example: portforward_example,
		Run: func(cmd *cobra.Command, args []string) {
			pf := &defaultPortForwarder{
				cmdOut: cmdOut,
				cmdErr: cmdErr,
			}
			err := RunPortForward(f, cmd, args, pf)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().StringP("pod", "p", "", "Pod name")
	// TODO support UID
	return cmd
}

type portForwarder interface {
	ForwardPorts(method string, url *url.URL, config *restclient.Config, ports []string, stopChan <-chan struct{}) error
}

type defaultPortForwarder struct {
	cmdOut, cmdErr io.Writer
}

func (f *defaultPortForwarder) ForwardPorts(method string, url *url.URL, config *restclient.Config, ports []string, stopChan <-chan struct{}) error {
	dialer, err := remotecommand.NewExecutor(config, method, url)
	if err != nil {
		return err
	}
	fw, err := portforward.New(dialer, ports, stopChan, f.cmdOut, f.cmdErr)
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

	postErr := fw.ForwardPorts("POST", req.URL(), config, args, stopCh)

	// if we don't have an error, return.  If we did get an error, try a GET because v3.0.0 shipped with port-forward running as a GET.
	if postErr == nil {
		return nil
	}
	// only try the get if the error is either a forbidden or method not supported, otherwise trying with a GET probably won't help
	if !apierrors.IsForbidden(postErr) && !apierrors.IsMethodNotSupported(postErr) {
		return postErr
	}

	getReq := client.RESTClient.Get().
		Resource("pods").
		Namespace(namespace).
		Name(pod.Name).
		SubResource("portforward")

	getErr := fw.ForwardPorts("GET", getReq.URL(), config, args, stopCh)
	if getErr == nil {
		return nil
	}

	// if we got a getErr, return the postErr because it's more likely to be correct.  GET is legacy
	return postErr
}
