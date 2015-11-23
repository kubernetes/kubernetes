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
	"fmt"
	"io"
	"math/rand"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/labels"

	docker "github.com/fsouza/go-dockerclient"
)

const (
	push_example = `
# Push image to cluster registry, use default registry name and port, i.e. kube-registry:5000
$ kubectl push nginx:1.9.3

# Push image to cluster registry, use registry name "my-registry", port "6000"
$ kubectl push nginx:1.9.3 -r my-registry -p 6000`

	default_endpoint      = "unix:///var/run/docker.sock"
	default_registry_name = "kube-registry"
	default_registry_port = "5000"
)

func NewCmdPush(f *cmdutil.Factory) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "push image [IMAGE_NAME]",
		Short:   "Push image to cluster reigstry.",
		Long:    "Push image to cluster reigstry.",
		Example: push_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunPush(f, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	// Default registry name "kube-registry" and port "5000", which matches addon configs.
	cmd.Flags().StringP("host", "H", default_endpoint, "Docker host to connect to")
	cmd.Flags().StringP("registry", "r", default_registry_name, "Registry service name")
	cmd.Flags().StringP("port", "p", default_registry_port, "Registry service port")
	return cmd
}

func RunPush(f *cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return cmdutil.UsageError(cmd, "IMAGE is required for pushing")
	}
	imageName := args[0]
	endpoint := cmdutil.GetFlagString(cmd, "host")
	registryName := cmdutil.GetFlagString(cmd, "registry")
	port := cmdutil.GetFlagString(cmd, "port")

	// Find the image and tag it with localhost.
	dockerClient, err := docker.NewClient(endpoint)
	if err != nil {
		return err
	}

	host, repo, tag := parseRepositoryTag(imageName)
	if len(host) != 0 {
		fmt.Printf("Host %v will be rewritten to localhost:%s\n", host, port)
	}
	if len(tag) == 0 {
		tag = "latest"
	}

	tagOpts := docker.TagImageOptions{
		Repo:  fmt.Sprintf("localhost:%s/%s", port, repo),
		Tag:   tag,
		Force: true,
	}
	err = dockerClient.TagImage(imageName, tagOpts)
	if err != nil {
		return err
	}

	// Find registry service and pods.
	client, err := f.Client()
	if err != nil {
		return err
	}

	service, err := client.Services(api.NamespaceSystem).Get(registryName)
	if err != nil {
		return err
	}

	pods, err := client.Pods(api.NamespaceSystem).List(labels.SelectorFromSet(service.Spec.Selector), fields.Everything())
	if err != nil {
		return err
	}

	// Pick a random pod to start forwarding.
	found := false
	seed := rand.Intn(len(pods.Items))
	var pod api.Pod
	for i := 0; i < len(pods.Items); i++ {
		pod = pods.Items[(i+seed)%len(pods.Items)]
		if pod.Status.Phase != api.PodRunning {
			continue
		} else {
			found = true
			break
		}
	}
	if !found {
		return cmdutil.UsageError(cmd, "Unable to find a running registry pod")
	}

	// Start portfoward for between localhost and remote registry pod.
	config, err := f.ClientConfig()
	if err != nil {
		return err
	}

	stopCh := make(chan struct{}, 1)
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt)
	defer signal.Stop(signals)

	go func() {
		<-signals
		close(stopCh)
	}()

	// Start port-forwarding for the pods.
	fw := &defaultPortForwarder{}
	req := client.RESTClient.Post().
		Resource("pods").
		Namespace(api.NamespaceSystem).
		Name(pod.Name).
		SubResource("portforward")

	go fw.ForwardPorts("POST", req.URL(), config, []string{port}, stopCh)
	// TODO: Wait until portforwarder starts.
	time.Sleep(3 * time.Second)

	// Properly display push progress.
	reader, writer := io.Pipe()
	opts := docker.PushImageOptions{
		Name:          fmt.Sprintf("localhost:%s/%s", port, repo),
		Tag:           tag,
		Registry:      fmt.Sprintf("localhost:%s", port),
		OutputStream:  writer,
		RawJSONStream: true,
	}
	go jsonmessage.DisplayJSONMessagesStream(reader, os.Stdout, uintptr(syscall.Stdout), true)

	// Finally, push the image.
	glog.Infof("Push image localhost:%s/%s:%s", port, repo, tag)
	err = dockerClient.PushImage(opts, docker.AuthConfiguration{})
	if err != nil {
		return err
	}

	glog.Infof("Image pushed, use \"localhost:%s/%s:%s\" in Pod.Container.Image to access the image", port, repo, tag)
	return nil
}

// parseRepositoryTag parse an image name and returns host, repo, and tag.
//   Ex: localhost.localdomain:5000/samalba/hipache:latest;
//   Gives: localhost.localdomain:5000, samalba/hipache, latest
func parseRepositoryTag(repos string) (string, string, string) {
	n := strings.Index(repos, "@")
	if n >= 0 {
		parts := strings.Split(repos, "@")
		return parseHost(parts[0], parts[1])
	}
	n = strings.LastIndex(repos, ":")
	if n < 0 {
		return parseHost(repos, "")
	}
	if tag := repos[n+1:]; !strings.Contains(tag, "/") {
		return parseHost(repos[:n], tag)
	}
	return parseHost(repos, "")
}

func parseHost(host string, tag string) (string, string, string) {
	n := strings.Index(host, "/")
	if n > 0 {
		return host[:n], host[n+1:], tag
	}
	return "", host, tag
}
