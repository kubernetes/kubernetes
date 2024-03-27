/*
Copyright 2023 The Kubernetes Authors.

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

package checkpoint

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/spf13/cobra"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	restclient "k8s.io/client-go/rest"

	"k8s.io/apimachinery/pkg/api/meta"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/cmd/util/podcmd"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	checkpointExample = templates.Examples(i18n.T(`
		# Checkpoint container from pod mypod, using the first container by default
		kubectl alpha checkpoint mypod

		# Checkpoint container ruby-container from pod mypod
		kubectl alpha checkpoint mypod -c ruby-container

		# Checkpoint container from the first pod of the deployment mydeployment, using the first container by default
		kubectl alpha checkpoint deploy/mydeployment

		# Checkpoint container from the first pod of the service myservice, using the first container by default
		kubectl alpha checkpoint svc/myservice
		`))
)

const (
	defaultPodCheckpointTimeout = 60 * time.Second
)

func NewCmdCheckpoint(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	options := &CheckpointOptions{
		Checkpointer: &DefaultRemoteCheckpointer{},
	}
	cmd := &cobra.Command{
		Use:                   "checkpoint (POD | TYPE/NAME) [-c CONTAINER] [flags]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Checkpoint a container"),
		Long:                  i18n.T("Checkpoint a container."),
		Example:               checkpointExample,
		ValidArgsFunction:     completion.PodResourceNameCompletionFunc(f),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.Run(f))
		},
	}
	cmdutil.AddPodRunningTimeoutFlag(cmd, defaultPodCheckpointTimeout)
	cmdutil.AddContainerVarFlags(cmd, &options.ContainerName, options.ContainerName)
	cmdutil.CheckErr(cmd.RegisterFlagCompletionFunc("container", completion.ContainerCompletionFunc(f)))
	cmd.Flags().BoolVarP(&options.Quiet, "quiet", "q", options.Quiet, "Only print output from the remote session")

	return cmd
}

// RemoteCheckpointer defines the checkpoint interface - used for testing
type RemoteCheckpointer interface {
	Checkpoint(request *restclient.Request) (string, error)
}

type DefaultRemoteCheckpointer struct{}

func (*DefaultRemoteCheckpointer) EvaluateResult() {

}

func (*DefaultRemoteCheckpointer) Checkpoint(request *restclient.Request) (string, error) {
	type checkpointResult struct {
		Items []string `json:"items"`
	}

	result := request.Do(context.Background())
	err := result.Error()

	if err != nil {
		return "", err
	}

	var statusCode int
	result = result.StatusCode(&statusCode)

	if statusCode != http.StatusOK {
		return "", fmt.Errorf(
			"unexpected status code (%d) returned. Expected %d",
			statusCode,
			http.StatusOK,
		)
	}

	body, err := result.Raw()
	if err != nil {
		return "", err
	}
	answer := checkpointResult{}
	err = json.Unmarshal(body, &answer)
	if err != nil {
		return "", err
	}

	if len(answer.Items) != 1 {
		return "", fmt.Errorf("expected exactly 1 answer but got %d", len(answer.Items))
	}

	return answer.Items[0], nil
}

// CheckpointOptions declare the arguments accepted by the Checkpoint command
type CheckpointOptions struct {
	Namespace        string
	ContainerName    string
	Quiet            bool
	ResourceName     string
	EnforceNamespace bool
	Builder          func() *resource.Builder
	PodFunction      polymorphichelpers.AttachablePodForObjectFunc
	restClientGetter genericclioptions.RESTClientGetter
	Pod              *corev1.Pod
	GetPodTimeout    time.Duration
	Config           *restclient.Config
	Checkpointer     RemoteCheckpointer
}

// Complete verifies command line arguments and loads data from the command environment
func (p *CheckpointOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, argsIn []string) error {
	if len(argsIn) != 1 {
		return cmdutil.UsageErrorf(cmd, "please specify exactly one pod or type/name")
	}

	p.ResourceName = argsIn[0]

	var err error
	p.Namespace, p.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	p.PodFunction = polymorphichelpers.AttachablePodForObjectFn

	p.GetPodTimeout, err = cmdutil.GetPodRunningTimeoutFlag(cmd)
	if err != nil {
		return cmdutil.UsageErrorf(cmd, err.Error())
	}

	p.Builder = f.NewBuilder
	p.restClientGetter = f

	p.Config, err = f.ToRESTConfig()
	if err != nil {
		return err
	}

	return nil
}

func (p *CheckpointOptions) Validate() error {
	if len(p.ResourceName) == 0 {
		return fmt.Errorf("pod or type/name must be specified")
	}
	return nil
}

// Run tries to checkpoint the given container (or the first container found) in
// the given pod.
func (p *CheckpointOptions) Run(f cmdutil.Factory) error {
	builder := p.Builder().
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		NamespaceParam(p.Namespace).DefaultNamespace()
	builder = builder.ResourceNames("pods", p.ResourceName)

	obj, err := builder.Do().Object()
	if err != nil {
		return err
	}

	if meta.IsListType(obj) {
		return fmt.Errorf("cannot checkpoint multiple objects at a time")
	}

	p.Pod, err = p.PodFunction(p.restClientGetter, obj, p.GetPodTimeout)
	if err != nil {
		return err
	}

	pod := p.Pod

	if pod.Status.Phase != corev1.PodRunning {
		return fmt.Errorf("cannot checkpoint a container in non-running pod; current phase is %s", pod.Status.Phase)
	}

	containerName := p.ContainerName
	if len(containerName) == 0 {
		container, err := podcmd.FindOrDefaultContainerByName(pod, containerName, p.Quiet, os.Stderr)
		if err != nil {
			return err
		}
		containerName = container.Name
	}

	restClient, err := restclient.RESTClientFor(p.Config)
	if err != nil {
		return err
	}
	req := restClient.Post().
		Resource("pods").
		Name(pod.Name).
		Namespace(pod.Namespace).
		SubResource("checkpoint")
	req.VersionedParams(&corev1.PodCheckpointOptions{
		Container: containerName,
	}, scheme.ParameterCodec)

	// indirection to be able to provide mock function for testing
	result, err := p.Checkpointer.Checkpoint(req)
	if err != nil {
		return err
	}

	fmt.Printf("Node:\t\t\t%s\n", pod.Spec.NodeName+"/"+pod.Status.HostIP)
	fmt.Printf("Namespace:\t\t%s\n", pod.Namespace)
	fmt.Printf("Pod:\t\t\t%s\n", pod.Name)
	fmt.Printf("Container:\t\t%s\n", containerName)
	fmt.Printf("Checkpoint Archive:\t%s\n", result)

	return nil
}
