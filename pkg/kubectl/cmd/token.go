/*
Copyright 2018 The Kubernetes Authors.

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

	"github.com/spf13/cobra"
	"k8s.io/api/authentication/v1"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	tokenExample = templates.Examples(i18n.T(`
		# Request token from serviceaccount test-sa and bounded with pod test-pod
		kubectl token --pod=test-pod test-sa

		# Request token from serviceaccount test-sa and bounded with secret test-secret
		kubectl token --secret=test-secret test-sa

		# Request token with specified expiration time in seconds
		kubectl token --pod=test-pod --exp-time=20000 test-sa`))
)

type TokenOptions struct {
	Name              string
	Namespace         string
	Pod               string
	Secret            string
	ExpirationSeconds int64
	Audiences         []string
	Client            corev1.CoreV1Interface
	Cmd               *cobra.Command
	Out               io.Writer
	OutputFormat      string
}

func NewCmdToken(f cmdutil.Factory, out io.Writer) *cobra.Command {
	o := &TokenOptions{}
	cmd := &cobra.Command{
		Use:     "token NAME --pod=pod/--secret=secret",
		Short:   i18n.T("Request token with bounded pod/secret"),
		Long:    "Request token with bounded pod or secret. It is a subresource of a serviceaccount.",
		Example: tokenExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, out, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunToken())
		},
	}

	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().StringVar(&o.Pod, "pod", o.Pod, "Specify a pod bounded to the token. ")
	cmd.Flags().StringVar(&o.Secret, "secret", o.Secret, "Specify a secret bounded to the token")
	cmd.Flags().StringSliceVar(&o.Audiences, "audience", o.Audiences, "Specify the audience of the token")
	cmd.Flags().Int64Var(&o.ExpirationSeconds, "exp-time", int64(3600), "The token's expirtion time in seconds")
	return cmd
}

func (o *TokenOptions) Complete(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	var err error
	if len(args) != 1 {
		return cmdutil.UsageErrorf(cmd, "exactly one NAME is required, got %d", len(args))
	}

	o.Name = args[0]
	o.Namespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}

	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.Client = clientset.Core()
	o.Cmd = cmd
	o.Out = out
	o.OutputFormat = cmdutil.GetFlagString(cmd, "output")

	return nil
}

func (o *TokenOptions) Validate() error {
	if o.Pod != "" && o.Secret != "" {
		return fmt.Errorf("can't specific pod and secret at same time")
	}
	if o.Pod == "" && o.Secret == "" {
		return fmt.Errorf("pod and secret must specify one")
	}
	if o.Name == "" {
		return fmt.Errorf("serviceaccount must be specificed")
	}
	return nil
}

func (o *TokenOptions) RunToken() error {
	tokenRequest := v1.TokenRequest{
		Spec: v1.TokenRequestSpec{},
	}
	//tokenRequest.TokenRequestSpec = v1.TokenRequestSpec{}
	objectRef := &v1.BoundObjectReference{}

	if o.Pod != "" {
		objectRef.Kind = "Pod"
		objectRef.Name = o.Pod
	}
	if o.Secret != "" {
		objectRef.Kind = "Secret"
		objectRef.Name = o.Secret
	}

	if len(o.Audiences) != 0 {
		tokenRequest.Spec.Audiences = o.Audiences
	}

	tokenRequest.Spec.ExpirationSeconds = &o.ExpirationSeconds
	tokenRequest.Spec.BoundObjectRef = objectRef

	ret, err := o.Client.ServiceAccounts(o.Namespace).CreateToken(o.Name, &tokenRequest)
	if err != nil {
		return err
	}
	if len(o.OutputFormat) > 0 {
		return cmdutil.PrintObject(o.Cmd, ret, o.Out)
	}
	fmt.Fprintln(o.Out, ret.Status.Token)
	return nil
}
