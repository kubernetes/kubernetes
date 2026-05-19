/*
Copyright 2022 The Kubernetes Authors.

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

package create

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/templates"
	"k8s.io/kubectl/pkg/util/term"
	"k8s.io/utils/ptr"
)

// TokenOptions is the data required to perform a token request operation.
type TokenOptions struct {
	// PrintFlags holds options necessary for obtaining a printer
	PrintFlags *genericclioptions.PrintFlags
	PrintObj   func(obj runtime.Object) error

	// Flags hold the parsed CLI flags.
	Flags *pflag.FlagSet

	// Name and namespace of service account to create a token for
	Name      string
	Namespace string

	// BoundObjectKind is the kind of object to bind the token to. Optional. Can be Pod or Secret.
	BoundObjectKind string
	// BoundObjectName is the name of the object to bind the token to. Required if BoundObjectKind is set.
	BoundObjectName string
	// BoundObjectUID is the uid of the object to bind the token to. If unset, defaults to the current uid of the bound object.
	BoundObjectUID string

	// Audiences indicate the valid audiences for the requested token. If unset, defaults to the Kubernetes API server audiences.
	Audiences []string

	// Duration is the requested token lifetime. Optional.
	Duration time.Duration

	// CoreClient is the API client used to request the token. Required.
	CoreClient corev1client.CoreV1Interface

	// IOStreams are the output streams for the operation. Required.
	genericiooptions.IOStreams
}

var (
	tokenLong = templates.LongDesc(`Request a service account token.`)

	tokenExample = templates.Examples(`
		# Request a token to authenticate to the kube-apiserver as the service account "myapp" in the current namespace
		kubectl create token myapp

		# Request a token for a service account in a custom namespace
		kubectl create token myapp --namespace myns

		# Request a token with a custom expiration
		kubectl create token myapp --duration 10m

		# Request a token with a custom audience
		kubectl create token myapp --audience https://example.com

		# Request a token bound to an instance of a Secret object
		kubectl create token myapp --bound-object-kind Secret --bound-object-name mysecret

		# Request a token bound to an instance of a Secret object with a specific UID
		kubectl create token myapp --bound-object-kind Secret --bound-object-name mysecret --bound-object-uid 0d4691ed-659b-4935-a832-355f77ee47cc
`)
)

var boundObjectKinds = map[string]string{
	"Pod":    "v1",
	"Secret": "v1",
	"Node":   "v1",
}

func NewTokenOpts(ioStreams genericiooptions.IOStreams) *TokenOptions {
	return &TokenOptions{
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// NewCmdCreateToken returns an initialized Command for 'create token' sub command
func NewCmdCreateToken(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	o := NewTokenOpts(ioStreams)

	cmd := &cobra.Command{
		Use:                   "token SERVICE_ACCOUNT_NAME",
		DisableFlagsInUseLine: true,
		Short:                 "Request a service account token",
		Long:                  tokenLong,
		Example:               tokenExample,
		ValidArgsFunction:     completion.ResourceNameCompletionFunc(f, "serviceaccount"),
		Run: func(cmd *cobra.Command, args []string) {
			if err := o.Complete(f, cmd, args); err != nil {
				cmdutil.CheckErr(err)
				return
			}
			if err := o.Validate(); err != nil {
				cmdutil.CheckErr(err)
				return
			}
			if err := o.Run(); err != nil {
				cmdutil.CheckErr(err)
				return
			}
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().StringArrayVar(&o.Audiences, "audience", o.Audiences, "Audience of the requested token. If unset, defaults to requesting a token for use with the Kubernetes API server. May be repeated to request a token valid for multiple audiences.")

	cmd.Flags().DurationVar(&o.Duration, "duration", o.Duration, "Requested lifetime of the issued token. If not set or if set to 0, the lifetime will be determined by the server automatically. The server may return a token with a longer or shorter lifetime.")

	cmd.Flags().StringVar(&o.BoundObjectKind, "bound-object-kind", o.BoundObjectKind, "Kind of an object to bind the token to. "+
		"Supported kinds are "+strings.Join(sets.List(sets.KeySet(boundObjectKinds)), ", ")+". "+
		"If set, --bound-object-name must be provided.")
	cmd.Flags().StringVar(&o.BoundObjectName, "bound-object-name", o.BoundObjectName, "Name of an object to bind the token to. "+
		"The token will expire when the object is deleted. "+
		"Requires --bound-object-kind.")
	cmd.Flags().StringVar(&o.BoundObjectUID, "bound-object-uid", o.BoundObjectUID, "UID of an object to bind the token to. "+
		"Requires --bound-object-kind and --bound-object-name. "+
		"If unset, the UID of the existing object is used.")

	o.Flags = cmd.Flags()

	return cmd
}

// Complete completes all the required options
func (o *TokenOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	o.Name, err = NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}

	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	client, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.CoreClient = client.CoreV1()

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	return nil
}

// Validate makes sure provided values for TokenOptions are valid
func (o *TokenOptions) Validate() error {
	if o.CoreClient == nil {
		return fmt.Errorf("no client provided")
	}
	if len(o.Name) == 0 {
		return fmt.Errorf("service account name is required")
	}
	if len(o.Namespace) == 0 {
		return fmt.Errorf("--namespace is required")
	}
	if o.Duration < 0 {
		return fmt.Errorf("--duration must be greater than or equal to 0")
	}
	if o.Duration%time.Second != 0 {
		return fmt.Errorf("--duration cannot be expressed in units less than seconds")
	}
	for _, aud := range o.Audiences {
		if len(aud) == 0 {
			return fmt.Errorf("--audience must not be an empty string")
		}
	}

	if len(o.BoundObjectKind) == 0 {
		if len(o.BoundObjectName) > 0 {
			return fmt.Errorf("--bound-object-name can only be set if --bound-object-kind is provided")
		}
		if len(o.BoundObjectUID) > 0 {
			return fmt.Errorf("--bound-object-uid can only be set if --bound-object-kind is provided")
		}
	} else {
		if _, ok := boundObjectKinds[o.BoundObjectKind]; !ok {
			return fmt.Errorf("supported --bound-object-kind values are %s", strings.Join(sets.List(sets.KeySet(boundObjectKinds)), ", "))
		}
		if len(o.BoundObjectName) == 0 {
			return fmt.Errorf("--bound-object-name is required if --bound-object-kind is provided")
		}
	}

	return nil
}

// Run requests a token
func (o *TokenOptions) Run() error {
	request := &authenticationv1.TokenRequest{
		Spec: authenticationv1.TokenRequestSpec{
			Audiences: o.Audiences,
		},
	}
	if o.Duration > 0 {
		request.Spec.ExpirationSeconds = ptr.To(int64(o.Duration / time.Second))
	}
	if len(o.BoundObjectKind) > 0 {
		request.Spec.BoundObjectRef = &authenticationv1.BoundObjectReference{
			Kind:       o.BoundObjectKind,
			APIVersion: boundObjectKinds[o.BoundObjectKind],
			Name:       o.BoundObjectName,
			UID:        types.UID(o.BoundObjectUID),
		}
	}

	response, err := o.CoreClient.ServiceAccounts(o.Namespace).CreateToken(context.TODO(), o.Name, request, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create token: %v", err)
	}
	if len(response.Status.Token) == 0 {
		return fmt.Errorf("failed to create token: no token in server response")
	}

	if o.PrintFlags.OutputFlagSpecified() {
		return o.PrintObj(response)
	}

	if term.IsTerminal(o.Out) {
		// include a newline when printing interactively
		fmt.Fprintf(o.Out, "%s\n", response.Status.Token)
	} else {
		// otherwise just print the token
		fmt.Fprintf(o.Out, "%s", response.Status.Token)
	}

	return nil
}
