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

package auth

import (
	"context"
	"fmt"
	"io"

	"github.com/spf13/cobra"
	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1alpha1 "k8s.io/api/authentication/v1alpha1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	authenticationv1client "k8s.io/client-go/kubernetes/typed/authentication/v1"
	authenticationv1alpha1client "k8s.io/client-go/kubernetes/typed/authentication/v1alpha1"
	authenticationv1beta1client "k8s.io/client-go/kubernetes/typed/authentication/v1beta1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/templates"
)

// WhoAmIFlags directly reflect the information that CLI is gathering via flags.  They will be converted to Options, which
// reflect the runtime requirements for the command.  This structure reduces the transformation to wiring and makes
// the logic itself easy to unit test.
type WhoAmIFlags struct {
	RESTClientGetter genericclioptions.RESTClientGetter
	PrintFlags       *genericclioptions.PrintFlags

	genericiooptions.IOStreams
}

// NewWhoAmIFlags returns a default WhoAmIFlags.
func NewWhoAmIFlags(restClientGetter genericclioptions.RESTClientGetter, streams genericiooptions.IOStreams) *WhoAmIFlags {
	return &WhoAmIFlags{
		RESTClientGetter: restClientGetter,
		PrintFlags:       genericclioptions.NewPrintFlags("").WithTypeSetter(scheme.Scheme),
		IOStreams:        streams,
	}
}

// AddFlags registers flags for a cli.
func (flags *WhoAmIFlags) AddFlags(cmd *cobra.Command) {
	flags.PrintFlags.AddFlags(cmd)
}

// ToOptions converts from CLI inputs to runtime inputs.
func (flags *WhoAmIFlags) ToOptions(ctx context.Context, args []string) (*WhoAmIOptions, error) {
	w := &WhoAmIOptions{
		ctx:       ctx,
		IOStreams: flags.IOStreams,
	}

	clientConfig, err := flags.RESTClientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}

	w.authV1alpha1Client, err = authenticationv1alpha1client.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	w.authV1beta1Client, err = authenticationv1beta1client.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	w.authV1Client, err = authenticationv1client.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	if !flags.PrintFlags.OutputFlagSpecified() {
		w.resourcePrinterFunc = printTableSelfSubjectAccessReview
	} else {
		printer, err := flags.PrintFlags.ToPrinter()
		if err != nil {
			return nil, err
		}
		w.resourcePrinterFunc = printer.PrintObj
	}

	return w, nil
}

// WhoAmIOptions is the start of the data required to perform the operation. As new fields are added,
// add them here instead of referencing the cmd.Flags()
type WhoAmIOptions struct {
	authV1alpha1Client authenticationv1alpha1client.AuthenticationV1alpha1Interface
	authV1beta1Client  authenticationv1beta1client.AuthenticationV1beta1Interface
	authV1Client       authenticationv1client.AuthenticationV1Interface

	ctx context.Context

	resourcePrinterFunc printers.ResourcePrinterFunc

	genericiooptions.IOStreams
}

var (
	whoAmILong = templates.LongDesc(`
		Experimental: Check who you are and your attributes (groups, extra).

        This command is helpful to get yourself aware of the current user attributes, 
        especially when dynamic authentication, e.g., token webhook, auth proxy, or OIDC provider, 
        is enabled in the Kubernetes cluster.
	`)

	whoAmIExample = templates.Examples(`
		# Get your subject attributes.
		kubectl auth whoami
		
		# Get your subject attributes in JSON format.
		kubectl auth whoami -o json
	`)
)

// NewCmdWhoAmI returns an initialized Command for 'auth whoami' sub command. Experimental.
func NewCmdWhoAmI(restClientGetter genericclioptions.RESTClientGetter, streams genericiooptions.IOStreams) *cobra.Command {
	flags := NewWhoAmIFlags(restClientGetter, streams)

	cmd := &cobra.Command{
		Use:                   "whoami",
		DisableFlagsInUseLine: true,
		Short:                 "Experimental: Check self subject attributes",
		Long:                  whoAmILong,
		Example:               whoAmIExample,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(cmd.Context(), args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Run())
		},
	}

	flags.AddFlags(cmd)
	return cmd
}

var (
	notEnabledErr = fmt.Errorf(
		"the selfsubjectreviews API is not enabled in the cluster\n" +
			"enable APISelfSubjectReview feature gate and authentication.k8s.io/v1alpha1 or authentication.k8s.io/v1beta1 API")

	forbiddenErr = fmt.Errorf(
		"the selfsubjectreviews API is not enabled in the cluster or you do not have permission to call it")
)

// Run prints all user attributes.
func (o WhoAmIOptions) Run() error {
	var (
		res runtime.Object
		err error
	)

	res, err = o.authV1Client.
		SelfSubjectReviews().
		Create(context.TODO(), &authenticationv1.SelfSubjectReview{}, metav1.CreateOptions{})
	if err != nil && errors.IsNotFound(err) {
		// Fallback to Beta API if Beta is not enabled
		res, err = o.authV1beta1Client.
			SelfSubjectReviews().
			Create(context.TODO(), &authenticationv1beta1.SelfSubjectReview{}, metav1.CreateOptions{})
		if err != nil && errors.IsNotFound(err) {
			// Fallback to Alpha API if Beta is not enabled
			res, err = o.authV1alpha1Client.
				SelfSubjectReviews().
				Create(context.TODO(), &authenticationv1alpha1.SelfSubjectReview{}, metav1.CreateOptions{})
		}
	}
	if err != nil {
		switch {
		case errors.IsForbidden(err):
			return forbiddenErr
		case errors.IsNotFound(err):
			return notEnabledErr
		default:
			return err
		}
	}
	return o.resourcePrinterFunc(res, o.Out)
}

func getUserInfo(obj runtime.Object) (authenticationv1.UserInfo, error) {
	switch obj.(type) {
	case *authenticationv1alpha1.SelfSubjectReview:
		return obj.(*authenticationv1alpha1.SelfSubjectReview).Status.UserInfo, nil
	case *authenticationv1beta1.SelfSubjectReview:
		return obj.(*authenticationv1beta1.SelfSubjectReview).Status.UserInfo, nil
	case *authenticationv1.SelfSubjectReview:
		return obj.(*authenticationv1.SelfSubjectReview).Status.UserInfo, nil
	default:
		return authenticationv1.UserInfo{}, fmt.Errorf("unexpected response type %T, expected SelfSubjectReview", obj)
	}
}

func printTableSelfSubjectAccessReview(obj runtime.Object, out io.Writer) error {
	ui, err := getUserInfo(obj)
	if err != nil {
		return err
	}

	w := printers.GetNewTabWriter(out)
	defer w.Flush()

	_, err = fmt.Fprintf(w, "ATTRIBUTE\tVALUE\n")
	if err != nil {
		return fmt.Errorf("cannot write a header: %w", err)
	}

	if ui.Username != "" {
		_, err := fmt.Fprintf(w, "Username\t%s\n", ui.Username)
		if err != nil {
			return fmt.Errorf("cannot write a username: %w", err)
		}
	}

	if ui.UID != "" {
		_, err := fmt.Fprintf(w, "UID\t%s\n", ui.UID)
		if err != nil {
			return fmt.Errorf("cannot write a uid: %w", err)
		}
	}

	if len(ui.Groups) > 0 {
		_, err := fmt.Fprintf(w, "Groups\t%v\n", ui.Groups)
		if err != nil {
			return fmt.Errorf("cannot write groups: %w", err)
		}
	}

	if len(ui.Extra) > 0 {
		for _, k := range sets.StringKeySet(ui.Extra).List() {
			v := ui.Extra[k]
			_, err := fmt.Fprintf(w, "Extra: %s\t%v\n", k, v)
			if err != nil {
				return fmt.Errorf("cannot write an extra: %w", err)
			}
		}

	}
	return nil
}
