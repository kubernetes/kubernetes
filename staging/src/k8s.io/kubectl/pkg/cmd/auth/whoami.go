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
	authenticationv1alpha1 "k8s.io/api/authentication/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	authenticationv1alpha1client "k8s.io/client-go/kubernetes/typed/authentication/v1alpha1"
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

	genericclioptions.IOStreams
}

// NewWhoAmIFlags returns a default WhoAmIFlags.
func NewWhoAmIFlags(restClientGetter genericclioptions.RESTClientGetter, streams genericclioptions.IOStreams) *WhoAmIFlags {
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

	w.authClient, err = authenticationv1alpha1client.NewForConfig(clientConfig)
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
	authClient authenticationv1alpha1client.AuthenticationV1alpha1Interface
	ctx        context.Context

	resourcePrinterFunc printers.ResourcePrinterFunc

	genericclioptions.IOStreams
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
		kubectl alpha auth whoami
		
		# Get your subject attributes in JSON format.
		kubectl alpha auth whoami -o json
	`)
)

// NewCmdWhoAmI returns an initialized Command for 'auth whoami' sub command. Experimental.
func NewCmdWhoAmI(restClientGetter genericclioptions.RESTClientGetter, streams genericclioptions.IOStreams) *cobra.Command {
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

var notEnabledErr = fmt.Errorf(
	"the selfsubjectreviews API is not enabled in the cluster or you do not have permission to call\n" +
		"enable APISelfSubjectReview feature gate and authentication.k8s.io/v1alpha1 API")

// Run prints all user attributes.
func (o WhoAmIOptions) Run() error {
	sar := &authenticationv1alpha1.SelfSubjectReview{}
	response, err := o.authClient.SelfSubjectReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
	if err != nil {
		if errors.IsForbidden(err) || errors.IsNotFound(err) {
			return notEnabledErr
		}
		return err
	}
	return o.resourcePrinterFunc(response, o.Out)
}

func printTableSelfSubjectAccessReview(obj runtime.Object, out io.Writer) error {
	ssr, ok := obj.(*authenticationv1alpha1.SelfSubjectReview)
	if !ok {
		return fmt.Errorf("object is not SelfSubjectReview")
	}

	w := printers.GetNewTabWriter(out)
	defer w.Flush()

	_, err := fmt.Fprintf(w, "ATTRIBUTE\tVALUE\n")
	if err != nil {
		return fmt.Errorf("cannot write a header: %w", err)
	}

	ui := ssr.Status.UserInfo

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
