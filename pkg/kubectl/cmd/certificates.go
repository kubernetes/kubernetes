/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/printers"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/resource"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"

	"github.com/spf13/cobra"
)

func NewCmdCertificate(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use: "certificate SUBCOMMAND",
		DisableFlagsInUseLine: true,
		Short: i18n.T("Modify certificate resources."),
		Long:  "Modify certificate resources.",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	cmd.AddCommand(NewCmdCertificateApprove(f, ioStreams))
	cmd.AddCommand(NewCmdCertificateDeny(f, ioStreams))

	return cmd
}

type CertificateOptions struct {
	resource.FilenameOptions

	PrintFlags *genericclioptions.PrintFlags
	PrintObj   printers.ResourcePrinterFunc

	csrNames    []string
	outputStyle string

	clientSet internalclientset.Interface
	builder   *resource.Builder

	genericclioptions.IOStreams
}

func (o *CertificateOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	o.csrNames = args
	o.outputStyle = cmdutil.GetFlagString(cmd, "output")

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = func(obj runtime.Object, out io.Writer) error {
		return printer.PrintObj(obj, out)
	}

	o.builder = f.NewBuilder()
	o.clientSet, err = f.ClientSet()
	if err != nil {
		return err
	}

	return nil
}

func (o *CertificateOptions) Validate() error {
	if len(o.csrNames) < 1 && cmdutil.IsFilenameSliceEmpty(o.Filenames) {
		return fmt.Errorf("one or more CSRs must be specified as <name> or -f <filename>")
	}
	return nil
}

func NewCmdCertificateApprove(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := CertificateOptions{
		PrintFlags: genericclioptions.NewPrintFlags("approved").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
	cmd := &cobra.Command{
		Use: "approve (-f FILENAME | NAME)",
		DisableFlagsInUseLine: true,
		Short: i18n.T("Approve a certificate signing request"),
		Long: templates.LongDesc(`
		Approve a certificate signing request.

		kubectl certificate approve allows a cluster admin to approve a certificate
		signing request (CSR). This action tells a certificate signing controller to
		issue a certificate to the requestor with the attributes requested in the CSR.

		SECURITY NOTICE: Depending on the requested attributes, the issued certificate
		can potentially grant a requester access to cluster resources or to authenticate
		as a requested identity. Before approving a CSR, ensure you understand what the
		signed certificate can do.
		`),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunCertificateApprove(cmdutil.GetFlagBool(cmd, "force")))
		},
	}

	options.PrintFlags.AddFlags(cmd)

	cmd.Flags().Bool("force", false, "Update the CSR even if it is already approved.")
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, "identifying the resource to update")

	return cmd
}

func (o *CertificateOptions) RunCertificateApprove(force bool) error {
	return o.modifyCertificateCondition(o.builder, o.clientSet, force, func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, bool) {
		var alreadyApproved bool
		for _, c := range csr.Status.Conditions {
			if c.Type == certificates.CertificateApproved {
				alreadyApproved = true
			}
		}
		if alreadyApproved {
			return csr, true
		}
		csr.Status.Conditions = append(csr.Status.Conditions, certificates.CertificateSigningRequestCondition{
			Type:           certificates.CertificateApproved,
			Reason:         "KubectlApprove",
			Message:        "This CSR was approved by kubectl certificate approve.",
			LastUpdateTime: metav1.Now(),
		})
		return csr, false
	})
}

func NewCmdCertificateDeny(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	options := CertificateOptions{
		PrintFlags: genericclioptions.NewPrintFlags("denied").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
	cmd := &cobra.Command{
		Use: "deny (-f FILENAME | NAME)",
		DisableFlagsInUseLine: true,
		Short: i18n.T("Deny a certificate signing request"),
		Long: templates.LongDesc(`
		Deny a certificate signing request.

		kubectl certificate deny allows a cluster admin to deny a certificate
		signing request (CSR). This action tells a certificate signing controller to
		not to issue a certificate to the requestor.
		`),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunCertificateDeny(cmdutil.GetFlagBool(cmd, "force")))
		},
	}

	options.PrintFlags.AddFlags(cmd)

	cmd.Flags().Bool("force", false, "Update the CSR even if it is already denied.")
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, "identifying the resource to update")

	return cmd
}

func (o *CertificateOptions) RunCertificateDeny(force bool) error {
	return o.modifyCertificateCondition(o.builder, o.clientSet, force, func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, bool) {
		var alreadyDenied bool
		for _, c := range csr.Status.Conditions {
			if c.Type == certificates.CertificateDenied {
				alreadyDenied = true
			}
		}
		if alreadyDenied {
			return csr, true
		}
		csr.Status.Conditions = append(csr.Status.Conditions, certificates.CertificateSigningRequestCondition{
			Type:           certificates.CertificateDenied,
			Reason:         "KubectlDeny",
			Message:        "This CSR was approved by kubectl certificate deny.",
			LastUpdateTime: metav1.Now(),
		})
		return csr, false
	})
}

func (options *CertificateOptions) modifyCertificateCondition(builder *resource.Builder, clientSet internalclientset.Interface, force bool, modify func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, bool)) error {
	var found int
	r := builder.
		WithScheme(legacyscheme.Scheme).
		ContinueOnError().
		FilenameParam(false, &options.FilenameOptions).
		ResourceNames("certificatesigningrequest", options.csrNames...).
		RequireObject(true).
		Flatten().
		Latest().
		Do()
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		for i := 0; ; i++ {
			csr := info.Object.(*certificates.CertificateSigningRequest)
			csr, hasCondition := modify(csr)
			if !hasCondition || force {
				csr, err = clientSet.Certificates().
					CertificateSigningRequests().
					UpdateApproval(csr)
				if errors.IsConflict(err) && i < 10 {
					if err := info.Get(); err != nil {
						return err
					}
					continue
				}
				if err != nil {
					return err
				}
			}
			break
		}
		found++

		return options.PrintObj(cmdutil.AsDefaultVersionedOrOriginal(info.Object, info.Mapping), options.Out)
	})
	if found == 0 {
		fmt.Fprintf(options.Out, "No resources found\n")
	}
	return err
}
