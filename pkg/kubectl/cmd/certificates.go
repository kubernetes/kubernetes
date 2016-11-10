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

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"

	"github.com/spf13/cobra"
)

func NewCmdCertificate(f cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "certificate SUBCOMMAND",
		Short: "Modify certificate resources.",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	cmd.AddCommand(NewCmdCertificateApprove(f, out))
	cmd.AddCommand(NewCmdCertificateDeny(f, out))

	return cmd
}

type CertificateOptions struct {
	resource.FilenameOptions
	csrNames    []string
	outputStyle string
}

func (options *CertificateOptions) Complete(cmd *cobra.Command, args []string) error {
	options.csrNames = args
	options.outputStyle = cmdutil.GetFlagString(cmd, "output")
	return nil
}

func (options *CertificateOptions) Validate() error {
	if len(options.csrNames) < 1 && cmdutil.IsFilenameEmpty(options.Filenames) {
		return fmt.Errorf("one or more CSRs must be specified as <name> or -f <filename>")
	}
	return nil
}

func NewCmdCertificateApprove(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := CertificateOptions{}
	cmd := &cobra.Command{
		Use:   "approve (-f FILENAME | NAME)",
		Short: "Approve a certificate signing request",
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
			cmdutil.CheckErr(options.Complete(cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunCertificateApprove(f, out))
		},
	}
	cmdutil.AddOutputFlagsForMutation(cmd)
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, "identifying the resource to update")

	return cmd
}

func (options *CertificateOptions) RunCertificateApprove(f cmdutil.Factory, out io.Writer) error {
	return options.modifyCertificateCondition(f, out, func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, string) {
		var alreadyApproved bool
		for _, c := range csr.Status.Conditions {
			if c.Type == certificates.CertificateApproved {
				alreadyApproved = true
			}
		}
		if alreadyApproved {
			return csr, "approved"
		}
		csr.Status.Conditions = append(csr.Status.Conditions, certificates.CertificateSigningRequestCondition{
			Type:           certificates.CertificateApproved,
			Reason:         "KubectlApprove",
			Message:        "This CSR was approved by kubectl certificate approve.",
			LastUpdateTime: unversioned.Now(),
		})
		return csr, "approved"
	})
}

func NewCmdCertificateDeny(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := CertificateOptions{}
	cmd := &cobra.Command{
		Use:   "deny (-f FILENAME | NAME)",
		Short: "Deny a certificate signing request",
		Long: templates.LongDesc(`
		Deny a certificate signing request.

		kubectl certificate deny allows a cluster admin to deny a certificate
		signing request (CSR). This action tells a certificate signing controller to
		not to issue a certificate to the requestor.
		`),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(cmd, args))
			cmdutil.CheckErr(options.Validate())
			cmdutil.CheckErr(options.RunCertificateDeny(f, out))
		},
	}
	cmdutil.AddOutputFlagsForMutation(cmd)
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, "identifying the resource to update")

	return cmd
}

func (options *CertificateOptions) RunCertificateDeny(f cmdutil.Factory, out io.Writer) error {
	return options.modifyCertificateCondition(f, out, func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, string) {
		var alreadyDenied bool
		for _, c := range csr.Status.Conditions {
			if c.Type == certificates.CertificateDenied {
				alreadyDenied = true
			}
		}
		if alreadyDenied {
			return csr, "denied"
		}
		csr.Status.Conditions = append(csr.Status.Conditions, certificates.CertificateSigningRequestCondition{
			Type:           certificates.CertificateDenied,
			Reason:         "KubectlDeny",
			Message:        "This CSR was approved by kubectl certificate deny.",
			LastUpdateTime: unversioned.Now(),
		})
		return csr, "denied"
	})
}

func (options *CertificateOptions) modifyCertificateCondition(f cmdutil.Factory, out io.Writer, modify func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, string)) error {
	var found int
	mapper, typer := f.Object()
	c, err := f.ClientSet()
	if err != nil {
		return err
	}
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		FilenameParam(false, &options.FilenameOptions).
		ResourceNames("certificatesigningrequest", options.csrNames...).
		RequireObject(true).
		Flatten().
		Latest().
		Do()
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		csr := info.Object.(*certificates.CertificateSigningRequest)
		csr, verb := modify(csr)
		csr, err = c.Certificates().
			CertificateSigningRequests().
			UpdateApproval(csr)
		if err != nil {
			return err
		}
		found++
		cmdutil.PrintSuccess(mapper, options.outputStyle == "name", out, info.Mapping.Resource, info.Name, false, verb)
		return nil
	})
	if found == 0 {
		fmt.Fprintf(out, "No resources found\n")
	}
	return err
}
