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
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"

	"github.com/spf13/cobra"
)

func NewCmdCertificate(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "certificate SUBCOMMAND",
		Short: "Modify certificate resources.",
		Long: `
`,
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	cmd.AddCommand(NewCmdCertificateApprove(f, out))
	cmd.AddCommand(NewCmdCertificateDeny(f, out))

	return cmd
}

type CertificateOptions struct {
	Recursive bool
	Filenames []string
}

func NewCmdCertificateApprove(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := CertificateOptions{}
	cmd := &cobra.Command{
		Use:   "approve SUBCOMMAND",
		Short: "approve a CSR.",
		Long: `
`,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(RunCertificateApprove(f, out, cmd, args, &options))
		},
	}
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, "filenames of certificatesigningrequest objects to approve")
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	cmdutil.AddOutputFlagsForMutation(cmd)

	return cmd
}

func RunCertificateApprove(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *CertificateOptions) error {
	return modifyCertificateCondition(f, out, cmd, args, options, func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, string) {
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
			Message:        "This csr was approved by kubectl alpha approve",
			LastUpdateTime: unversioned.Now(),
		})
		return csr, "approved"
	})
}

func NewCmdCertificateDeny(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := CertificateOptions{}
	cmd := &cobra.Command{
		Use:   "deny SUBCOMMAND",
		Short: "deny a CSR.",
		Long:  ``,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(RunCertificateDeny(f, out, cmd, args, &options))
		},
	}
	kubectl.AddJsonFilenameFlag(cmd, &options.Filenames, "filenames of certificatesigningrequest objects to deny")
	cmdutil.AddRecursiveFlag(cmd, &options.Recursive)
	cmdutil.AddOutputFlagsForMutation(cmd)

	return cmd
}

func RunCertificateDeny(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *CertificateOptions) error {
	return modifyCertificateCondition(f, out, cmd, args, options, func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, string) {
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
			Message:        "This csr was approved by kubectl alpha certificate deny",
			LastUpdateTime: unversioned.Now(),
		})
		return csr, "denied"
	})
}

func modifyCertificateCondition(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *CertificateOptions, modify func(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, string)) error {
	var found int
	mapper, typer := f.Object(false)
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		FilenameParam(false, options.Recursive, options.Filenames...).
		ResourceNames("certificatesigningrequest", args...).
		RequireObject(true).
		Flatten().
		Latest().
		Do()
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		csr := info.Object.(*certificates.CertificateSigningRequest)
		csr, verb := modify(csr)
		err = info.Client.Put().
			Resource("certificatesigningrequests").
			Name(csr.Name).
			Body(csr).
			SubResource("approval").
			Do().
			Error()
		if err != nil {
			return err
		}
		found++
		cmdutil.PrintSuccess(mapper, cmdutil.GetFlagString(cmd, "output") == "name", out, info.Mapping.Resource, info.Name, verb)
		return nil
	})
	if found == 0 {
		fmt.Fprintf(out, "No resources found\n")
	}
	return err
}
