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

package certificates

import (
	"context"
	"fmt"
	"io"

	"github.com/spf13/cobra"

	certificatesv1 "k8s.io/api/certificates/v1"
	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	v1 "k8s.io/client-go/kubernetes/typed/certificates/v1"
	"k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// NewCmdCertificate returns `certificate` Cobra command
func NewCmdCertificate(restClientGetter genericclioptions.RESTClientGetter, ioStreams genericclioptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "certificate SUBCOMMAND",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Modify certificate resources."),
		Long:                  i18n.T("Modify certificate resources."),
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	cmd.AddCommand(NewCmdCertificateApprove(restClientGetter, ioStreams))
	cmd.AddCommand(NewCmdCertificateDeny(restClientGetter, ioStreams))

	return cmd
}

// CertificateOptions declares the arguments accepted by the certificate command
type CertificateOptions struct {
	resource.FilenameOptions

	PrintFlags *genericclioptions.PrintFlags
	PrintObj   printers.ResourcePrinterFunc

	csrNames    []string
	outputStyle string

	certificatesV1Client      v1.CertificatesV1Interface
	certificatesV1Beta1Client v1beta1.CertificatesV1beta1Interface
	builder                   *resource.Builder

	genericclioptions.IOStreams
}

// NewCertificateOptions creates CertificateOptions struct for `certificate` command
func NewCertificateOptions(ioStreams genericclioptions.IOStreams, operation string) *CertificateOptions {
	return &CertificateOptions{
		PrintFlags: genericclioptions.NewPrintFlags(operation).WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// Complete loads data from the command environment
func (o *CertificateOptions) Complete(restClientGetter genericclioptions.RESTClientGetter, cmd *cobra.Command, args []string) error {
	o.csrNames = args
	o.outputStyle = cmdutil.GetFlagString(cmd, "output")

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = func(obj runtime.Object, out io.Writer) error {
		return printer.PrintObj(obj, out)
	}

	o.builder = resource.NewBuilder(restClientGetter)

	clientConfig, err := restClientGetter.ToRESTConfig()
	if err != nil {
		return err
	}

	o.certificatesV1Client, err = v1.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	o.certificatesV1Beta1Client, err = v1beta1.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	return nil
}

// Validate checks if the provided `certificate` arguments are valid
func (o *CertificateOptions) Validate() error {
	if len(o.csrNames) < 1 && cmdutil.IsFilenameSliceEmpty(o.Filenames, o.Kustomize) {
		return fmt.Errorf("one or more CSRs must be specified as <name> or -f <filename>")
	}
	return nil
}

// NewCmdCertificateApprove returns the `certificate approve` Cobra command
func NewCmdCertificateApprove(restClientGetter genericclioptions.RESTClientGetter, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCertificateOptions(ioStreams, "approved")

	cmd := &cobra.Command{
		Use:                   "approve (-f FILENAME | NAME)",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Approve a certificate signing request"),
		Long: templates.LongDesc(i18n.T(`
		Approve a certificate signing request.

		kubectl certificate approve allows a cluster admin to approve a certificate
		signing request (CSR). This action tells a certificate signing controller to
		issue a certificate to the requestor with the attributes requested in the CSR.

		SECURITY NOTICE: Depending on the requested attributes, the issued certificate
		can potentially grant a requester access to cluster resources or to authenticate
		as a requested identity. Before approving a CSR, ensure you understand what the
		signed certificate can do.
		`)),
		Example: templates.Examples(i18n.T(`
			# Approve CSR 'csr-sqgzp'
			kubectl certificate approve csr-sqgzp
		`)),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(restClientGetter, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunCertificateApprove(cmdutil.GetFlagBool(cmd, "force")))
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().Bool("force", false, "Update the CSR even if it is already approved.")
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, "identifying the resource to update")

	return cmd
}

// RunCertificateApprove approves a certificate signing request
func (o *CertificateOptions) RunCertificateApprove(force bool) error {
	return o.modifyCertificateCondition(
		o.builder,
		force,
		addConditionIfNeeded(string(certificatesv1.CertificateDenied), string(certificatesv1.CertificateApproved), "KubectlApprove", "This CSR was approved by kubectl certificate approve."),
	)
}

// NewCmdCertificateDeny returns the `certificate deny` Cobra command
func NewCmdCertificateDeny(restClientGetter genericclioptions.RESTClientGetter, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCertificateOptions(ioStreams, "denied")

	cmd := &cobra.Command{
		Use:                   "deny (-f FILENAME | NAME)",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Deny a certificate signing request"),
		Long: templates.LongDesc(i18n.T(`
		Deny a certificate signing request.

		kubectl certificate deny allows a cluster admin to deny a certificate
		signing request (CSR). This action tells a certificate signing controller to
		not to issue a certificate to the requestor.
		`)),
		Example: templates.Examples(i18n.T(`
			# Deny CSR 'csr-sqgzp'
			kubectl certificate deny csr-sqgzp
		`)),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(restClientGetter, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunCertificateDeny(cmdutil.GetFlagBool(cmd, "force")))
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmd.Flags().Bool("force", false, "Update the CSR even if it is already denied.")
	cmdutil.AddFilenameOptionFlags(cmd, &o.FilenameOptions, "identifying the resource to update")

	return cmd
}

// RunCertificateDeny denies a certificate signing request
func (o *CertificateOptions) RunCertificateDeny(force bool) error {
	return o.modifyCertificateCondition(
		o.builder,
		force,
		addConditionIfNeeded(string(certificatesv1.CertificateApproved), string(certificatesv1.CertificateDenied), "KubectlDeny", "This CSR was denied by kubectl certificate deny."),
	)
}

func (o *CertificateOptions) modifyCertificateCondition(builder *resource.Builder, force bool, modify func(csr runtime.Object) (runtime.Object, bool, error)) error {
	var found int
	r := builder.
		Unstructured().
		ContinueOnError().
		FilenameParam(false, &o.FilenameOptions).
		ResourceNames("certificatesigningrequests", o.csrNames...).
		RequireObject(true).
		Flatten().
		Latest().
		Do()
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		for i := 0; ; i++ {
			obj, ok := info.Object.(*unstructured.Unstructured)
			if !ok {
				return fmt.Errorf("expected *unstructured.Unstructured, got %T", obj)
			}
			if want, got := certificatesv1.Kind("CertificateSigningRequest"), obj.GetObjectKind().GroupVersionKind().GroupKind(); want != got {
				return fmt.Errorf("can only handle %s objects, got %s", want.String(), got.String())
			}
			var csr runtime.Object
			// get a typed object
			// first try v1
			csr, err = o.certificatesV1Client.CertificateSigningRequests().Get(context.TODO(), obj.GetName(), metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				// fall back to v1beta1
				csr, err = o.certificatesV1Beta1Client.CertificateSigningRequests().Get(context.TODO(), obj.GetName(), metav1.GetOptions{})
			}
			if apierrors.IsNotFound(err) {
				return fmt.Errorf("could not find v1 or v1beta1 version of %s: %v", obj.GetName(), err)
			}
			if err != nil {
				return err
			}

			modifiedCSR, hasCondition, err := modify(csr)
			if err != nil {
				return err
			}
			if !hasCondition || force {
				switch modifiedCSR := modifiedCSR.(type) {
				case *certificatesv1.CertificateSigningRequest:
					_, err = o.certificatesV1Client.CertificateSigningRequests().UpdateApproval(context.TODO(), modifiedCSR.Name, modifiedCSR, metav1.UpdateOptions{})
				case *certificatesv1beta1.CertificateSigningRequest:
					_, err = o.certificatesV1Beta1Client.CertificateSigningRequests().UpdateApproval(context.TODO(), modifiedCSR, metav1.UpdateOptions{})
				default:
					return fmt.Errorf("can only handle certificates.k8s.io CertificateSigningRequest objects, got %T", modifiedCSR)
				}
				if apierrors.IsConflict(err) && i < 10 {
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

		return o.PrintObj(info.Object, o.Out)
	})
	if found == 0 && err == nil {
		fmt.Fprintf(o.Out, "No resources found\n")
	}
	return err
}

func addConditionIfNeeded(mustNotHaveConditionType, conditionType, reason, message string) func(runtime.Object) (runtime.Object, bool, error) {
	return func(csr runtime.Object) (runtime.Object, bool, error) {
		switch csr := csr.(type) {
		case *certificatesv1.CertificateSigningRequest:
			var alreadyHasCondition bool
			for _, c := range csr.Status.Conditions {
				if string(c.Type) == mustNotHaveConditionType {
					return nil, false, fmt.Errorf("certificate signing request %q is already %s", csr.Name, c.Type)
				}
				if string(c.Type) == conditionType {
					alreadyHasCondition = true
				}
			}
			if alreadyHasCondition {
				return csr, true, nil
			}
			csr.Status.Conditions = append(csr.Status.Conditions, certificatesv1.CertificateSigningRequestCondition{
				Type:           certificatesv1.RequestConditionType(conditionType),
				Status:         corev1.ConditionTrue,
				Reason:         reason,
				Message:        message,
				LastUpdateTime: metav1.Now(),
			})
			return csr, false, nil

		case *certificatesv1beta1.CertificateSigningRequest:
			var alreadyHasCondition bool
			for _, c := range csr.Status.Conditions {
				if string(c.Type) == mustNotHaveConditionType {
					return nil, false, fmt.Errorf("certificate signing request %q is already %s", csr.Name, c.Type)
				}
				if string(c.Type) == conditionType {
					alreadyHasCondition = true
				}
			}
			if alreadyHasCondition {
				return csr, true, nil
			}
			csr.Status.Conditions = append(csr.Status.Conditions, certificatesv1beta1.CertificateSigningRequestCondition{
				Type:           certificatesv1beta1.RequestConditionType(conditionType),
				Status:         corev1.ConditionTrue,
				Reason:         reason,
				Message:        message,
				LastUpdateTime: metav1.Now(),
			})
			return csr, false, nil

		default:
			return csr, false, nil
		}
	}
}
