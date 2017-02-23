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

package phases

import (
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/validation/field"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func NewCmdCerts() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "certs",
		Aliases: []string{"certificates"},
		Short:   "Generate certificates for a Kubernetes cluster.",
		RunE:    subCmdRunE("certs"),
	}

	cmd.AddCommand(NewCmdSelfSign())
	return cmd
}

func NewCmdSelfSign() *cobra.Command {
	// TODO: Move this into a Phase API object
	config := &kubeadmapi.MasterConfiguration{}
	cmd := &cobra.Command{
		Use:   "selfsign",
		Short: "Generate the CA, APIServer signing/client cert, the ServiceAccount public/private keys and a CA and client cert for the front proxy",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunSelfSign(config)
			kubeadmutil.CheckErr(err)
		},
	}
	cmd.Flags().StringVar(&config.Networking.DNSDomain, "dns-domain", kubeadmapiext.DefaultServiceDNSDomain, "The DNS Domain for the Kubernetes cluster.")
	cmd.Flags().StringVar(&config.CertificatesDir, "cert-dir", kubeadmapiext.DefaultCertificatesDir, "The path where to save and store the certificates")
	cmd.Flags().StringVar(&config.Networking.ServiceSubnet, "service-cidr", kubeadmapiext.DefaultServicesSubnet, "The subnet for the Services in the cluster.")
	cmd.Flags().StringSliceVar(&config.CertAltNames, "cert-altnames", []string{}, "Optional extra altnames to use for the API Server serving cert. Can be both IP addresses and dns names.")
	return cmd
}

// RunSelfSign generates certificate assets
func RunSelfSign(config *kubeadmapi.MasterConfiguration) error {
	if err := validateArgs(config); err != nil {
		return fmt.Errorf("validation failed: %v", err)
	}
	err := certphase.CreatePKIAssets(config)
	if err != nil {
		return err
	}
	return nil
}

func validateArgs(config *kubeadmapi.MasterConfiguration) error {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateNetworking(&config.Networking, field.NewPath("networking"))...)
	allErrs = append(allErrs, validation.ValidateAbsolutePath(config.CertificatesDir, field.NewPath("cert-dir"))...)
	allErrs = append(allErrs, validation.ValidateCertAltNames(config.CertAltNames, field.NewPath("cert-altnames"))...)
	return allErrs.ToAggregate()
}
