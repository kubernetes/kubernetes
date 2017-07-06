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
	"net"

	"github.com/spf13/cobra"

	netutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/validation/field"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	certphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
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
	// TODO: Move this into a dedicated Certificates Phase API object
	cfg := &kubeadmapiext.MasterConfiguration{}
	// Default values for the cobra help text
	api.Scheme.Default(cfg)

	cmd := &cobra.Command{
		Use:   "selfsign",
		Short: "Generate the CA, APIServer signing/client cert, the ServiceAccount public/private keys and a CA and client cert for the front proxy",
		Run: func(cmd *cobra.Command, args []string) {

			// Run the defaulting once again to take passed flags into account
			api.Scheme.Default(cfg)
			internalcfg := &kubeadmapi.MasterConfiguration{}
			api.Scheme.Convert(cfg, internalcfg, nil)

			err := RunSelfSign(internalcfg)
			kubeadmutil.CheckErr(err)
		},
	}
	cmd.Flags().StringVar(&cfg.Networking.DNSDomain, "dns-domain", cfg.Networking.DNSDomain, "The DNS Domain for the Kubernetes cluster.")
	cmd.Flags().StringVar(&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir, "The path where to save and store the certificates.")
	cmd.Flags().StringVar(&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet, "The subnet for the Services in the cluster.")
	cmd.Flags().StringSliceVar(&cfg.APIServerCertSANs, "cert-altnames", []string{}, "Optional extra altnames to use for the API Server serving cert. Can be both IP addresses and dns names.")
	cmd.Flags().StringVar(&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress, "The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.")

	return cmd
}

// RunSelfSign generates certificate assets in the specified directory
func RunSelfSign(config *kubeadmapi.MasterConfiguration) error {
	if err := validateArgs(config); err != nil {
		return fmt.Errorf("The argument validation failed: %v", err)
	}

	// If it's possible to detect the default IP, add it to the SANs as well. Otherwise, just go with the provided ones
	ip, err := netutil.ChooseBindAddress(net.ParseIP(config.API.AdvertiseAddress))
	if err == nil {
		config.API.AdvertiseAddress = ip.String()
	}

	if err = certphase.CreatePKIAssets(config); err != nil {
		return err
	}
	return nil
}

func validateArgs(config *kubeadmapi.MasterConfiguration) error {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validation.ValidateNetworking(&config.Networking, field.NewPath("networking"))...)
	allErrs = append(allErrs, validation.ValidateAbsolutePath(config.CertificatesDir, field.NewPath("cert-dir"))...)
	allErrs = append(allErrs, validation.ValidateAPIServerCertSANs(config.APIServerCertSANs, field.NewPath("cert-altnames"))...)
	allErrs = append(allErrs, validation.ValidateIPFromString(config.API.AdvertiseAddress, field.NewPath("apiserver-advertise-address"))...)

	return allErrs.ToAggregate()
}
