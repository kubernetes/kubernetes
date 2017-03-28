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
	"io"

	"github.com/spf13/cobra"

	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func NewCmdKubeConfig(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubeconfig",
		Short: "Create KubeConfig files from given credentials.",
		RunE:  subCmdRunE("kubeconfig"),
	}

	cmd.AddCommand(NewCmdToken(out))
	cmd.AddCommand(NewCmdClientCerts(out))
	return cmd
}

func NewCmdToken(out io.Writer) *cobra.Command {
	config := &kubeconfigphase.BuildConfigProperties{
		MakeClientCerts: false,
	}
	cmd := &cobra.Command{
		Use:   "token",
		Short: "Output a valid KubeConfig file to STDOUT with a token as the authentication method.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunCreateWithToken(out, config)
			kubeadmutil.CheckErr(err)
		},
	}
	addCommonFlags(cmd, config)
	cmd.Flags().StringVar(&config.Token, "token", "", "The path to the directory where the certificates are.")
	return cmd
}

func NewCmdClientCerts(out io.Writer) *cobra.Command {
	config := &kubeconfigphase.BuildConfigProperties{
		MakeClientCerts: true,
	}
	cmd := &cobra.Command{
		Use:   "client-certs",
		Short: "Output a valid KubeConfig file to STDOUT with a client certificates as the authentication method.",
		Run: func(cmd *cobra.Command, args []string) {
			err := RunCreateWithClientCerts(out, config)
			kubeadmutil.CheckErr(err)
		},
	}
	addCommonFlags(cmd, config)
	cmd.Flags().StringSliceVar(&config.Organization, "organization", []string{}, "The organization (group) the certificate should be in.")
	return cmd
}

func addCommonFlags(cmd *cobra.Command, config *kubeconfigphase.BuildConfigProperties) {
	cmd.Flags().StringVar(&config.CertDir, "cert-dir", kubeadmapiext.DefaultCertificatesDir, "The path to the directory where the certificates are.")
	cmd.Flags().StringVar(&config.ClientName, "client-name", "", "The name of the client for which the KubeConfig file will be generated.")
	cmd.Flags().StringVar(&config.APIServer, "server", "", "The location of the api server.")
}

func validateCommonFlags(config *kubeconfigphase.BuildConfigProperties) error {
	if len(config.ClientName) == 0 {
		return fmt.Errorf("The --client-name flag is required")
	}
	if len(config.APIServer) == 0 {
		return fmt.Errorf("The --server flag is required")
	}
	return nil
}

// RunCreateWithToken generates a kubeconfig file from with a token as the authentication mechanism
func RunCreateWithToken(out io.Writer, config *kubeconfigphase.BuildConfigProperties) error {
	if len(config.Token) == 0 {
		return fmt.Errorf("The --token flag is required")
	}
	if err := validateCommonFlags(config); err != nil {
		return err
	}
	kubeConfigBytes, err := kubeconfigphase.GetKubeConfigBytesFromSpec(*config)
	if err != nil {
		return err
	}
	fmt.Fprintln(out, string(kubeConfigBytes))
	return nil
}

// RunCreateWithClientCerts generates a kubeconfig file from with client certs as the authentication mechanism
func RunCreateWithClientCerts(out io.Writer, config *kubeconfigphase.BuildConfigProperties) error {
	if err := validateCommonFlags(config); err != nil {
		return err
	}
	kubeConfigBytes, err := kubeconfigphase.GetKubeConfigBytesFromSpec(*config)
	if err != nil {
		return err
	}
	fmt.Fprintln(out, string(kubeConfigBytes))
	return nil
}
