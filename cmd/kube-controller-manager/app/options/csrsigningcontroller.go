/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"fmt"

	"github.com/spf13/pflag"

	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
)

// CSRSigningControllerOptions holds the CSRSigningController options.
type CSRSigningControllerOptions struct {
	*csrsigningconfig.CSRSigningControllerConfiguration
}

// AddFlags adds flags related to CSRSigningController for controller manager to the specified FlagSet.
func (o *CSRSigningControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.ClusterSigningCertFile, "cluster-signing-cert-file", o.ClusterSigningCertFile, "Filename containing a PEM-encoded X509 CA certificate used to issue cluster-scoped certificates.  If specified, no more specific --cluster-signing-* flag may be specified.")
	fs.StringVar(&o.ClusterSigningKeyFile, "cluster-signing-key-file", o.ClusterSigningKeyFile, "Filename containing a PEM-encoded RSA or ECDSA private key used to sign cluster-scoped certificates.  If specified, no more specific --cluster-signing-* flag may be specified.")
	fs.StringVar(&o.KubeletServingSignerConfiguration.CertFile, "cluster-signing-kubelet-serving-cert-file", o.KubeletServingSignerConfiguration.CertFile, "Filename containing a PEM-encoded X509 CA certificate used to issue certificates for the kubernetes.io/kubelet-serving signer.  If specified, --cluster-signing-{cert,key}-file must not be set.")
	fs.StringVar(&o.KubeletServingSignerConfiguration.KeyFile, "cluster-signing-kubelet-serving-key-file", o.KubeletServingSignerConfiguration.KeyFile, "Filename containing a PEM-encoded RSA or ECDSA private key used to sign certificates for the kubernetes.io/kubelet-serving signer.  If specified, --cluster-signing-{cert,key}-file must not be set.")
	fs.StringVar(&o.KubeletClientSignerConfiguration.CertFile, "cluster-signing-kubelet-client-cert-file", o.KubeletClientSignerConfiguration.CertFile, "Filename containing a PEM-encoded X509 CA certificate used to issue certificates for the kubernetes.io/kube-apiserver-client-kubelet signer.  If specified, --cluster-signing-{cert,key}-file must not be set.")
	fs.StringVar(&o.KubeletClientSignerConfiguration.KeyFile, "cluster-signing-kubelet-client-key-file", o.KubeletClientSignerConfiguration.KeyFile, "Filename containing a PEM-encoded RSA or ECDSA private key used to sign certificates for the kubernetes.io/kube-apiserver-client-kubelet signer.  If specified, --cluster-signing-{cert,key}-file must not be set.")
	fs.StringVar(&o.KubeAPIServerClientSignerConfiguration.CertFile, "cluster-signing-kube-apiserver-client-cert-file", o.KubeAPIServerClientSignerConfiguration.CertFile, "Filename containing a PEM-encoded X509 CA certificate used to issue certificates for the kubernetes.io/kube-apiserver-client signer.  If specified, --cluster-signing-{cert,key}-file must not be set.")
	fs.StringVar(&o.KubeAPIServerClientSignerConfiguration.KeyFile, "cluster-signing-kube-apiserver-client-key-file", o.KubeAPIServerClientSignerConfiguration.KeyFile, "Filename containing a PEM-encoded RSA or ECDSA private key used to sign certificates for the kubernetes.io/kube-apiserver-client signer.  If specified, --cluster-signing-{cert,key}-file must not be set.")
	fs.StringVar(&o.LegacyUnknownSignerConfiguration.CertFile, "cluster-signing-legacy-unknown-cert-file", o.LegacyUnknownSignerConfiguration.CertFile, "Filename containing a PEM-encoded X509 CA certificate used to issue certificates for the kubernetes.io/legacy-unknown signer.  If specified, --cluster-signing-{cert,key}-file must not be set.")
	fs.StringVar(&o.LegacyUnknownSignerConfiguration.KeyFile, "cluster-signing-legacy-unknown-key-file", o.LegacyUnknownSignerConfiguration.KeyFile, "Filename containing a PEM-encoded RSA or ECDSA private key used to sign certificates for the kubernetes.io/legacy-unknown signer.  If specified, --cluster-signing-{cert,key}-file must not be set.")
	fs.DurationVar(&o.ClusterSigningDuration.Duration, "cluster-signing-duration", o.ClusterSigningDuration.Duration, "The length of duration signed certificates will be given.")
	fs.DurationVar(&o.ClusterSigningDuration.Duration, "experimental-cluster-signing-duration", o.ClusterSigningDuration.Duration, "The length of duration signed certificates will be given.")
	fs.MarkDeprecated("experimental-cluster-signing-duration", "use --cluster-signing-duration")
}

// ApplyTo fills up CSRSigningController config with options.
func (o *CSRSigningControllerOptions) ApplyTo(cfg *csrsigningconfig.CSRSigningControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ClusterSigningCertFile = o.ClusterSigningCertFile
	cfg.ClusterSigningKeyFile = o.ClusterSigningKeyFile
	cfg.KubeletServingSignerConfiguration = o.KubeletServingSignerConfiguration
	cfg.KubeletClientSignerConfiguration = o.KubeletClientSignerConfiguration
	cfg.KubeAPIServerClientSignerConfiguration = o.KubeAPIServerClientSignerConfiguration
	cfg.LegacyUnknownSignerConfiguration = o.LegacyUnknownSignerConfiguration
	cfg.ClusterSigningDuration = o.ClusterSigningDuration

	return nil
}

// Validate checks validation of CSRSigningControllerOptions.
func (o *CSRSigningControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	if err := csrSigningFilesValid(o.KubeletServingSignerConfiguration); err != nil {
		errs = append(errs, fmt.Errorf("%q: %v", "cluster-signing-kubelet-serving", err))
	}
	if err := csrSigningFilesValid(o.KubeletClientSignerConfiguration); err != nil {
		errs = append(errs, fmt.Errorf("%q: %v", "cluster-signing-kube-apiserver-client", err))
	}
	if err := csrSigningFilesValid(o.KubeAPIServerClientSignerConfiguration); err != nil {
		errs = append(errs, fmt.Errorf("%q: %v", "cluster-signing-kube-apiserver", err))
	}
	if err := csrSigningFilesValid(o.LegacyUnknownSignerConfiguration); err != nil {
		errs = append(errs, fmt.Errorf("%q: %v", "cluster-signing-legacy-unknown", err))
	}

	singleSigningFile := len(o.ClusterSigningCertFile) > 0 || len(o.ClusterSigningKeyFile) > 0
	anySpecificFilesSet := len(o.KubeletServingSignerConfiguration.CertFile) > 0 || len(o.KubeletServingSignerConfiguration.KeyFile) > 0 ||
		len(o.KubeletClientSignerConfiguration.CertFile) > 0 || len(o.KubeletClientSignerConfiguration.KeyFile) > 0 ||
		len(o.KubeAPIServerClientSignerConfiguration.CertFile) > 0 || len(o.KubeAPIServerClientSignerConfiguration.KeyFile) > 0 ||
		len(o.LegacyUnknownSignerConfiguration.CertFile) > 0 || len(o.LegacyUnknownSignerConfiguration.KeyFile) > 0
	if singleSigningFile && anySpecificFilesSet {
		errs = append(errs, fmt.Errorf("cannot specify --cluster-signing-{cert,key}-file and other --cluster-signing-*-file flags at the same time"))
	}

	return errs
}

// both must be specified or both must be empty
func csrSigningFilesValid(config csrsigningconfig.CSRSigningConfiguration) error {
	switch {
	case (len(config.CertFile) == 0) && (len(config.KeyFile) == 0):
		return nil
	case (len(config.CertFile) != 0) && (len(config.KeyFile) != 0):
		return nil
	case (len(config.CertFile) == 0) && (len(config.KeyFile) != 0):
		return fmt.Errorf("cannot specify key without cert")
	case (len(config.CertFile) != 0) && (len(config.KeyFile) == 0):
		return fmt.Errorf("cannot specify cert without key")
	}

	return fmt.Errorf("math broke")
}
