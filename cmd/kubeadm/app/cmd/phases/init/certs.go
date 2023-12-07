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
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

var (
	saKeyLongDesc = fmt.Sprintf(cmdutil.LongDesc(`
		Generate the private key for signing service account tokens along with its public key, and save them into
		%s and %s files.
		
		If both files already exist, kubeadm skips the generation step and existing files will be used.
		`), kubeadmconstants.ServiceAccountPrivateKeyName, kubeadmconstants.ServiceAccountPublicKeyName)

	genericLongDesc = cmdutil.LongDesc(`
		Generate the %[1]s, and save them into %[2]s.crt and %[2]s.key files.%[3]s

		If both files already exist, kubeadm skips the generation step and existing files will be used.
		`)
)

// NewCertsPhase returns the phase for the certs
func NewCertsPhase() workflow.Phase {
	return workflow.Phase{
		Name:   "certs",
		Short:  "Certificate generation",
		Phases: newCertSubPhases(),
		Run:    runCerts,
		Long:   cmdutil.MacroCommandLongDescription,
	}
}

// newCertSubPhases returns sub phases for certs phase
func newCertSubPhases() []workflow.Phase {
	subPhases := []workflow.Phase{}

	// All subphase
	allPhase := workflow.Phase{
		Name:           "all",
		Short:          "Generate all certificates",
		InheritFlags:   getCertPhaseFlags("all"),
		RunAllSiblings: true,
	}

	subPhases = append(subPhases, allPhase)

	// This loop assumes that GetDefaultCertList() always returns a list of
	// certificate that is preceded by the CAs that sign them.
	var lastCACert *certsphase.KubeadmCert
	for _, cert := range certsphase.GetDefaultCertList() {
		var phase workflow.Phase
		if cert.CAName == "" {
			phase = newCertSubPhase(cert, runCAPhase(cert))
			lastCACert = cert
		} else {
			phase = newCertSubPhase(cert, runCertPhase(cert, lastCACert))
		}
		subPhases = append(subPhases, phase)
	}

	// SA creates the private/public key pair, which doesn't use x509 at all
	saPhase := workflow.Phase{
		Name:         "sa",
		Short:        "Generate a private key for signing service account tokens along with its public key",
		Long:         saKeyLongDesc,
		Run:          runCertsSa,
		InheritFlags: []string{options.CertificatesDir},
	}

	subPhases = append(subPhases, saPhase)

	return subPhases
}

func newCertSubPhase(certSpec *certsphase.KubeadmCert, run func(c workflow.RunData) error) workflow.Phase {
	phase := workflow.Phase{
		Name:  certSpec.Name,
		Short: fmt.Sprintf("Generate the %s", certSpec.LongName),
		Long: fmt.Sprintf(
			genericLongDesc,
			certSpec.LongName,
			certSpec.BaseName,
			getSANDescription(certSpec),
		),
		Run:          run,
		InheritFlags: getCertPhaseFlags(certSpec.Name),
	}
	return phase
}

func getCertPhaseFlags(name string) []string {
	flags := []string{
		options.CertificatesDir,
		options.CfgPath,
		options.KubernetesVersion,
		options.DryRun,
	}
	if name == "all" || name == "apiserver" {
		flags = append(flags,
			options.APIServerAdvertiseAddress,
			options.ControlPlaneEndpoint,
			options.APIServerCertSANs,
			options.NetworkingDNSDomain,
			options.NetworkingServiceSubnet,
		)
	}
	return flags
}

func getSANDescription(certSpec *certsphase.KubeadmCert) string {
	// Defaulted config we will use to get SAN certs
	defaultConfig := &kubeadmapiv1.InitConfiguration{
		LocalAPIEndpoint: kubeadmapiv1.APIEndpoint{
			// GetAPIServerAltNames errors without an AdvertiseAddress; this is as good as any.
			AdvertiseAddress: "127.0.0.1",
		},
	}

	defaultInternalConfig := &kubeadmapi.InitConfiguration{}

	kubeadmscheme.Scheme.Default(defaultConfig)
	if err := kubeadmscheme.Scheme.Convert(defaultConfig, defaultInternalConfig, nil); err != nil {
		return ""
	}

	certConfig, err := certSpec.GetConfig(defaultInternalConfig)
	if err != nil {
		return ""
	}

	if len(certConfig.AltNames.DNSNames) == 0 && len(certConfig.AltNames.IPs) == 0 {
		return ""
	}
	// This mutates the certConfig, but we're throwing it after we construct the command anyway
	sans := []string{}

	for _, dnsName := range certConfig.AltNames.DNSNames {
		if dnsName != "" {
			sans = append(sans, dnsName)
		}
	}

	for _, ip := range certConfig.AltNames.IPs {
		sans = append(sans, ip.String())
	}
	return fmt.Sprintf("\n\nDefault SANs are %s", strings.Join(sans, ", "))
}

func runCertsSa(c workflow.RunData) error {
	data, ok := c.(InitData)
	if !ok {
		return errors.New("certs phase invoked with an invalid data struct")
	}

	// if external CA mode, skip service account key generation
	if data.ExternalCA() {
		fmt.Printf("[certs] Using existing sa keys\n")
		return nil
	}

	// create the new service account key (or use existing)
	return certsphase.CreateServiceAccountKeyAndPublicKeyFiles(data.CertificateWriteDir(), data.Cfg().ClusterConfiguration.EncryptionAlgorithmType())
}

func runCerts(c workflow.RunData) error {
	data, ok := c.(InitData)
	if !ok {
		return errors.New("certs phase invoked with an invalid data struct")
	}

	fmt.Printf("[certs] Using certificateDir folder %q\n", data.CertificateWriteDir())
	return nil
}

func runCAPhase(ca *certsphase.KubeadmCert) func(c workflow.RunData) error {
	return func(c workflow.RunData) error {
		data, ok := c.(InitData)
		if !ok {
			return errors.New("certs phase invoked with an invalid data struct")
		}

		// if using external etcd, skips etcd certificate authority generation
		if data.Cfg().Etcd.External != nil && ca.Name == "etcd-ca" {
			fmt.Printf("[certs] External etcd mode: Skipping %s certificate authority generation\n", ca.BaseName)
			return nil
		}

		if cert, err := pkiutil.TryLoadCertFromDisk(data.CertificateDir(), ca.BaseName); err == nil {
			certsphase.CheckCertificatePeriodValidity(ca.BaseName, cert)

			// If CA Cert existed while dryrun, copy CA Cert to dryrun dir for later use
			if data.DryRun() {
				err := kubeadmutil.CopyFile(filepath.Join(data.CertificateDir(), kubeadmconstants.CACertName), filepath.Join(data.CertificateWriteDir(), kubeadmconstants.CACertName))
				if err != nil {
					return errors.Wrapf(err, "could not copy %s to dry run directory %s", kubeadmconstants.CACertName, data.CertificateWriteDir())
				}
			}
			if _, err := pkiutil.TryLoadKeyFromDisk(data.CertificateDir(), ca.BaseName); err == nil {
				// If CA Key existed while dryrun, copy CA Key to dryrun dir for later use
				if data.DryRun() {
					err := kubeadmutil.CopyFile(filepath.Join(data.CertificateDir(), kubeadmconstants.CAKeyName), filepath.Join(data.CertificateWriteDir(), kubeadmconstants.CAKeyName))
					if err != nil {
						return errors.Wrapf(err, "could not copy %s to dry run directory %s", kubeadmconstants.CAKeyName, data.CertificateWriteDir())
					}
				}
				fmt.Printf("[certs] Using existing %s certificate authority\n", ca.BaseName)
				return nil
			}
			fmt.Printf("[certs] Using existing %s keyless certificate authority\n", ca.BaseName)
			return nil
		}

		// if dryrunning, write certificates authority to a temporary folder (and defer restore to the path originally specified by the user)
		cfg := data.Cfg()
		cfg.CertificatesDir = data.CertificateWriteDir()
		defer func() { cfg.CertificatesDir = data.CertificateDir() }()

		// create the new certificate authority (or use existing)
		return certsphase.CreateCACertAndKeyFiles(ca, cfg)
	}
}

func runCertPhase(cert *certsphase.KubeadmCert, caCert *certsphase.KubeadmCert) func(c workflow.RunData) error {
	return func(c workflow.RunData) error {
		data, ok := c.(InitData)
		if !ok {
			return errors.New("certs phase invoked with an invalid data struct")
		}

		// if using external etcd, skips etcd certificates generation
		if data.Cfg().Etcd.External != nil && cert.CAName == "etcd-ca" {
			fmt.Printf("[certs] External etcd mode: Skipping %s certificate generation\n", cert.BaseName)
			return nil
		}

		if certData, intermediates, err := pkiutil.TryLoadCertChainFromDisk(data.CertificateDir(), cert.BaseName); err == nil {
			certsphase.CheckCertificatePeriodValidity(cert.BaseName, certData)

			caCertData, err := pkiutil.TryLoadCertFromDisk(data.CertificateDir(), caCert.BaseName)
			if err != nil {
				return errors.Wrapf(err, "couldn't load CA certificate %s", caCert.Name)
			}

			certsphase.CheckCertificatePeriodValidity(caCert.BaseName, caCertData)

			if err := pkiutil.VerifyCertChain(certData, intermediates, caCertData); err != nil {
				return errors.Wrapf(err, "[certs] certificate %s not signed by CA certificate %s", cert.BaseName, caCert.BaseName)
			}

			fmt.Printf("[certs] Using existing %s certificate and key on disk\n", cert.BaseName)
			return nil
		}

		// if dryrunning, write certificates to a temporary folder (and defer restore to the path originally specified by the user)
		cfg := data.Cfg()
		cfg.CertificatesDir = data.CertificateWriteDir()
		defer func() { cfg.CertificatesDir = data.CertificateDir() }()

		// create the new certificate (or use existing)
		return certsphase.CreateCertAndKeyFilesWithCA(cert, caCert, cfg)
	}
}
