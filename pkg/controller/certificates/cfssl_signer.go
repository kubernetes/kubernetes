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
	"crypto"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"

	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
)

var onlySigningPolicy = &config.Signing{
	Default: &config.SigningProfile{
		Usage:        []string{"signing"},
		Expiry:       helpers.OneYear,
		ExpiryString: "8760h",
	},
}

type CFSSLSigner struct {
	ca      *x509.Certificate
	priv    crypto.Signer
	sigAlgo x509.SignatureAlgorithm
}

func NewCFSSLSigner(caFile, caKeyFile string) (*CFSSLSigner, error) {
	ca, err := ioutil.ReadFile(caFile)
	if err != nil {
		return nil, err
	}
	cakey, err := ioutil.ReadFile(caKeyFile)
	if err != nil {
		return nil, err
	}

	parsedCa, err := helpers.ParseCertificatePEM(ca)
	if err != nil {
		return nil, err
	}

	strPassword := os.Getenv("CFSSL_CA_PK_PASSWORD")
	password := []byte(strPassword)
	if strPassword == "" {
		password = nil
	}

	priv, err := helpers.ParsePrivateKeyPEMWithPassword(cakey, password)
	if err != nil {
		return nil, fmt.Errorf("Malformed private key %v", err)
	}
	return &CFSSLSigner{
		priv:    priv,
		ca:      parsedCa,
		sigAlgo: signer.DefaultSigAlgo(priv),
	}, nil
}

func (cs *CFSSLSigner) Sign(csr *certificates.CertificateSigningRequest) ([]byte, error) {
	var usages []string
	for _, usage := range csr.Spec.Usages {
		usages = append(usages, string(usage))
	}
	policy := &config.Signing{
		Default: &config.SigningProfile{
			Usage:        usages,
			Expiry:       helpers.OneYear,
			ExpiryString: "8760h",
		},
	}
	s, err := local.NewSigner(cs.priv, cs.ca, cs.sigAlgo, policy)
	if err != nil {
		return nil, err
	}
	return s.Sign(signer.SignRequest{
		Request: string(csr.Spec.Request),
	})
}
