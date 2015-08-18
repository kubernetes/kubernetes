/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package self_signed

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/cloudflare/cfssl/cli/genkey"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/selfsign"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/secret"
	"k8s.io/kubernetes/pkg/secret/server_certificate"
)

func ProbeSecretPlugins() []secret.SecretPlugin {
	return []secret.SecretPlugin{&selfSignedCertificatePlugin{}}
}

type selfSignedCertificatePlugin struct {
}

func (plugin *selfSignedCertificatePlugin) Init() {
}

func (plugin *selfSignedCertificatePlugin) SecretType() api.SecretType {
	return server_certificate.SecretTypeServerCertificate
}

func (plugin *selfSignedCertificatePlugin) RevokeSecret(secret api.Secret) error {
	return nil
}

func (plugin *selfSignedCertificatePlugin) GenerateSecret(secret api.Secret) (*api.Secret, error) {
	if _, ok := secret.Data[server_certificate.CertificateDataKey]; ok {
		glog.V(5).Infof("Secret[%s] already has a certificate - skipping generation", secret.Name)
		return nil, nil
	}
	if _, ok := secret.Data[server_certificate.PrivateKeyDataKey]; ok {
		glog.V(5).Infof("Secret[%s] already has a private key - skipping generation", secret.Name)
		return nil, nil
	}

	hosts, ok := secret.ObjectMeta.Annotations[server_certificate.HostNamesKey]
	if !ok {
		return nil, fmt.Errorf("Secret[%s] has no hosts specified - skipping generation", secret.Name)
	}

	splitHosts := strings.Split(hosts, ",")
	req := csr.CertificateRequest{
		CN:    splitHosts[0],
		Hosts: splitHosts,
	}

	req.KeyRequest = &csr.KeyRequest{
		Algo: csr.DefaultKeyRequest.Algo,
		Size: csr.DefaultKeyRequest.Size,
	}
	algo, ok := secret.ObjectMeta.Annotations[server_certificate.PrivateKeyAlgoKey]
	if ok {
		req.KeyRequest.Algo = algo
	}
	if algo == "rsa" {
		req.KeyRequest.Size = 2048
	}
	size, ok := secret.ObjectMeta.Annotations[server_certificate.PrivateKeySizeKey]
	if ok {
		s, err := strconv.Atoi(size)
		if err != nil {
			return nil, fmt.Errorf("Secret[%s] has invalid key size specified (%s) - skipping generation", secret.Name, size)
		}
		req.KeyRequest.Size = s
	}

	var key, csrPEM []byte
	g := &csr.Generator{Validator: genkey.Validator}
	csrPEM, key, err := g.ProcessRequest(&req)
	if err != nil {
		return nil, err
	}

	priv, err := helpers.ParsePrivateKeyPEM(key)
	if err != nil {
		return nil, err
	}

	profile := config.DefaultConfig()

	cert, err := selfsign.Sign(priv, csrPEM, profile)
	if err != nil {
		return nil, err
	}

	if secret.Data == nil {
		secret.Data = map[string][]byte{}
	}

	secret.Data[server_certificate.CertificateDataKey] = cert
	secret.Data[server_certificate.PrivateKeyDataKey] = key

	return &secret, nil
}
