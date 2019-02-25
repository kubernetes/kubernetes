/*
Copyright 2019 The Kubernetes Authors.

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

package uploadcerts

import (
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	bootstraputil "k8s.io/cluster-bootstrap/token/util"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	nodebootstraptokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	cryptoutil "k8s.io/kubernetes/cmd/kubeadm/app/util/crypto"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
)

const (
	externalEtcdCA   = "external-etcd-ca.crt"
	externalEtcdCert = "external-etcd.crt"
	externalEtcdKey  = "external-etcd.key"
)

// createShortLivedBootstrapToken creates the token used to manager kubeadm-certs
// and return the tokenID
func createShortLivedBootstrapToken(client clientset.Interface) (string, error) {
	tokenStr, err := bootstraputil.GenerateBootstrapToken()
	if err != nil {
		return "", errors.Wrap(err, "error generating token to upload certs")
	}
	token, err := kubeadmapi.NewBootstrapTokenString(tokenStr)
	if err != nil {
		return "", errors.Wrap(err, "error creating upload certs token")
	}
	tokens := []kubeadmapi.BootstrapToken{{
		Token:       token,
		Description: "Proxy for managing TTL for the kubeadm-certs secret",
		TTL: &metav1.Duration{
			Duration: kubeadmconstants.DefaultCertTokenDuration,
		},
	}}

	if err := nodebootstraptokenphase.CreateNewTokens(client, tokens); err != nil {
		return "", errors.Wrap(err, "error creating token")
	}
	return tokens[0].Token.ID, nil
}

//CreateCertificateKey returns a cryptographically secure random key
func CreateCertificateKey() (string, error) {
	randBytes, err := cryptoutil.CreateRandBytes(kubeadmconstants.CertificateKeySize)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(randBytes), nil
}

//UploadCerts save certs needs to join a new control-plane on kubeadm-certs sercret.
func UploadCerts(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, key string) error {
	fmt.Printf("[upload-certs] storing the certificates in ConfigMap %q in the %q Namespace\n", kubeadmconstants.KubeadmCertsSecret, metav1.NamespaceSystem)
	decodedKey, err := hex.DecodeString(key)
	if err != nil {
		return err
	}
	tokenID, err := createShortLivedBootstrapToken(client)
	if err != nil {
		return err
	}

	secretData, err := getSecretData(cfg, decodedKey)
	if err != nil {
		return err
	}
	ref, err := getSecretOwnerRef(client, tokenID)
	if err != nil {
		return err
	}

	err = apiclient.CreateOrUpdateSecret(client, &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            kubeadmconstants.KubeadmCertsSecret,
			Namespace:       metav1.NamespaceSystem,
			OwnerReferences: ref,
		},
		Data: secretData,
	})
	if err != nil {
		return err
	}

	return createRBAC(client)
}

func createRBAC(client clientset.Interface) error {
	err := apiclient.CreateOrUpdateRole(client, &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeadmCertsClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		Rules: []rbac.PolicyRule{
			rbachelper.NewRule("get").Groups("").Resources("secrets").Names(kubeadmconstants.KubeadmCertsSecret).RuleOrDie(),
		},
	})
	if err != nil {
		return err
	}

	return apiclient.CreateOrUpdateRoleBinding(client, &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeadmCertsClusterRoleName,
			Namespace: metav1.NamespaceSystem,
		},
		RoleRef: rbac.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     kubeadmconstants.KubeadmCertsClusterRoleName,
		},
		Subjects: []rbac.Subject{
			{
				Kind: rbac.GroupKind,
				Name: kubeadmconstants.NodeBootstrapTokenAuthGroup,
			},
		},
	})
}

func getSecretOwnerRef(client clientset.Interface, tokenID string) ([]metav1.OwnerReference, error) {
	secretName := bootstraputil.BootstrapTokenSecretName(tokenID)
	secret, err := client.CoreV1().Secrets(metav1.NamespaceSystem).Get(secretName, metav1.GetOptions{})
	if err != nil {
		return nil, errors.Wrap(err, "error to get token reference")
	}

	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Secret"}
	ref := metav1.NewControllerRef(secret, gvk)
	return []metav1.OwnerReference{*ref}, nil
}

func loadAndEncryptCert(certPath string, key []byte) ([]byte, error) {
	cert, err := ioutil.ReadFile(certPath)
	if err != nil {
		return nil, err
	}
	return cryptoutil.EncryptBytes(cert, key)
}

func certsToUpload(cfg *kubeadmapi.InitConfiguration) map[string]string {
	certsDir := cfg.CertificatesDir
	certs := map[string]string{
		kubeadmconstants.CACertName:                   path.Join(certsDir, kubeadmconstants.CACertName),
		kubeadmconstants.CAKeyName:                    path.Join(certsDir, kubeadmconstants.CAKeyName),
		kubeadmconstants.FrontProxyCACertName:         path.Join(certsDir, kubeadmconstants.FrontProxyCACertName),
		kubeadmconstants.FrontProxyCAKeyName:          path.Join(certsDir, kubeadmconstants.FrontProxyCAKeyName),
		kubeadmconstants.ServiceAccountPublicKeyName:  path.Join(certsDir, kubeadmconstants.ServiceAccountPublicKeyName),
		kubeadmconstants.ServiceAccountPrivateKeyName: path.Join(certsDir, kubeadmconstants.ServiceAccountPrivateKeyName),
	}

	if cfg.Etcd.External == nil {
		certs[kubeadmconstants.EtcdCACertName] = path.Join(certsDir, kubeadmconstants.EtcdCACertName)
		certs[kubeadmconstants.EtcdCAKeyName] = path.Join(certsDir, kubeadmconstants.EtcdCAKeyName)
	} else {
		certs[externalEtcdCA] = cfg.Etcd.External.CAFile
		certs[externalEtcdCert] = cfg.Etcd.External.CertFile
		certs[externalEtcdKey] = cfg.Etcd.External.KeyFile
	}
	return certs
}

func getSecretData(cfg *kubeadmapi.InitConfiguration, key []byte) (map[string][]byte, error) {
	secretData := map[string][]byte{}
	for certName, certPath := range certsToUpload(cfg) {
		cert, err := loadAndEncryptCert(certPath, key)
		if err == nil || (err != nil && os.IsNotExist(err)) {
			secretData[strings.Replace(certName, "/", "-", -1)] = cert
		} else {
			return nil, err
		}
	}
	return secretData, nil
}
