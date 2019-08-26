// +build !providerless

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

package eks

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/sts"
	"golang.org/x/oauth2"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/klog"
)

// ClientConfig defines AWS EKS specific client/authentication configuration.
type ClientConfig struct {
	ContentType string

	KubeconfigPath     string
	ClusterContextName string
	Region             string
	ClusterName        string
}

// NewClientConfig creates a new Kubernetes rest client
// and if successful, registers EKS auth provider.
func NewClientConfig(cfg ClientConfig) (clientConfig *restclient.Config, err error) {
	if cfg.ClusterContextName == "" {
		return nil, fmt.Errorf("empty cluster context name is given for auth provider 'eks'")
	}
	if cfg.Region == "" {
		return nil, fmt.Errorf("empty region is given for auth provider 'eks'")
	}
	if cfg.ClusterName == "" {
		return nil, fmt.Errorf("empty cluster name is given for auth provider 'eks'")
	}

	var kcfg *clientcmdapi.Config
	kcfg, err = clientcmd.LoadFromFile(cfg.KubeconfigPath)
	if err != nil {
		return nil, fmt.Errorf("error while loading kubeconfig from file %q: %v", cfg.KubeconfigPath, err)
	}
	v, ok := kcfg.Clusters[cfg.ClusterContextName]
	if !ok {
		keys := make([]string, 0, len(kcfg.Clusters))
		for k := range kcfg.Clusters {
			keys = append(keys, k)
		}
		if err != nil {
			return nil, fmt.Errorf("cluster context name %q not found in kubeconfig 'clusters' (%v)", cfg.ClusterContextName, keys)
		}
	}

	klog.Infof("setting rest client to server %q from kubeconfig %q", v.Server, cfg.KubeconfigPath)
	clientConfig = &restclient.Config{
		// set default values
		QPS:   10.0,
		Burst: 20,

		Host: v.Server,
		TLSClientConfig: restclient.TLSClientConfig{
			// no need base64-decode, the field is already PEM-encoded when returned from 'clientcmd.LoadFromFile'
			CAData: v.CertificateAuthorityData,
		},
		AuthProvider: &clientcmdapi.AuthProviderConfig{
			Name: "eks",
			Config: map[string]string{
				// TODO: support temporary credentials
				"region":       cfg.Region,
				"cluster-name": cfg.ClusterName,
			},
		},
	}
	clientConfig.ContentConfig.ContentType = cfg.ContentType

	// required for custom EKS auth provider
	restclient.RegisterAuthProviderPlugin("eks", newEKSAuthProvider)
	klog.Infof("successfully registered auth provider 'eks' with region %q and cluster name %q", cfg.Region, cfg.ClusterName)

	return clientConfig, nil
}

func newEKSAuthProvider(_ string, config map[string]string, _ restclient.AuthProviderConfigPersister) (restclient.AuthProvider, error) {
	// TODO: support temporary credentials
	awsRegion, ok := config["region"]
	if !ok {
		return nil, fmt.Errorf("'clientcmdapi.AuthProviderConfig' does not include 'region' key %+v", config)
	}
	clusterName, ok := config["cluster-name"]
	if !ok {
		return nil, fmt.Errorf("'clientcmdapi.AuthProviderConfig' does not include 'cluster-name' key %+v", config)
	}
	sess := session.Must(session.NewSession(aws.NewConfig().WithRegion(awsRegion)))
	return &eksAuthProvider{ts: newEKSTokenSource(sess, clusterName)}, nil
}

type eksAuthProvider struct {
	ts oauth2.TokenSource
}

func (p *eksAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &oauth2.Transport{
		Source: p.ts,
		Base:   rt,
	}
}

func (p *eksAuthProvider) Login() error {
	return nil
}

func newEKSTokenSource(sess *session.Session, clusterName string) oauth2.TokenSource {
	return &eksTokenSource{sess: sess, clusterName: clusterName}
}

type eksTokenSource struct {
	sess        *session.Session
	clusterName string
}

// Reference
// https://github.com/kubernetes-sigs/aws-iam-authenticator/blob/master/README.md#api-authorization-from-outside-a-cluster
// https://github.com/kubernetes-sigs/aws-iam-authenticator/blob/master/pkg/token/token.go
const (
	eksV1Prefix        = "k8s-aws-v1."
	eksClusterIDHeader = "x-k8s-aws-id"
)

func (s *eksTokenSource) Token() (*oauth2.Token, error) {
	stsAPI := sts.New(s.sess)
	request, _ := stsAPI.GetCallerIdentityRequest(&sts.GetCallerIdentityInput{})
	request.HTTPRequest.Header.Add(eksClusterIDHeader, s.clusterName)

	payload, err := request.Presign(60)
	if err != nil {
		return nil, err
	}
	token := eksV1Prefix + base64.RawURLEncoding.EncodeToString([]byte(payload))
	tokenExpiration := time.Now().Local().Add(14 * time.Minute)
	return &oauth2.Token{
		AccessToken: token,
		TokenType:   "Bearer",
		Expiry:      tokenExpiration,
	}, nil
}
