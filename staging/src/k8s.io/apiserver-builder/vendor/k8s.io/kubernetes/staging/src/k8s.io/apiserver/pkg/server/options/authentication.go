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

package options

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"time"

	"github.com/golang/glog"
	"github.com/spf13/pflag"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	"k8s.io/apiserver/pkg/server"
	authenticationclient "k8s.io/client-go/kubernetes/typed/authentication/v1beta1"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

type RequestHeaderAuthenticationOptions struct {
	UsernameHeaders     []string
	GroupHeaders        []string
	ExtraHeaderPrefixes []string
	ClientCAFile        string
	AllowedNames        []string
}

func (s *RequestHeaderAuthenticationOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringSliceVar(&s.UsernameHeaders, "requestheader-username-headers", s.UsernameHeaders, ""+
		"List of request headers to inspect for usernames. X-Remote-User is common.")

	fs.StringSliceVar(&s.GroupHeaders, "requestheader-group-headers", s.GroupHeaders, ""+
		"List of request headers to inspect for groups. X-Remote-Group is suggested.")

	fs.StringSliceVar(&s.ExtraHeaderPrefixes, "requestheader-extra-headers-prefix", s.ExtraHeaderPrefixes, ""+
		"List of request header prefixes to inspect. X-Remote-Extra- is suggested.")

	fs.StringVar(&s.ClientCAFile, "requestheader-client-ca-file", s.ClientCAFile, ""+
		"Root certificate bundle to use to verify client certificates on incoming requests "+
		"before trusting usernames in headers specified by --requestheader-username-headers")

	fs.StringSliceVar(&s.AllowedNames, "requestheader-allowed-names", s.AllowedNames, ""+
		"List of client certificate common names to allow to provide usernames in headers "+
		"specified by --requestheader-username-headers. If empty, any client certificate validated "+
		"by the authorities in --requestheader-client-ca-file is allowed.")
}

// ToAuthenticationRequestHeaderConfig returns a RequestHeaderConfig config object for these options
// if necessary, nil otherwise.
func (s *RequestHeaderAuthenticationOptions) ToAuthenticationRequestHeaderConfig() *authenticatorfactory.RequestHeaderConfig {
	if len(s.ClientCAFile) == 0 {
		return nil
	}

	return &authenticatorfactory.RequestHeaderConfig{
		UsernameHeaders:     s.UsernameHeaders,
		GroupHeaders:        s.GroupHeaders,
		ExtraHeaderPrefixes: s.ExtraHeaderPrefixes,
		ClientCA:            s.ClientCAFile,
		AllowedClientNames:  s.AllowedNames,
	}
}

type ClientCertAuthenticationOptions struct {
	// ClientCA is the certificate bundle for all the signers that you'll recognize for incoming client certificates
	ClientCA string
}

func (s *ClientCertAuthenticationOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.ClientCA, "client-ca-file", s.ClientCA, ""+
		"If set, any request presenting a client certificate signed by one of "+
		"the authorities in the client-ca-file is authenticated with an identity "+
		"corresponding to the CommonName of the client certificate.")
}

// DelegatingAuthenticationOptions provides an easy way for composing API servers to delegate their authentication to
// the root kube API server.  The API federator will act as
// a front proxy and direction connections will be able to delegate to the core kube API server
type DelegatingAuthenticationOptions struct {
	// RemoteKubeConfigFile is the file to use to connect to a "normal" kube API server which hosts the
	// TokenAccessReview.authentication.k8s.io endpoint for checking tokens.
	RemoteKubeConfigFile string

	// CacheTTL is the length of time that a token authentication answer will be cached.
	CacheTTL time.Duration

	ClientCert    ClientCertAuthenticationOptions
	RequestHeader RequestHeaderAuthenticationOptions

	SkipInClusterLookup bool
}

func NewDelegatingAuthenticationOptions() *DelegatingAuthenticationOptions {
	return &DelegatingAuthenticationOptions{
		// very low for responsiveness, but high enough to handle storms
		CacheTTL:   10 * time.Second,
		ClientCert: ClientCertAuthenticationOptions{},
		RequestHeader: RequestHeaderAuthenticationOptions{
			UsernameHeaders:     []string{"x-remote-user"},
			GroupHeaders:        []string{"x-remote-group"},
			ExtraHeaderPrefixes: []string{"x-remote-extra-"},
		},
	}
}

func (s *DelegatingAuthenticationOptions) Validate() []error {
	allErrors := []error{}
	return allErrors
}

func (s *DelegatingAuthenticationOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.RemoteKubeConfigFile, "authentication-kubeconfig", s.RemoteKubeConfigFile, ""+
		"kubeconfig file pointing at the 'core' kubernetes server with enough rights to create "+
		"tokenaccessreviews.authentication.k8s.io.")

	fs.DurationVar(&s.CacheTTL, "authentication-token-webhook-cache-ttl", s.CacheTTL,
		"The duration to cache responses from the webhook token authenticator.")

	s.ClientCert.AddFlags(fs)
	s.RequestHeader.AddFlags(fs)

	fs.BoolVar(&s.SkipInClusterLookup, "authentication-skip-lookup", s.SkipInClusterLookup, ""+
		"If false, the authentication-kubeconfig will be used to lookup missing authentication "+
		"configuration from the cluster.")

}

func (s *DelegatingAuthenticationOptions) ApplyTo(c *server.Config) error {
	clientCA, err := s.getClientCA()
	if err != nil {
		return err
	}
	c, err = c.ApplyClientCert(clientCA.ClientCA)
	if err != nil {
		return fmt.Errorf("unable to load client CA file: %v", err)
	}

	requestHeader, err := s.getRequestHeader()
	if err != nil {
		return err
	}
	c, err = c.ApplyClientCert(requestHeader.ClientCAFile)
	if err != nil {
		return fmt.Errorf("unable to load client CA file: %v", err)
	}

	cfg, err := s.ToAuthenticationConfig()
	if err != nil {
		return err
	}
	authenticator, securityDefinitions, err := cfg.New()
	if err != nil {
		return err
	}

	c.Authenticator = authenticator
	if c.OpenAPIConfig != nil {
		c.OpenAPIConfig.SecurityDefinitions = securityDefinitions
	}
	c.SupportsBasicAuth = false

	return nil
}

func (s *DelegatingAuthenticationOptions) ToAuthenticationConfig() (authenticatorfactory.DelegatingAuthenticatorConfig, error) {
	tokenClient, err := s.newTokenAccessReview()
	if err != nil {
		return authenticatorfactory.DelegatingAuthenticatorConfig{}, err
	}

	clientCA, err := s.getClientCA()
	if err != nil {
		return authenticatorfactory.DelegatingAuthenticatorConfig{}, err
	}
	requestHeader, err := s.getRequestHeader()
	if err != nil {
		return authenticatorfactory.DelegatingAuthenticatorConfig{}, err
	}

	ret := authenticatorfactory.DelegatingAuthenticatorConfig{
		Anonymous:               true,
		TokenAccessReviewClient: tokenClient,
		CacheTTL:                s.CacheTTL,
		ClientCAFile:            clientCA.ClientCA,
		RequestHeaderConfig:     requestHeader.ToAuthenticationRequestHeaderConfig(),
	}
	return ret, nil
}

const (
	authenticationConfigMapNamespace = metav1.NamespaceSystem
	authenticationConfigMapName      = "extension-apiserver-authentication"
	authenticationRoleName           = "extension-apiserver-authentication-reader"
)

func (s *DelegatingAuthenticationOptions) getClientCA() (*ClientCertAuthenticationOptions, error) {
	if len(s.ClientCert.ClientCA) > 0 || s.SkipInClusterLookup {
		return &s.ClientCert, nil
	}

	incluster, err := s.lookupInClusterClientCA()
	if err != nil {
		glog.Warningf("Unable to get configmap/%s in %s.  Usually fixed by "+
			"'kubectl create rolebinding -n %s ROLE_NAME --role=%s --serviceaccount=YOUR_NS:YOUR_SA'",
			authenticationConfigMapName, authenticationConfigMapNamespace, authenticationConfigMapNamespace, authenticationRoleName)
		return nil, err
	}
	if incluster == nil {
		return nil, fmt.Errorf("cluster doesn't provide client-ca-file")
	}
	return incluster, nil
}

func (s *DelegatingAuthenticationOptions) getRequestHeader() (*RequestHeaderAuthenticationOptions, error) {
	if len(s.RequestHeader.ClientCAFile) > 0 || s.SkipInClusterLookup {
		return &s.RequestHeader, nil
	}

	incluster, err := s.lookupInClusterRequestHeader()
	if err != nil {
		glog.Warningf("Unable to get configmap/%s in %s.  Usually fixed by "+
			"'kubectl create rolebinding -n %s ROLE_NAME --role=%s --serviceaccount=YOUR_NS:YOUR_SA'",
			authenticationConfigMapName, authenticationConfigMapNamespace, authenticationConfigMapNamespace, authenticationRoleName)
		return nil, err
	}
	if incluster == nil {
		return nil, fmt.Errorf("cluster doesn't provide requestheader-client-ca-file")
	}
	return incluster, nil
}

func (s *DelegatingAuthenticationOptions) lookupInClusterClientCA() (*ClientCertAuthenticationOptions, error) {
	clientConfig, err := s.getClientConfig()
	if err != nil {
		return nil, err
	}
	client, err := coreclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	authConfigMap, err := client.ConfigMaps(authenticationConfigMapNamespace).Get(authenticationConfigMapName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	clientCA, ok := authConfigMap.Data["client-ca-file"]
	if !ok {
		return nil, nil
	}

	f, err := ioutil.TempFile("", "client-ca-file")
	if err != nil {
		return nil, err
	}
	if err := ioutil.WriteFile(f.Name(), []byte(clientCA), 0600); err != nil {
		return nil, err
	}
	return &ClientCertAuthenticationOptions{ClientCA: f.Name()}, nil
}

func (s *DelegatingAuthenticationOptions) lookupInClusterRequestHeader() (*RequestHeaderAuthenticationOptions, error) {
	clientConfig, err := s.getClientConfig()
	if err != nil {
		return nil, err
	}
	client, err := coreclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	authConfigMap, err := client.ConfigMaps(authenticationConfigMapNamespace).Get(authenticationConfigMapName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	requestHeaderCA, ok := authConfigMap.Data["requestheader-client-ca-file"]
	if !ok {
		return nil, nil
	}

	f, err := ioutil.TempFile("", "requestheader-client-ca-file")
	if err != nil {
		return nil, err
	}
	if err := ioutil.WriteFile(f.Name(), []byte(requestHeaderCA), 0600); err != nil {
		return nil, err
	}
	usernameHeaders, err := deserializeStrings(authConfigMap.Data["requestheader-username-headers"])
	if err != nil {
		return nil, err
	}
	groupHeaders, err := deserializeStrings(authConfigMap.Data["requestheader-group-headers"])
	if err != nil {
		return nil, err
	}
	extraHeaderPrefixes, err := deserializeStrings(authConfigMap.Data["requestheader-extra-headers-prefix"])
	if err != nil {
		return nil, err
	}
	allowedNames, err := deserializeStrings(authConfigMap.Data["requestheader-allowed-names"])
	if err != nil {
		return nil, err
	}

	return &RequestHeaderAuthenticationOptions{
		UsernameHeaders:     usernameHeaders,
		GroupHeaders:        groupHeaders,
		ExtraHeaderPrefixes: extraHeaderPrefixes,
		ClientCAFile:        f.Name(),
		AllowedNames:        allowedNames,
	}, nil
}

func deserializeStrings(in string) ([]string, error) {
	if len(in) == 0 {
		return nil, nil
	}
	var ret []string
	if err := json.Unmarshal([]byte(in), &ret); err != nil {
		return nil, err
	}
	return ret, nil
}

func (s *DelegatingAuthenticationOptions) getClientConfig() (*rest.Config, error) {
	var clientConfig *rest.Config
	var err error
	if len(s.RemoteKubeConfigFile) > 0 {
		loadingRules := &clientcmd.ClientConfigLoadingRules{ExplicitPath: s.RemoteKubeConfigFile}
		loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})

		clientConfig, err = loader.ClientConfig()

	} else {
		// without the remote kubeconfig file, try to use the in-cluster config.  Most addon API servers will
		// use this path
		clientConfig, err = rest.InClusterConfig()
	}
	if err != nil {
		return nil, err
	}

	// set high qps/burst limits since this will effectively limit API server responsiveness
	clientConfig.QPS = 200
	clientConfig.Burst = 400

	return clientConfig, nil
}

func (s *DelegatingAuthenticationOptions) newTokenAccessReview() (authenticationclient.TokenReviewInterface, error) {
	clientConfig, err := s.getClientConfig()
	if err != nil {
		return nil, err
	}
	client, err := authenticationclient.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	return client.TokenReviews(), nil
}
