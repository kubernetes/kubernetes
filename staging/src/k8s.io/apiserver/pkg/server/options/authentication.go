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

	"github.com/spf13/pflag"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	openapicommon "k8s.io/kube-openapi/pkg/common"
)

type RequestHeaderAuthenticationOptions struct {
	// ClientCAFile is the root certificate bundle to verify client certificates on incoming requests
	// before trusting usernames in headers.
	ClientCAFile string

	UsernameHeaders     []string
	GroupHeaders        []string
	ExtraHeaderPrefixes []string
	AllowedNames        []string
}

func (s *RequestHeaderAuthenticationOptions) AddFlags(fs *pflag.FlagSet) {
	if s == nil {
		return
	}

	fs.StringSliceVar(&s.UsernameHeaders, "requestheader-username-headers", s.UsernameHeaders, ""+
		"List of request headers to inspect for usernames. X-Remote-User is common.")

	fs.StringSliceVar(&s.GroupHeaders, "requestheader-group-headers", s.GroupHeaders, ""+
		"List of request headers to inspect for groups. X-Remote-Group is suggested.")

	fs.StringSliceVar(&s.ExtraHeaderPrefixes, "requestheader-extra-headers-prefix", s.ExtraHeaderPrefixes, ""+
		"List of request header prefixes to inspect. X-Remote-Extra- is suggested.")

	fs.StringVar(&s.ClientCAFile, "requestheader-client-ca-file", s.ClientCAFile, ""+
		"Root certificate bundle to use to verify client certificates on incoming requests "+
		"before trusting usernames in headers specified by --requestheader-username-headers. "+
		"WARNING: generally do not depend on authorization being already done for incoming requests.")

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
	// RemoteKubeConfigFileOptional is specifying whether not specifying the kubeconfig or
	// a missing in-cluster config will be fatal.
	RemoteKubeConfigFileOptional bool

	// CacheTTL is the length of time that a token authentication answer will be cached.
	CacheTTL time.Duration

	ClientCert    ClientCertAuthenticationOptions
	RequestHeader RequestHeaderAuthenticationOptions

	// SkipInClusterLookup indicates missing authentication configuration should not be retrieved from the cluster configmap
	SkipInClusterLookup bool

	// TolerateInClusterLookupFailure indicates failures to look up authentication configuration from the cluster configmap should not be fatal.
	// Setting this can result in an authenticator that will reject all requests.
	TolerateInClusterLookupFailure bool
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
	if s == nil {
		return
	}

	var optionalKubeConfigSentence string
	if s.RemoteKubeConfigFileOptional {
		optionalKubeConfigSentence = " This is optional. If empty, all token requests are considered to be anonymous and no client CA is looked up in the cluster."
	}
	fs.StringVar(&s.RemoteKubeConfigFile, "authentication-kubeconfig", s.RemoteKubeConfigFile, ""+
		"kubeconfig file pointing at the 'core' kubernetes server with enough rights to create "+
		"tokenaccessreviews.authentication.k8s.io."+optionalKubeConfigSentence)

	fs.DurationVar(&s.CacheTTL, "authentication-token-webhook-cache-ttl", s.CacheTTL,
		"The duration to cache responses from the webhook token authenticator.")

	s.ClientCert.AddFlags(fs)
	s.RequestHeader.AddFlags(fs)

	fs.BoolVar(&s.SkipInClusterLookup, "authentication-skip-lookup", s.SkipInClusterLookup, ""+
		"If false, the authentication-kubeconfig will be used to lookup missing authentication "+
		"configuration from the cluster.")
	fs.BoolVar(&s.TolerateInClusterLookupFailure, "authentication-tolerate-lookup-failure", s.TolerateInClusterLookupFailure, ""+
		"If true, failures to look up missing authentication configuration from the cluster are not considered fatal. "+
		"Note that this can result in authentication that treats all requests as anonymous.")
}

func (s *DelegatingAuthenticationOptions) ApplyTo(c *server.AuthenticationInfo, servingInfo *server.SecureServingInfo, openAPIConfig *openapicommon.Config) error {
	if s == nil {
		c.Authenticator = nil
		return nil
	}

	cfg := authenticatorfactory.DelegatingAuthenticatorConfig{
		Anonymous: true,
		CacheTTL:  s.CacheTTL,
	}

	client, err := s.getClient()
	if err != nil {
		return fmt.Errorf("failed to get delegated authentication kubeconfig: %v", err)
	}

	// configure token review
	if client != nil {
		cfg.TokenAccessReviewClient = client.AuthenticationV1beta1().TokenReviews()
	}

	// look into configmaps/external-apiserver-authentication for missing authn info
	if !s.SkipInClusterLookup {
		err := s.lookupMissingConfigInCluster(client)
		if err != nil {
			if s.TolerateInClusterLookupFailure {
				klog.Warningf("Error looking up in-cluster authentication configuration: %v", err)
				klog.Warningf("Continuing without authentication configuration. This may treat all requests as anonymous.")
				klog.Warningf("To require authentication configuration lookup to succeed, set --authentication-tolerate-lookup-failure=false")
			} else {
				return err
			}
		}
	}

	// configure AuthenticationInfo config
	cfg.ClientCAFile = s.ClientCert.ClientCA
	if err = c.ApplyClientCert(s.ClientCert.ClientCA, servingInfo); err != nil {
		return fmt.Errorf("unable to load client CA file: %v", err)
	}

	cfg.RequestHeaderConfig = s.RequestHeader.ToAuthenticationRequestHeaderConfig()
	if err = c.ApplyClientCert(s.RequestHeader.ClientCAFile, servingInfo); err != nil {
		return fmt.Errorf("unable to load client CA file: %v", err)
	}

	// create authenticator
	authenticator, securityDefinitions, err := cfg.New()
	if err != nil {
		return err
	}
	c.Authenticator = authenticator
	if openAPIConfig != nil {
		openAPIConfig.SecurityDefinitions = securityDefinitions
	}
	c.SupportsBasicAuth = false

	return nil
}

const (
	authenticationConfigMapNamespace = metav1.NamespaceSystem
	// authenticationConfigMapName is the name of ConfigMap in the kube-system namespace holding the root certificate
	// bundle to use to verify client certificates on incoming requests before trusting usernames in headers specified
	// by --requestheader-username-headers. This is created in the cluster by the kube-apiserver.
	// "WARNING: generally do not depend on authorization being already done for incoming requests.")
	authenticationConfigMapName = "extension-apiserver-authentication"
	authenticationRoleName      = "extension-apiserver-authentication-reader"
)

func (s *DelegatingAuthenticationOptions) lookupMissingConfigInCluster(client kubernetes.Interface) error {
	if len(s.ClientCert.ClientCA) > 0 && len(s.RequestHeader.ClientCAFile) > 0 {
		return nil
	}
	if client == nil {
		if len(s.ClientCert.ClientCA) == 0 {
			klog.Warningf("No authentication-kubeconfig provided in order to lookup client-ca-file in configmap/%s in %s, so client certificate authentication won't work.", authenticationConfigMapName, authenticationConfigMapNamespace)
		}
		if len(s.RequestHeader.ClientCAFile) == 0 {
			klog.Warningf("No authentication-kubeconfig provided in order to lookup requestheader-client-ca-file in configmap/%s in %s, so request-header client certificate authentication won't work.", authenticationConfigMapName, authenticationConfigMapNamespace)
		}
		return nil
	}

	authConfigMap, err := client.CoreV1().ConfigMaps(authenticationConfigMapNamespace).Get(authenticationConfigMapName, metav1.GetOptions{})
	switch {
	case errors.IsNotFound(err):
		// ignore, authConfigMap is nil now
	case errors.IsForbidden(err):
		klog.Warningf("Unable to get configmap/%s in %s.  Usually fixed by "+
			"'kubectl create rolebinding -n %s ROLE_NAME --role=%s --serviceaccount=YOUR_NS:YOUR_SA'",
			authenticationConfigMapName, authenticationConfigMapNamespace, authenticationConfigMapNamespace, authenticationRoleName)
		return err
	case err != nil:
		return err
	}

	if len(s.ClientCert.ClientCA) == 0 {
		if authConfigMap != nil {
			opt, err := inClusterClientCA(authConfigMap)
			if err != nil {
				return err
			}
			if opt != nil {
				s.ClientCert = *opt
			}
		}
		if len(s.ClientCert.ClientCA) == 0 {
			klog.Warningf("Cluster doesn't provide client-ca-file in configmap/%s in %s, so client certificate authentication won't work.", authenticationConfigMapName, authenticationConfigMapNamespace)
		}
	}

	if len(s.RequestHeader.ClientCAFile) == 0 {
		if authConfigMap != nil {
			opt, err := inClusterRequestHeader(authConfigMap)
			if err != nil {
				return err
			}
			if opt != nil {
				s.RequestHeader = *opt
			}
		}
		if len(s.RequestHeader.ClientCAFile) == 0 {
			klog.Warningf("Cluster doesn't provide requestheader-client-ca-file in configmap/%s in %s, so request-header client certificate authentication won't work.", authenticationConfigMapName, authenticationConfigMapNamespace)
		}
	}

	return nil
}

func inClusterClientCA(authConfigMap *v1.ConfigMap) (*ClientCertAuthenticationOptions, error) {
	clientCA, ok := authConfigMap.Data["client-ca-file"]
	if !ok {
		// not having a client-ca is fine, return nil
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

func inClusterRequestHeader(authConfigMap *v1.ConfigMap) (*RequestHeaderAuthenticationOptions, error) {
	requestHeaderCA, ok := authConfigMap.Data["requestheader-client-ca-file"]
	if !ok {
		// not having a requestheader-client-ca is fine, return nil
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

// getClient returns a Kubernetes clientset. If s.RemoteKubeConfigFileOptional is true, nil will be returned
// if no kubeconfig is specified by the user and the in-cluster config is not found.
func (s *DelegatingAuthenticationOptions) getClient() (kubernetes.Interface, error) {
	var clientConfig *rest.Config
	var err error
	if len(s.RemoteKubeConfigFile) > 0 {
		loadingRules := &clientcmd.ClientConfigLoadingRules{ExplicitPath: s.RemoteKubeConfigFile}
		loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})

		clientConfig, err = loader.ClientConfig()
	} else {
		// without the remote kubeconfig file, try to use the in-cluster config.  Most addon API servers will
		// use this path. If it is optional, ignore errors.
		clientConfig, err = rest.InClusterConfig()
		if err != nil && s.RemoteKubeConfigFileOptional {
			if err != rest.ErrNotInCluster {
				klog.Warningf("failed to read in-cluster kubeconfig for delegated authentication: %v", err)
			}
			return nil, nil
		}
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get delegated authentication kubeconfig: %v", err)
	}

	// set high qps/burst limits since this will effectively limit API server responsiveness
	clientConfig.QPS = 200
	clientConfig.Burst = 400

	return kubernetes.NewForConfig(clientConfig)
}
