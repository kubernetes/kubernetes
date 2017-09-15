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
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/spf13/pflag"

	genericapiserver "k8s.io/apiserver/pkg/server"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/kubernetes/pkg/kubeapiserver/authenticator"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
)

type BuiltInAuthenticationOptions struct {
	Anonymous       *AnonymousAuthenticationOptions
	BootstrapToken  *BootstrapTokenAuthenticationOptions
	ClientCert      *genericoptions.ClientCertAuthenticationOptions
	Keystone        *KeystoneAuthenticationOptions
	OIDC            *OIDCAuthenticationOptions
	PasswordFile    *PasswordFileAuthenticationOptions
	RequestHeader   *genericoptions.RequestHeaderAuthenticationOptions
	ServiceAccounts *ServiceAccountAuthenticationOptions
	TokenFile       *TokenFileAuthenticationOptions
	WebHook         *WebHookAuthenticationOptions

	TokenSuccessCacheTTL time.Duration
	TokenFailureCacheTTL time.Duration
}

type AnonymousAuthenticationOptions struct {
	Allow bool
}

type BootstrapTokenAuthenticationOptions struct {
	Enable bool
}

type KeystoneAuthenticationOptions struct {
	URL    string
	CAFile string
}

type OIDCAuthenticationOptions struct {
	CAFile         string
	ClientID       string
	IssuerURL      string
	UsernameClaim  string
	UsernamePrefix string
	GroupsClaim    string
	GroupsPrefix   string
}

type PasswordFileAuthenticationOptions struct {
	BasicAuthFile string
}

type ServiceAccountAuthenticationOptions struct {
	KeyFiles []string
	Lookup   bool
}

type TokenFileAuthenticationOptions struct {
	TokenFile string
}

type WebHookAuthenticationOptions struct {
	ConfigFile string
	CacheTTL   time.Duration
}

func NewBuiltInAuthenticationOptions() *BuiltInAuthenticationOptions {
	return &BuiltInAuthenticationOptions{
		TokenSuccessCacheTTL: 10 * time.Second,
		TokenFailureCacheTTL: 0 * time.Second,
	}
}

func (s *BuiltInAuthenticationOptions) WithAll() *BuiltInAuthenticationOptions {
	return s.
		WithAnonymous().
		WithBootstrapToken().
		WithClientCert().
		WithKeystone().
		WithOIDC().
		WithPasswordFile().
		WithRequestHeader().
		WithServiceAccounts().
		WithTokenFile().
		WithWebHook()
}

func (s *BuiltInAuthenticationOptions) WithAnonymous() *BuiltInAuthenticationOptions {
	s.Anonymous = &AnonymousAuthenticationOptions{Allow: true}
	return s
}

func (s *BuiltInAuthenticationOptions) WithBootstrapToken() *BuiltInAuthenticationOptions {
	s.BootstrapToken = &BootstrapTokenAuthenticationOptions{}
	return s
}

func (s *BuiltInAuthenticationOptions) WithClientCert() *BuiltInAuthenticationOptions {
	s.ClientCert = &genericoptions.ClientCertAuthenticationOptions{}
	return s
}

func (s *BuiltInAuthenticationOptions) WithKeystone() *BuiltInAuthenticationOptions {
	s.Keystone = &KeystoneAuthenticationOptions{}
	return s
}

func (s *BuiltInAuthenticationOptions) WithOIDC() *BuiltInAuthenticationOptions {
	s.OIDC = &OIDCAuthenticationOptions{}
	return s
}

func (s *BuiltInAuthenticationOptions) WithPasswordFile() *BuiltInAuthenticationOptions {
	s.PasswordFile = &PasswordFileAuthenticationOptions{}
	return s
}

func (s *BuiltInAuthenticationOptions) WithRequestHeader() *BuiltInAuthenticationOptions {
	s.RequestHeader = &genericoptions.RequestHeaderAuthenticationOptions{}
	return s
}

func (s *BuiltInAuthenticationOptions) WithServiceAccounts() *BuiltInAuthenticationOptions {
	s.ServiceAccounts = &ServiceAccountAuthenticationOptions{Lookup: true}
	return s
}

func (s *BuiltInAuthenticationOptions) WithTokenFile() *BuiltInAuthenticationOptions {
	s.TokenFile = &TokenFileAuthenticationOptions{}
	return s
}

func (s *BuiltInAuthenticationOptions) WithWebHook() *BuiltInAuthenticationOptions {
	s.WebHook = &WebHookAuthenticationOptions{
		CacheTTL: 2 * time.Minute,
	}
	return s
}

// Validate checks invalid config combination
func (s *BuiltInAuthenticationOptions) Validate() []error {
	allErrors := []error{}

	if s.OIDC != nil && (len(s.OIDC.IssuerURL) > 0) != (len(s.OIDC.ClientID) > 0) {
		allErrors = append(allErrors, fmt.Errorf("oidc-issuer-url and oidc-client-id should be specified together"))
	}

	return allErrors
}

func (s *BuiltInAuthenticationOptions) AddFlags(fs *pflag.FlagSet) {
	if s.Anonymous != nil {
		fs.BoolVar(&s.Anonymous.Allow, "anonymous-auth", s.Anonymous.Allow, ""+
			"Enables anonymous requests to the secure port of the API server. "+
			"Requests that are not rejected by another authentication method are treated as anonymous requests. "+
			"Anonymous requests have a username of system:anonymous, and a group name of system:unauthenticated.")
	}

	if s.BootstrapToken != nil {
		fs.BoolVar(&s.BootstrapToken.Enable, "enable-bootstrap-token-auth", s.BootstrapToken.Enable, ""+
			"Enable to allow secrets of type 'bootstrap.kubernetes.io/token' in the 'kube-system' "+
			"namespace to be used for TLS bootstrapping authentication.")
	}

	if s.ClientCert != nil {
		s.ClientCert.AddFlags(fs)
	}

	if s.Keystone != nil {
		fs.StringVar(&s.Keystone.URL, "experimental-keystone-url", s.Keystone.URL,
			"If passed, activates the keystone authentication plugin.")

		fs.StringVar(&s.Keystone.CAFile, "experimental-keystone-ca-file", s.Keystone.CAFile, ""+
			"If set, the Keystone server's certificate will be verified by one of the authorities "+
			"in the experimental-keystone-ca-file, otherwise the host's root CA set will be used.")
	}

	if s.OIDC != nil {
		fs.StringVar(&s.OIDC.IssuerURL, "oidc-issuer-url", s.OIDC.IssuerURL, ""+
			"The URL of the OpenID issuer, only HTTPS scheme will be accepted. "+
			"If set, it will be used to verify the OIDC JSON Web Token (JWT).")

		fs.StringVar(&s.OIDC.ClientID, "oidc-client-id", s.OIDC.ClientID,
			"The client ID for the OpenID Connect client, must be set if oidc-issuer-url is set.")

		fs.StringVar(&s.OIDC.CAFile, "oidc-ca-file", s.OIDC.CAFile, ""+
			"If set, the OpenID server's certificate will be verified by one of the authorities "+
			"in the oidc-ca-file, otherwise the host's root CA set will be used.")

		fs.StringVar(&s.OIDC.UsernameClaim, "oidc-username-claim", "sub", ""+
			"The OpenID claim to use as the user name. Note that claims other than the default ('sub') "+
			"is not guaranteed to be unique and immutable. This flag is experimental, please see "+
			"the authentication documentation for further details.")

		fs.StringVar(&s.OIDC.UsernamePrefix, "oidc-username-prefix", "", ""+
			"If provided, all usernames will be prefixed with this value. If not provided, "+
			"username claims other than 'email' are prefixed by the issuer URL to avoid "+
			"clashes. To skip any prefixing, provide the value '-'.")

		fs.StringVar(&s.OIDC.GroupsClaim, "oidc-groups-claim", "", ""+
			"If provided, the name of a custom OpenID Connect claim for specifying user groups. "+
			"The claim value is expected to be a string or array of strings. This flag is experimental, "+
			"please see the authentication documentation for further details.")

		fs.StringVar(&s.OIDC.GroupsPrefix, "oidc-groups-prefix", "", ""+
			"If provided, all groups will be prefixed with this value to prevent conflicts with "+
			"other authentication strategies.")

	}

	if s.PasswordFile != nil {
		fs.StringVar(&s.PasswordFile.BasicAuthFile, "basic-auth-file", s.PasswordFile.BasicAuthFile, ""+
			"If set, the file that will be used to admit requests to the secure port of the API server "+
			"via http basic authentication.")
	}

	if s.RequestHeader != nil {
		s.RequestHeader.AddFlags(fs)
	}

	if s.ServiceAccounts != nil {
		fs.StringArrayVar(&s.ServiceAccounts.KeyFiles, "service-account-key-file", s.ServiceAccounts.KeyFiles, ""+
			"File containing PEM-encoded x509 RSA or ECDSA private or public keys, used to verify "+
			"ServiceAccount tokens. If unspecified, --tls-private-key-file is used. "+
			"The specified file can contain multiple keys, and the flag can be specified multiple times with different files.")

		fs.BoolVar(&s.ServiceAccounts.Lookup, "service-account-lookup", s.ServiceAccounts.Lookup,
			"If true, validate ServiceAccount tokens exist in etcd as part of authentication.")
	}

	if s.TokenFile != nil {
		fs.StringVar(&s.TokenFile.TokenFile, "token-auth-file", s.TokenFile.TokenFile, ""+
			"If set, the file that will be used to secure the secure port of the API server "+
			"via token authentication.")
	}

	if s.WebHook != nil {
		fs.StringVar(&s.WebHook.ConfigFile, "authentication-token-webhook-config-file", s.WebHook.ConfigFile, ""+
			"File with webhook configuration for token authentication in kubeconfig format. "+
			"The API server will query the remote service to determine authentication for bearer tokens.")

		fs.DurationVar(&s.WebHook.CacheTTL, "authentication-token-webhook-cache-ttl", s.WebHook.CacheTTL,
			"The duration to cache responses from the webhook token authenticator.")
	}
}

func (s *BuiltInAuthenticationOptions) ToAuthenticationConfig() authenticator.AuthenticatorConfig {
	ret := authenticator.AuthenticatorConfig{
		TokenSuccessCacheTTL: s.TokenSuccessCacheTTL,
		TokenFailureCacheTTL: s.TokenFailureCacheTTL,
	}

	if s.Anonymous != nil {
		ret.Anonymous = s.Anonymous.Allow
	}

	if s.BootstrapToken != nil {
		ret.BootstrapToken = s.BootstrapToken.Enable
	}

	if s.ClientCert != nil {
		ret.ClientCAFile = s.ClientCert.ClientCA
	}

	if s.Keystone != nil {
		ret.KeystoneURL = s.Keystone.URL
		ret.KeystoneCAFile = s.Keystone.CAFile
	}

	if s.OIDC != nil {
		ret.OIDCCAFile = s.OIDC.CAFile
		ret.OIDCClientID = s.OIDC.ClientID
		ret.OIDCGroupsClaim = s.OIDC.GroupsClaim
		ret.OIDCIssuerURL = s.OIDC.IssuerURL
		ret.OIDCUsernameClaim = s.OIDC.UsernameClaim
	}

	if s.PasswordFile != nil {
		ret.BasicAuthFile = s.PasswordFile.BasicAuthFile
	}

	if s.RequestHeader != nil {
		ret.RequestHeaderConfig = s.RequestHeader.ToAuthenticationRequestHeaderConfig()
	}

	if s.ServiceAccounts != nil {
		ret.ServiceAccountKeyFiles = s.ServiceAccounts.KeyFiles
		ret.ServiceAccountLookup = s.ServiceAccounts.Lookup
	}

	if s.TokenFile != nil {
		ret.TokenAuthFile = s.TokenFile.TokenFile
	}

	if s.WebHook != nil {
		ret.WebhookTokenAuthnConfigFile = s.WebHook.ConfigFile
		ret.WebhookTokenAuthnCacheTTL = s.WebHook.CacheTTL

		if len(s.WebHook.ConfigFile) > 0 && s.WebHook.CacheTTL > 0 {
			if s.TokenSuccessCacheTTL > 0 && s.WebHook.CacheTTL < s.TokenSuccessCacheTTL {
				glog.Warningf("the webhook cache ttl of %s is shorter than the overall cache ttl of %s for successful token authentication attempts.", s.WebHook.CacheTTL, s.TokenSuccessCacheTTL)
			}
			if s.TokenFailureCacheTTL > 0 && s.WebHook.CacheTTL < s.TokenFailureCacheTTL {
				glog.Warningf("the webhook cache ttl of %s is shorter than the overall cache ttl of %s for failed token authentication attempts.", s.WebHook.CacheTTL, s.TokenFailureCacheTTL)
			}
		}
	}

	return ret
}

func (o *BuiltInAuthenticationOptions) ApplyTo(c *genericapiserver.Config) error {
	if o == nil {
		return nil
	}

	var err error
	if o.ClientCert != nil {
		c, err = c.ApplyClientCert(o.ClientCert.ClientCA)
		if err != nil {
			return fmt.Errorf("unable to load client CA file: %v", err)
		}
	}
	if o.RequestHeader != nil {
		c, err = c.ApplyClientCert(o.RequestHeader.ClientCAFile)
		if err != nil {
			return fmt.Errorf("unable to load client CA file: %v", err)
		}
	}

	c.SupportsBasicAuth = o.PasswordFile != nil && len(o.PasswordFile.BasicAuthFile) > 0

	return nil
}

// ApplyAuthorization will conditionally modify the authentication options based on the authorization options
func (o *BuiltInAuthenticationOptions) ApplyAuthorization(authorization *BuiltInAuthorizationOptions) {
	if o == nil || authorization == nil || o.Anonymous == nil {
		return
	}

	// authorization ModeAlwaysAllow cannot be combined with AnonymousAuth.
	// in such a case the AnonymousAuth is stomped to false and you get a message
	if o.Anonymous.Allow {
		found := false
		for _, mode := range strings.Split(authorization.Mode, ",") {
			if mode == authzmodes.ModeAlwaysAllow {
				found = true
				break
			}
		}
		if found {
			glog.Warningf("AnonymousAuth is not allowed with the AllowAll authorizer.  Resetting AnonymousAuth to false. You should use a different authorizer")
			o.Anonymous.Allow = false
		}
	}
}
