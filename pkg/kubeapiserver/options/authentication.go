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
	"time"

	"github.com/spf13/pflag"

	"k8s.io/kubernetes/pkg/genericapiserver"
	genericoptions "k8s.io/kubernetes/pkg/genericapiserver/options"
	"k8s.io/kubernetes/pkg/kubeapiserver/authenticator"
)

type BuiltInAuthenticationOptions struct {
	Anonymous       *AnonymousAuthenticationOptions
	AnyToken        *AnyTokenAuthenticationOptions
	ClientCert      *genericoptions.ClientCertAuthenticationOptions
	Keystone        *KeystoneAuthenticationOptions
	OIDC            *OIDCAuthenticationOptions
	PasswordFile    *PasswordFileAuthenticationOptions
	RequestHeader   *genericoptions.RequestHeaderAuthenticationOptions
	ServiceAccounts *ServiceAccountAuthenticationOptions
	TokenFile       *TokenFileAuthenticationOptions
	WebHook         *WebHookAuthenticationOptions
}

type AnyTokenAuthenticationOptions struct {
	Allow bool
}

type AnonymousAuthenticationOptions struct {
	Allow bool
}

type KeystoneAuthenticationOptions struct {
	URL    string
	CAFile string
}

type OIDCAuthenticationOptions struct {
	CAFile        string
	ClientID      string
	IssuerURL     string
	UsernameClaim string
	GroupsClaim   string
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
	return &BuiltInAuthenticationOptions{}
}

func (s *BuiltInAuthenticationOptions) WithAll() *BuiltInAuthenticationOptions {
	return s.
		WithAnyonymous().
		WithAnyToken().
		WithClientCert().
		WithKeystone().
		WithOIDC().
		WithPasswordFile().
		WithRequestHeader().
		WithServiceAccounts().
		WithTokenFile().
		WithWebHook()
}

func (s *BuiltInAuthenticationOptions) WithAnyonymous() *BuiltInAuthenticationOptions {
	s.Anonymous = &AnonymousAuthenticationOptions{Allow: true}
	return s
}

func (s *BuiltInAuthenticationOptions) WithAnyToken() *BuiltInAuthenticationOptions {
	s.AnyToken = &AnyTokenAuthenticationOptions{}
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
	s.ServiceAccounts = &ServiceAccountAuthenticationOptions{}
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

func (s *BuiltInAuthenticationOptions) Validate() []error {
	allErrors := []error{}
	return allErrors
}

func (s *BuiltInAuthenticationOptions) AddFlags(fs *pflag.FlagSet) {
	if s.Anonymous != nil {
		fs.BoolVar(&s.Anonymous.Allow, "anonymous-auth", s.Anonymous.Allow, ""+
			"Enables anonymous requests to the secure port of the API server. "+
			"Requests that are not rejected by another authentication method are treated as anonymous requests. "+
			"Anonymous requests have a username of system:anonymous, and a group name of system:unauthenticated.")
	}

	if s.AnyToken != nil {
		fs.BoolVar(&s.AnyToken.Allow, "insecure-allow-any-token", s.AnyToken.Allow, ""+
			"If set, your server will be INSECURE.  Any token will be allowed and user information will be parsed "+
			"from the token as `username/group1,group2`")

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

		fs.StringVar(&s.OIDC.GroupsClaim, "oidc-groups-claim", "", ""+
			"If provided, the name of a custom OpenID Connect claim for specifying user groups. "+
			"The claim value is expected to be a string or array of strings. This flag is experimental, "+
			"please see the authentication documentation for further details.")
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
			"The duration to cache responses from the webhook token authenticator. Default is 2m.")
	}
}

func (s *BuiltInAuthenticationOptions) ToAuthenticationConfig() authenticator.AuthenticatorConfig {
	ret := authenticator.AuthenticatorConfig{}

	if s.Anonymous != nil {
		ret.Anonymous = s.Anonymous.Allow
	}

	if s.AnyToken != nil {
		ret.AnyToken = s.AnyToken.Allow
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
	}

	return ret
}

func (o *BuiltInAuthenticationOptions) Apply(c *genericapiserver.Config) error {
	if o == nil || o.PasswordFile == nil {
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

	c.SupportsBasicAuth = len(o.PasswordFile.BasicAuthFile) > 0
	return nil
}
