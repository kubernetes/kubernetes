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
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/spf13/pflag"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	genericapiserver "k8s.io/apiserver/pkg/server"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/kubernetes/pkg/features"
	kubeauthenticator "k8s.io/kubernetes/pkg/kubeapiserver/authenticator"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
)

type BuiltInAuthenticationOptions struct {
	APIAudiences    []string
	Anonymous       *AnonymousAuthenticationOptions
	BootstrapToken  *BootstrapTokenAuthenticationOptions
	ClientCert      *genericoptions.ClientCertAuthenticationOptions
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

type OIDCAuthenticationOptions struct {
	CAFile         string
	ClientID       string
	IssuerURL      string
	UsernameClaim  string
	UsernamePrefix string
	GroupsClaim    string
	GroupsPrefix   string
	SigningAlgs    []string
	RequiredClaims map[string]string
}

type PasswordFileAuthenticationOptions struct {
	BasicAuthFile string
}

type ServiceAccountAuthenticationOptions struct {
	KeyFiles      []string
	Lookup        bool
	Issuer        string
	MaxExpiration time.Duration
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

	if s.ServiceAccounts != nil && len(s.ServiceAccounts.Issuer) > 0 && strings.Contains(s.ServiceAccounts.Issuer, ":") {
		if _, err := url.Parse(s.ServiceAccounts.Issuer); err != nil {
			allErrors = append(allErrors, fmt.Errorf("service-account-issuer contained a ':' but was not a valid URL: %v", err))
		}
	}
	if s.ServiceAccounts != nil && utilfeature.DefaultFeatureGate.Enabled(features.BoundServiceAccountTokenVolume) {
		if !utilfeature.DefaultFeatureGate.Enabled(features.TokenRequest) || !utilfeature.DefaultFeatureGate.Enabled(features.TokenRequestProjection) {
			allErrors = append(allErrors, errors.New("If the BoundServiceAccountTokenVolume feature is enabled,"+
				" the TokenRequest and TokenRequestProjection features must also be enabled"))
		}
		if len(s.ServiceAccounts.Issuer) == 0 {
			allErrors = append(allErrors, errors.New("service-account-issuer is a required flag when BoundServiceAccountTokenVolume is enabled"))
		}
		if len(s.ServiceAccounts.KeyFiles) == 0 {
			allErrors = append(allErrors, errors.New("service-account-key-file is a required flag when BoundServiceAccountTokenVolume is enabled"))
		}
	}

	return allErrors
}

func (s *BuiltInAuthenticationOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringSliceVar(&s.APIAudiences, "api-audiences", s.APIAudiences, ""+
		"Identifiers of the API. The service account token authenticator will validate that "+
		"tokens used against the API are bound to at least one of these audiences. If the "+
		"--service-account-issuer flag is configured and this flag is not, this field "+
		"defaults to a single element list containing the issuer URL .")

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

		fs.StringSliceVar(&s.OIDC.SigningAlgs, "oidc-signing-algs", []string{"RS256"}, ""+
			"Comma-separated list of allowed JOSE asymmetric signing algorithms. JWTs with a "+
			"'alg' header value not in this list will be rejected. "+
			"Values are defined by RFC 7518 https://tools.ietf.org/html/rfc7518#section-3.1.")

		fs.Var(cliflag.NewMapStringStringNoSplit(&s.OIDC.RequiredClaims), "oidc-required-claim", ""+
			"A key=value pair that describes a required claim in the ID Token. "+
			"If set, the claim is verified to be present in the ID Token with a matching value. "+
			"Repeat this flag to specify multiple claims.")
	}

	if s.PasswordFile != nil {
		fs.StringVar(&s.PasswordFile.BasicAuthFile, "basic-auth-file", s.PasswordFile.BasicAuthFile, ""+
			"If set, the file that will be used to admit requests to the secure port of the API server "+
			"via http basic authentication.")
		fs.MarkDeprecated("basic-auth-file", "Basic authentication mode is deprecated and will be removed in a future release. It is not recommended for production environments.")
	}

	if s.RequestHeader != nil {
		s.RequestHeader.AddFlags(fs)
	}

	if s.ServiceAccounts != nil {
		fs.StringArrayVar(&s.ServiceAccounts.KeyFiles, "service-account-key-file", s.ServiceAccounts.KeyFiles, ""+
			"File containing PEM-encoded x509 RSA or ECDSA private or public keys, used to verify "+
			"ServiceAccount tokens. The specified file can contain multiple keys, and the flag can "+
			"be specified multiple times with different files. If unspecified, "+
			"--tls-private-key-file is used. Must be specified when "+
			"--service-account-signing-key is provided")

		fs.BoolVar(&s.ServiceAccounts.Lookup, "service-account-lookup", s.ServiceAccounts.Lookup,
			"If true, validate ServiceAccount tokens exist in etcd as part of authentication.")

		fs.StringVar(&s.ServiceAccounts.Issuer, "service-account-issuer", s.ServiceAccounts.Issuer, ""+
			"Identifier of the service account token issuer. The issuer will assert this identifier "+
			"in \"iss\" claim of issued tokens. This value is a string or URI.")

		// Deprecated in 1.13
		fs.StringSliceVar(&s.APIAudiences, "service-account-api-audiences", s.APIAudiences, ""+
			"Identifiers of the API. The service account token authenticator will validate that "+
			"tokens used against the API are bound to at least one of these audiences.")
		fs.MarkDeprecated("service-account-api-audiences", "Use --api-audiences")

		fs.DurationVar(&s.ServiceAccounts.MaxExpiration, "service-account-max-token-expiration", s.ServiceAccounts.MaxExpiration, ""+
			"The maximum validity duration of a token created by the service account token issuer. If an otherwise valid "+
			"TokenRequest with a validity duration larger than this value is requested, a token will be issued with a validity duration of this value.")
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

func (s *BuiltInAuthenticationOptions) ToAuthenticationConfig() kubeauthenticator.Config {
	ret := kubeauthenticator.Config{
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

	if s.OIDC != nil {
		ret.OIDCCAFile = s.OIDC.CAFile
		ret.OIDCClientID = s.OIDC.ClientID
		ret.OIDCGroupsClaim = s.OIDC.GroupsClaim
		ret.OIDCGroupsPrefix = s.OIDC.GroupsPrefix
		ret.OIDCIssuerURL = s.OIDC.IssuerURL
		ret.OIDCUsernameClaim = s.OIDC.UsernameClaim
		ret.OIDCUsernamePrefix = s.OIDC.UsernamePrefix
		ret.OIDCSigningAlgs = s.OIDC.SigningAlgs
		ret.OIDCRequiredClaims = s.OIDC.RequiredClaims
	}

	if s.PasswordFile != nil {
		ret.BasicAuthFile = s.PasswordFile.BasicAuthFile
	}

	if s.RequestHeader != nil {
		ret.RequestHeaderConfig = s.RequestHeader.ToAuthenticationRequestHeaderConfig()
	}

	ret.APIAudiences = s.APIAudiences
	if s.ServiceAccounts != nil {
		if s.ServiceAccounts.Issuer != "" && len(s.APIAudiences) == 0 {
			ret.APIAudiences = authenticator.Audiences{s.ServiceAccounts.Issuer}
		}
		ret.ServiceAccountKeyFiles = s.ServiceAccounts.KeyFiles
		ret.ServiceAccountIssuer = s.ServiceAccounts.Issuer
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
				klog.Warningf("the webhook cache ttl of %s is shorter than the overall cache ttl of %s for successful token authentication attempts.", s.WebHook.CacheTTL, s.TokenSuccessCacheTTL)
			}
			if s.TokenFailureCacheTTL > 0 && s.WebHook.CacheTTL < s.TokenFailureCacheTTL {
				klog.Warningf("the webhook cache ttl of %s is shorter than the overall cache ttl of %s for failed token authentication attempts.", s.WebHook.CacheTTL, s.TokenFailureCacheTTL)
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
		if err = c.Authentication.ApplyClientCert(o.ClientCert.ClientCA, c.SecureServing); err != nil {
			return fmt.Errorf("unable to load client CA file: %v", err)
		}
	}
	if o.RequestHeader != nil {
		if err = c.Authentication.ApplyClientCert(o.RequestHeader.ClientCAFile, c.SecureServing); err != nil {
			return fmt.Errorf("unable to load client CA file: %v", err)
		}
	}

	c.Authentication.SupportsBasicAuth = o.PasswordFile != nil && len(o.PasswordFile.BasicAuthFile) > 0

	c.Authentication.APIAudiences = o.APIAudiences
	if o.ServiceAccounts != nil && o.ServiceAccounts.Issuer != "" && len(o.APIAudiences) == 0 {
		c.Authentication.APIAudiences = authenticator.Audiences{o.ServiceAccounts.Issuer}
	}

	return nil
}

// ApplyAuthorization will conditionally modify the authentication options based on the authorization options
func (o *BuiltInAuthenticationOptions) ApplyAuthorization(authorization *BuiltInAuthorizationOptions) {
	if o == nil || authorization == nil || o.Anonymous == nil {
		return
	}

	// authorization ModeAlwaysAllow cannot be combined with AnonymousAuth.
	// in such a case the AnonymousAuth is stomped to false and you get a message
	if o.Anonymous.Allow && sets.NewString(authorization.Modes...).Has(authzmodes.ModeAlwaysAllow) {
		klog.Warningf("AnonymousAuth is not allowed with the AlwaysAllow authorizer. Resetting AnonymousAuth to false. You should use a different authorizer")
		o.Anonymous.Allow = false
	}
}
