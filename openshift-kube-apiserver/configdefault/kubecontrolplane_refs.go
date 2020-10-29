package configdefault

import (
	kubecontrolplanev1 "github.com/openshift/api/kubecontrolplane/v1"
	osinv1 "github.com/openshift/api/osin/v1"
	"github.com/openshift/library-go/pkg/config/helpers"
)

func GetKubeAPIServerConfigFileReferences(config *kubecontrolplanev1.KubeAPIServerConfig) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}

	refs = append(refs, helpers.GetGenericAPIServerConfigFileReferences(&config.GenericAPIServerConfig)...)
	refs = append(refs, GetKubeletConnectionInfoFileReferences(&config.KubeletClientInfo)...)

	if config.OAuthConfig != nil {
		refs = append(refs, GetOAuthConfigFileReferences(config.OAuthConfig)...)
	}

	refs = append(refs, &config.AggregatorConfig.ProxyClientInfo.CertFile)
	refs = append(refs, &config.AggregatorConfig.ProxyClientInfo.KeyFile)

	if config.AuthConfig.RequestHeader != nil {
		refs = append(refs, &config.AuthConfig.RequestHeader.ClientCA)
	}
	for k := range config.AuthConfig.WebhookTokenAuthenticators {
		refs = append(refs, &config.AuthConfig.WebhookTokenAuthenticators[k].ConfigFile)
	}
	if len(config.AuthConfig.OAuthMetadataFile) > 0 {
		refs = append(refs, &config.AuthConfig.OAuthMetadataFile)
	}

	refs = append(refs, &config.AggregatorConfig.ProxyClientInfo.CertFile)
	refs = append(refs, &config.AggregatorConfig.ProxyClientInfo.KeyFile)

	for i := range config.ServiceAccountPublicKeyFiles {
		refs = append(refs, &config.ServiceAccountPublicKeyFiles[i])
	}

	return refs
}

func GetKubeletConnectionInfoFileReferences(config *kubecontrolplanev1.KubeletConnectionInfo) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, helpers.GetCertFileReferences(&config.CertInfo)...)
	refs = append(refs, &config.CA)
	return refs
}

func GetOAuthConfigFileReferences(config *osinv1.OAuthConfig) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}

	if config.MasterCA != nil {
		refs = append(refs, config.MasterCA)
	}

	refs = append(refs, GetSessionConfigFileReferences(config.SessionConfig)...)
	for _, identityProvider := range config.IdentityProviders {
		switch provider := identityProvider.Provider.Object.(type) {
		case (*osinv1.RequestHeaderIdentityProvider):
			refs = append(refs, &provider.ClientCA)

		case (*osinv1.HTPasswdPasswordIdentityProvider):
			refs = append(refs, &provider.File)

		case (*osinv1.LDAPPasswordIdentityProvider):
			refs = append(refs, &provider.CA)
			refs = append(refs, helpers.GetStringSourceFileReferences(&provider.BindPassword)...)

		case (*osinv1.BasicAuthPasswordIdentityProvider):
			refs = append(refs, helpers.GetRemoteConnectionInfoFileReferences(&provider.RemoteConnectionInfo)...)

		case (*osinv1.KeystonePasswordIdentityProvider):
			refs = append(refs, helpers.GetRemoteConnectionInfoFileReferences(&provider.RemoteConnectionInfo)...)

		case (*osinv1.GitLabIdentityProvider):
			refs = append(refs, &provider.CA)
			refs = append(refs, helpers.GetStringSourceFileReferences(&provider.ClientSecret)...)

		case (*osinv1.OpenIDIdentityProvider):
			refs = append(refs, &provider.CA)
			refs = append(refs, helpers.GetStringSourceFileReferences(&provider.ClientSecret)...)

		case (*osinv1.GoogleIdentityProvider):
			refs = append(refs, helpers.GetStringSourceFileReferences(&provider.ClientSecret)...)

		case (*osinv1.GitHubIdentityProvider):
			refs = append(refs, helpers.GetStringSourceFileReferences(&provider.ClientSecret)...)
			refs = append(refs, &provider.CA)

		}
	}

	if config.Templates != nil {
		refs = append(refs, &config.Templates.Login)
		refs = append(refs, &config.Templates.ProviderSelection)
		refs = append(refs, &config.Templates.Error)
	}

	return refs
}

func GetSessionConfigFileReferences(config *osinv1.SessionConfig) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, &config.SessionSecretsFile)
	return refs
}
