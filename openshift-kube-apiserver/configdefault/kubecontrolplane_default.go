package configdefault

import (
	"io/ioutil"
	"os"
	"path/filepath"

	kubecontrolplanev1 "github.com/openshift/api/kubecontrolplane/v1"
	"github.com/openshift/library-go/pkg/config/configdefaults"
	"k8s.io/klog/v2"
)

// ResolveDirectoriesForSATokenVerification takes our config (which allows directories) and navigates one level of
// those directories for files.  This makes it easy to build a single configmap that contains lots of aggregated files.
// if we fail to open the file for inspection, the resolving code in kube-apiserver may have drifted from us
// we include the raw file and let the kube-apiserver succeed or fail.
func ResolveDirectoriesForSATokenVerification(config *kubecontrolplanev1.KubeAPIServerConfig) {
	// kube doesn't honor directories, but we want to allow them in our sa token validators
	resolvedSATokenValidationCerts := []string{}
	for _, filename := range config.ServiceAccountPublicKeyFiles {
		file, err := os.Open(filename)
		if err != nil {
			resolvedSATokenValidationCerts = append(resolvedSATokenValidationCerts, filename)
			klog.Warningf(err.Error())
			continue
		}
		fileInfo, err := file.Stat()
		if err != nil {
			resolvedSATokenValidationCerts = append(resolvedSATokenValidationCerts, filename)
			klog.Warningf(err.Error())
			continue
		}
		if !fileInfo.IsDir() {
			resolvedSATokenValidationCerts = append(resolvedSATokenValidationCerts, filename)
			continue
		}

		contents, err := ioutil.ReadDir(filename)
		switch {
		case os.IsNotExist(err) || os.IsPermission(err):
			klog.Warningf(err.Error())
		case err != nil:
			panic(err) // some weird, unexpected error
		default:
			for _, content := range contents {
				if !content.Mode().IsRegular() {
					continue
				}
				resolvedSATokenValidationCerts = append(resolvedSATokenValidationCerts, filepath.Join(filename, content.Name()))
			}
		}
	}

	config.ServiceAccountPublicKeyFiles = resolvedSATokenValidationCerts
}

func SetRecommendedKubeAPIServerConfigDefaults(config *kubecontrolplanev1.KubeAPIServerConfig) {
	configdefaults.DefaultString(&config.GenericAPIServerConfig.StorageConfig.StoragePrefix, "kubernetes.io")
	configdefaults.DefaultString(&config.GenericAPIServerConfig.ServingInfo.BindAddress, "0.0.0.0:6443")

	configdefaults.SetRecommendedGenericAPIServerConfigDefaults(&config.GenericAPIServerConfig)
	SetRecommendedMasterAuthConfigDefaults(&config.AuthConfig)
	SetRecommendedAggregatorConfigDefaults(&config.AggregatorConfig)
	SetRecommendedKubeletConnectionInfoDefaults(&config.KubeletClientInfo)

	configdefaults.DefaultString(&config.ServicesSubnet, "10.0.0.0/24")
	configdefaults.DefaultString(&config.ServicesNodePortRange, "30000-32767")

	if len(config.ServiceAccountPublicKeyFiles) == 0 {
		config.ServiceAccountPublicKeyFiles = append([]string{}, "/etc/kubernetes/static-pod-resources/configmaps/sa-token-signing-certs")
	}

	// after the aggregator defaults are set, we can default the auth config values
	// TODO this indicates that we're set two different things to the same value
	if config.AuthConfig.RequestHeader == nil {
		config.AuthConfig.RequestHeader = &kubecontrolplanev1.RequestHeaderAuthenticationOptions{}
		configdefaults.DefaultStringSlice(&config.AuthConfig.RequestHeader.ClientCommonNames, []string{"system:openshift-aggregator"})
		configdefaults.DefaultString(&config.AuthConfig.RequestHeader.ClientCA, "/var/run/configmaps/aggregator-client-ca/ca-bundle.crt")
		configdefaults.DefaultStringSlice(&config.AuthConfig.RequestHeader.UsernameHeaders, []string{"X-Remote-User"})
		configdefaults.DefaultStringSlice(&config.AuthConfig.RequestHeader.GroupHeaders, []string{"X-Remote-Group"})
		configdefaults.DefaultStringSlice(&config.AuthConfig.RequestHeader.ExtraHeaderPrefixes, []string{"X-Remote-Extra-"})
	}

	// Set defaults Cache TTLs for external Webhook Token Reviewers
	for i := range config.AuthConfig.WebhookTokenAuthenticators {
		if len(config.AuthConfig.WebhookTokenAuthenticators[i].CacheTTL) == 0 {
			config.AuthConfig.WebhookTokenAuthenticators[i].CacheTTL = "2m"
		}
	}

	if config.OAuthConfig != nil {
		for i := range config.OAuthConfig.IdentityProviders {
			// By default, only let one identity provider authenticate a particular user
			// If multiple identity providers collide, the second one in will fail to auth
			// The admin can set this to "add" if they want to allow new identities to join existing users
			configdefaults.DefaultString(&config.OAuthConfig.IdentityProviders[i].MappingMethod, "claim")
		}
	}
}

func SetRecommendedMasterAuthConfigDefaults(config *kubecontrolplanev1.MasterAuthConfig) {
}

func SetRecommendedAggregatorConfigDefaults(config *kubecontrolplanev1.AggregatorConfig) {
	configdefaults.DefaultString(&config.ProxyClientInfo.KeyFile, "/var/run/secrets/aggregator-client/tls.key")
	configdefaults.DefaultString(&config.ProxyClientInfo.CertFile, "/var/run/secrets/aggregator-client/tls.crt")
}

func SetRecommendedKubeletConnectionInfoDefaults(config *kubecontrolplanev1.KubeletConnectionInfo) {
	if config.Port == 0 {
		config.Port = 10250
	}
	configdefaults.DefaultString(&config.CertInfo.KeyFile, "/var/run/secrets/kubelet-client/tls.key")
	configdefaults.DefaultString(&config.CertInfo.CertFile, "/var/run/secrets/kubelet-client/tls.crt")
}
