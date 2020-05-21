package configdefaults

import (
	"time"

	configv1 "github.com/openshift/api/config/v1"
	"github.com/openshift/library-go/pkg/crypto"
)

func DefaultString(target *string, defaultVal string) {
	if len(*target) == 0 {
		*target = defaultVal
	}
}

func DefaultInt(target *int, defaultVal int) {
	if *target == 0 {
		*target = defaultVal
	}
}

func DefaultMetaDuration(target *time.Duration, defaultVal time.Duration) {
	if *target == 0 {
		*target = defaultVal
	}
}

func DefaultStringSlice(target *[]string, defaultVal []string) {
	if len(*target) == 0 {
		*target = defaultVal
	}
}

func SetRecommendedHTTPServingInfoDefaults(config *configv1.HTTPServingInfo) {
	if config.MaxRequestsInFlight == 0 {
		config.MaxRequestsInFlight = 3000
	}
	if config.RequestTimeoutSeconds == 0 {
		config.RequestTimeoutSeconds = 60 * 60 // one hour
	}

	SetRecommendedServingInfoDefaults(&config.ServingInfo)
}

func SetRecommendedServingInfoDefaults(config *configv1.ServingInfo) {
	DefaultString(&config.BindAddress, "0.0.0.0:8443")
	DefaultString(&config.BindNetwork, "tcp")
	DefaultString(&config.CertInfo.KeyFile, "/var/run/secrets/serving-cert/tls.key")
	DefaultString(&config.CertInfo.CertFile, "/var/run/secrets/serving-cert/tls.crt")
	DefaultString(&config.ClientCA, "/var/run/configmaps/client-ca/ca-bundle.crt")
	DefaultString(&config.MinTLSVersion, crypto.TLSVersionToNameOrDie(crypto.DefaultTLSVersion()))

	if len(config.CipherSuites) == 0 {
		config.CipherSuites = crypto.CipherSuitesToNamesOrDie(crypto.DefaultCiphers())
	}
}

func SetRecommendedGenericAPIServerConfigDefaults(config *configv1.GenericAPIServerConfig) {
	SetRecommendedHTTPServingInfoDefaults(&config.ServingInfo)
	SetRecommendedEtcdConnectionInfoDefaults(&config.StorageConfig.EtcdConnectionInfo)
	SetRecommendedKubeClientConfigDefaults(&config.KubeClientConfig)
}

func SetRecommendedEtcdConnectionInfoDefaults(config *configv1.EtcdConnectionInfo) {
	DefaultStringSlice(&config.URLs, []string{"https://etcd.kube-system.svc:2379"})
	DefaultString(&config.CertInfo.KeyFile, "/var/run/secrets/etcd-client/tls.key")
	DefaultString(&config.CertInfo.CertFile, "/var/run/secrets/etcd-client/tls.crt")
	DefaultString(&config.CA, "/var/run/configmaps/etcd-serving-ca/ca-bundle.crt")
}

func SetRecommendedKubeClientConfigDefaults(config *configv1.KubeClientConfig) {
	// these are historical values
	if config.ConnectionOverrides.QPS <= 0 {
		config.ConnectionOverrides.QPS = 150.0
	}
	if config.ConnectionOverrides.Burst <= 0 {
		config.ConnectionOverrides.Burst = 300
	}
	DefaultString(&config.ConnectionOverrides.AcceptContentTypes, "application/vnd.kubernetes.protobuf,application/json")
	DefaultString(&config.ConnectionOverrides.ContentType, "application/vnd.kubernetes.protobuf")
}
