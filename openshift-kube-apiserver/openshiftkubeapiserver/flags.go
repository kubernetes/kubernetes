package openshiftkubeapiserver

import (
	"fmt"
	"io/ioutil"
	"net"
	"strings"

	configv1 "github.com/openshift/api/config/v1"
	kubecontrolplanev1 "github.com/openshift/api/kubecontrolplane/v1"
	"github.com/openshift/apiserver-library-go/pkg/configflags"
	"github.com/openshift/library-go/pkg/config/helpers"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	apiserverv1alpha1 "k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
)

func ConfigToFlags(kubeAPIServerConfig *kubecontrolplanev1.KubeAPIServerConfig) ([]string, error) {
	args := unmaskArgs(kubeAPIServerConfig.APIServerArguments)

	host, portString, err := net.SplitHostPort(kubeAPIServerConfig.ServingInfo.BindAddress)
	if err != nil {
		return nil, err
	}

	admissionFlags, err := admissionFlags(kubeAPIServerConfig.AdmissionConfig)
	if err != nil {
		return nil, err
	}
	for flag, value := range admissionFlags {
		configflags.SetIfUnset(args, flag, value...)
	}
	for flag, value := range configflags.AuditFlags(&kubeAPIServerConfig.AuditConfig, configflags.ArgsWithPrefix(args, "audit-")) {
		configflags.SetIfUnset(args, flag, value...)
	}
	configflags.SetIfUnset(args, "bind-address", host)
	configflags.SetIfUnset(args, "cors-allowed-origins", kubeAPIServerConfig.CORSAllowedOrigins...)
	configflags.SetIfUnset(args, "secure-port", portString)
	configflags.SetIfUnset(args, "service-account-key-file", kubeAPIServerConfig.ServiceAccountPublicKeyFiles...)
	configflags.SetIfUnset(args, "service-cluster-ip-range", kubeAPIServerConfig.ServicesSubnet)
	configflags.SetIfUnset(args, "tls-cipher-suites", kubeAPIServerConfig.ServingInfo.CipherSuites...)
	configflags.SetIfUnset(args, "tls-min-version", kubeAPIServerConfig.ServingInfo.MinTLSVersion)
	configflags.SetIfUnset(args, "tls-sni-cert-key", sniCertKeys(kubeAPIServerConfig.ServingInfo.NamedCertificates)...)

	return configflags.ToFlagSlice(args), nil
}

func admissionFlags(admissionConfig configv1.AdmissionConfig) (map[string][]string, error) {
	args := map[string][]string{}

	upstreamAdmissionConfig, err := ConvertOpenshiftAdmissionConfigToKubeAdmissionConfig(admissionConfig.PluginConfig)
	if err != nil {
		return nil, err
	}
	configBytes, err := helpers.WriteYAML(upstreamAdmissionConfig, apiserverv1alpha1.AddToScheme)
	if err != nil {
		return nil, err
	}

	tempFile, err := ioutil.TempFile("", "kubeapiserver-admission-config.yaml")
	if err != nil {
		return nil, err
	}
	if _, err := tempFile.Write(configBytes); err != nil {
		return nil, err
	}
	tempFile.Close()

	configflags.SetIfUnset(args, "admission-control-config-file", tempFile.Name())

	return args, nil
}

func sniCertKeys(namedCertificates []configv1.NamedCertificate) []string {
	args := []string{}
	for _, nc := range namedCertificates {
		names := ""
		if len(nc.Names) > 0 {
			names = ":" + strings.Join(nc.Names, ",")
		}
		args = append(args, fmt.Sprintf("%s,%s%s", nc.CertFile, nc.KeyFile, names))
	}
	return args
}

func unmaskArgs(args map[string]kubecontrolplanev1.Arguments) map[string][]string {
	ret := map[string][]string{}
	for key, slice := range args {
		for _, val := range slice {
			ret[key] = append(ret[key], val)
		}
	}
	return ret
}

func ConvertOpenshiftAdmissionConfigToKubeAdmissionConfig(in map[string]configv1.AdmissionPluginConfig) (*apiserverv1alpha1.AdmissionConfiguration, error) {
	ret := &apiserverv1alpha1.AdmissionConfiguration{}

	for _, pluginName := range sets.StringKeySet(in).List() {
		kubeConfig := apiserverv1alpha1.AdmissionPluginConfiguration{
			Name: pluginName,
			Path: in[pluginName].Location,
			Configuration: &runtime.Unknown{
				Raw: in[pluginName].Configuration.Raw,
			},
		}

		ret.Plugins = append(ret.Plugins, kubeConfig)
	}

	return ret, nil
}
