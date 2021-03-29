package helpers

import (
	"strings"

	configv1 "github.com/openshift/api/config/v1"
)

func GetHTTPServingInfoFileReferences(config *configv1.HTTPServingInfo) []*string {
	if config == nil {
		return []*string{}
	}

	return GetServingInfoFileReferences(&config.ServingInfo)
}

func GetServingInfoFileReferences(config *configv1.ServingInfo) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, GetCertFileReferences(&config.CertInfo)...)
	refs = append(refs, &config.ClientCA)
	for i := range config.NamedCertificates {
		refs = append(refs, &config.NamedCertificates[i].CertFile)
		refs = append(refs, &config.NamedCertificates[i].KeyFile)
	}

	return refs
}

func GetCertFileReferences(config *configv1.CertInfo) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, &config.CertFile)
	refs = append(refs, &config.KeyFile)
	return refs
}

func GetRemoteConnectionInfoFileReferences(config *configv1.RemoteConnectionInfo) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, GetCertFileReferences(&config.CertInfo)...)
	refs = append(refs, &config.CA)
	return refs
}

func GetEtcdConnectionInfoFileReferences(config *configv1.EtcdConnectionInfo) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, GetCertFileReferences(&config.CertInfo)...)
	refs = append(refs, &config.CA)
	return refs
}

func GetStringSourceFileReferences(s *configv1.StringSource) []*string {
	if s == nil {
		return []*string{}
	}

	return []*string{
		&s.File,
		&s.KeyFile,
	}
}

func GetAdmissionPluginConfigFileReferences(config *configv1.AdmissionPluginConfig) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, &config.Location)
	return refs
}

func GetAuditConfigFileReferences(config *configv1.AuditConfig) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, &config.PolicyFile)
	refs = append(refs, &config.AuditFilePath)
	return refs
}

func GetKubeClientConfigFileReferences(config *configv1.KubeClientConfig) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, &config.KubeConfig)
	return refs
}

func GetGenericAPIServerConfigFileReferences(config *configv1.GenericAPIServerConfig) []*string {
	if config == nil {
		return []*string{}
	}

	refs := []*string{}
	refs = append(refs, GetHTTPServingInfoFileReferences(&config.ServingInfo)...)
	refs = append(refs, GetEtcdConnectionInfoFileReferences(&config.StorageConfig.EtcdConnectionInfo)...)
	refs = append(refs, GetAuditConfigFileReferences(&config.AuditConfig)...)
	refs = append(refs, GetKubeClientConfigFileReferences(&config.KubeClientConfig)...)

	// TODO admission config file resolution is currently broken.
	//for k := range config.AdmissionPluginConfig {
	//	refs = append(refs, GetAdmissionPluginConfigReferences(&(config.AdmissionPluginConfig[k]))...)
	//}
	return refs
}

func GetFlagsWithFileExtensionsFileReferences(args map[string][]string) []*string {
	if args == nil {
		return []*string{}
	}

	refs := []*string{}
	for key, s := range args {
		if len(s) == 0 {
			continue
		}
		if !strings.HasSuffix(key, "-file") && !strings.HasSuffix(key, "-dir") {
			continue
		}
		for i := range s {
			refs = append(refs, &s[i])
		}
	}

	return refs
}
