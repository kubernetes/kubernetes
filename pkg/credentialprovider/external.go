/*
Copyright 2020 The Kubernetes Authors.

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

package credentialprovider

import (
	"bytes"
	"fmt"
	"net/url"
	"os/exec"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/credentialprovider/apis/registrycredentials"
	"k8s.io/kubernetes/pkg/credentialprovider/apis/registrycredentials/v1alpha1"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	registrycredentials.AddToScheme(scheme)
	v1alpha1.AddToScheme(scheme)
}

type externalProviderKeyring struct {
	// map of registry matcher to provider (each provider may have multiple)
	providers map[string]ExternalCredentialProvider
	// reverse sorted index of registries
	index []string
}

func NewExternalProviderKeyring(configPath string) (DockerKeyring, error) {
	keyring := &externalProviderKeyring{
		providers: make(map[string]ExternalCredentialProvider),
		index:     make([]string, 0),
	}

	config, err := readExternalProviderConfig(configPath)
	if err != nil {
		return keyring, err
	}

	for _, p := range config.Providers {
		provider := externalCredentialProvider{
			command: p.Exec.Command,
			args:    p.Exec.Args,
			env:     convertEnvs(p.Exec.Env),
		}
		for _, m := range p.ImageMatchers {

			value := m
			if !strings.HasPrefix(value, "https://") && !strings.HasPrefix(value, "http://") {
				value = "https://" + value
			}
			parsed, err := url.Parse(value)
			if err != nil {
				klog.Errorf("Entry %q in dockercfg invalid (%v), ignoring", m, err)
				continue
			}

			// The docker client allows exact matches:
			//    foo.bar.com/namespace
			// Or hostname matches:
			//    foo.bar.com
			// It also considers /v2/  and /v1/ equivalent to the hostname
			// See ResolveAuthConfig in docker/registry/auth.go.
			effectivePath := parsed.Path
			if strings.HasPrefix(effectivePath, "/v2/") || strings.HasPrefix(effectivePath, "/v1/") {
				effectivePath = effectivePath[3:]
			}
			var key string
			if (len(effectivePath) > 0) && (effectivePath != "/") {
				key = parsed.Host + effectivePath
			} else {
				key = parsed.Host
			}

			keyring.providers[key] = &provider
			keyring.index = append(keyring.index, key)
		}
	}

	eliminateDupes := sets.NewString(keyring.index...)
	keyring.index = eliminateDupes.List()

	// Update the index used to identify which credentials to use for a given
	// image. The index is reverse-sorted so more specific paths are matched
	// first. For example, if for the given image "quay.io/coreos/etcd",
	// credentials for "quay.io/coreos" should match before "quay.io".
	sort.Sort(sort.Reverse(sort.StringSlice(keyring.index)))

	return keyring, nil
}

func (e *externalProviderKeyring) Lookup(image string) ([]AuthConfig, bool) {
	var provider ExternalCredentialProvider
	klog.V(2).Infof("Looking up %s with external provider.", image)

	// range over the index as iterating over a map does not provide a predictable ordering
	for _, matcher := range e.index {
		// both k and image are schemeless URLs because even though schemes are allowed
		// in the credential configurations, we remove them in Add.
		if matched, _ := urlsMatchStr(matcher, image); matched {
			provider = e.providers[matcher]
			break
		}
	}

	if provider == nil {
		klog.Infof("No external provider available for the image: %s", image)
		return []AuthConfig{}, false
	}

	cfg := provider.Provide(image)

	providedConfigs := make([]AuthConfig, 0)
	for _, ident := range cfg {
		creds := AuthConfig{
			Username: ident.Username,
			Password: ident.Password,
			Email:    ident.Email,
		}
		providedConfigs = append(providedConfigs, creds)
	}
	return providedConfigs, (len(providedConfigs) > 0)
}

type ExternalCredentialProvider interface {
	Provide(image string) DockerConfig
}

type externalCredentialProvider struct {
	command string
	args    []string

	// Each environment variable is in the form key=value
	env []string
}

func (e *externalCredentialProvider) Provide(image string) DockerConfig {
	cfg := make(DockerConfig)

	response, err := e.request(image)
	if err != nil {
		klog.Errorf("Failed getting credential from external registry credential provider: %v", err)
		return cfg
	}

	cfg[image] = DockerConfigEntry{
		Username: *response.Username,
		Password: *response.Password,
	}
	return cfg
}

func newRegistryCredentialEncoder(targetVersion schema.GroupVersion) (runtime.Encoder, error) {
	mediaType := "application/json"
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unsupported media type %q", mediaType)
	}
	return codecs.EncoderForVersion(info.Serializer, targetVersion), nil
}

func (e *externalCredentialProvider) request(image string) (*registrycredentials.RegistryCredentialPluginResponse, error) {
	var stdout, stderr, stdin bytes.Buffer
	cmd := exec.Command(e.command, e.args...)
	cmd.Env = e.env
	cmd.Stdout, cmd.Stderr, cmd.Stdin = &stdout, &stderr, &stdin

	encoder, err := newRegistryCredentialEncoder(v1alpha1.SchemeGroupVersion)
	if err != nil {
		return nil, fmt.Errorf("Failed to set up YAML encoder: %v", err)
	}

	err = encoder.Encode(&v1alpha1.RegistryCredentialPluginRequest{Image: image}, &stdin)
	if err != nil {
		return nil, fmt.Errorf("Failed to encode the image %s as a request to the registry credential provider: %v", image, err)
	}

	err = cmd.Run()
	if err != nil {
		klog.V(2).Infof("Registry credential provider binary stderr:\n%s", stderr.String())
		return nil, fmt.Errorf("Error execing external provider %s providing registry credentials for %s: %v\n", e.command, image, err)
	}

	obj, gvk, err := codecs.UniversalDecoder(v1alpha1.SchemeGroupVersion).Decode(stdout.Bytes(), nil, nil)
	if err != nil {
		return nil, err
	}

	if gvk.Kind != "RegistryCredentialPluginResponse" {
		return nil, fmt.Errorf("failed to decode %q (missing Kind)", gvk.Kind)
	}
	config, err := scheme.ConvertToVersion(obj, registrycredentials.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}
	if internalResponse, ok := config.(*registrycredentials.RegistryCredentialPluginResponse); ok {
		return internalResponse, nil
	}
	return nil, fmt.Errorf("unable to convert %T to *RegistryCredentialPluginResponse", config)
}

func convertEnvs(envs []registrycredentials.ExecEnvVar) []string {
	s := []string{}
	for _, e := range envs {
		s = append(s, fmt.Sprintf("%s=%s", e.Name, e.Value))
	}
	return s
}
