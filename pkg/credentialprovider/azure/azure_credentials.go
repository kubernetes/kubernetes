//go:build !providerless
// +build !providerless

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

package azure

import (
	"context"
	"errors"
	"io"
	"io/ioutil"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry"
	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/spf13/pflag"

	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/legacy-cloud-providers/azure/auth"
	"sigs.k8s.io/yaml"
)

var flagConfigFile = pflag.String("azure-container-registry-config", "",
	"Path to the file containing Azure container registry configuration information.")

const (
	dummyRegistryEmail = "name@contoso.com"
	maxReadLength      = 10 * 1 << 20 // 10MB
)

var (
	containerRegistryUrls = []string{"*.azurecr.io", "*.azurecr.cn", "*.azurecr.de", "*.azurecr.us"}
	acrRE                 = regexp.MustCompile(`.*\.azurecr\.io|.*\.azurecr\.cn|.*\.azurecr\.de|.*\.azurecr\.us`)
	warnOnce              sync.Once
)

// init registers the various means by which credentials may
// be resolved on Azure.
func init() {
	credentialprovider.RegisterCredentialProvider(
		"azure",
		NewACRProvider(flagConfigFile),
	)
}

type cacheEntry struct {
	expiresAt   time.Time
	credentials credentialprovider.DockerConfigEntry
	registry    string
}

// acrExpirationPolicy implements ExpirationPolicy from client-go.
type acrExpirationPolicy struct{}

// stringKeyFunc returns the cache key as a string
func stringKeyFunc(obj interface{}) (string, error) {
	key := obj.(*cacheEntry).registry
	return key, nil
}

// IsExpired checks if the ACR credentials are expired.
func (p *acrExpirationPolicy) IsExpired(entry *cache.TimestampedEntry) bool {
	return time.Now().After(entry.Obj.(*cacheEntry).expiresAt)
}

// RegistriesClient is a testable interface for the ACR client List operation.
type RegistriesClient interface {
	List(ctx context.Context) ([]containerregistry.Registry, error)
}

// NewACRProvider parses the specified configFile and returns a DockerConfigProvider
func NewACRProvider(configFile *string) credentialprovider.DockerConfigProvider {
	return &acrProvider{
		file:  configFile,
		cache: cache.NewExpirationStore(stringKeyFunc, &acrExpirationPolicy{}),
	}
}

type acrProvider struct {
	file                  *string
	config                *auth.AzureAuthConfig
	environment           *azure.Environment
	servicePrincipalToken *adal.ServicePrincipalToken
	cache                 cache.Store
}

// ParseConfig returns a parsed configuration for an Azure cloudprovider config file
func parseConfig(configReader io.Reader) (*auth.AzureAuthConfig, error) {
	var config auth.AzureAuthConfig

	if configReader == nil {
		return &config, nil
	}

	limitedReader := &io.LimitedReader{R: configReader, N: maxReadLength}
	configContents, err := ioutil.ReadAll(limitedReader)
	if err != nil {
		return nil, err
	}
	if limitedReader.N <= 0 {
		return nil, errors.New("the read limit is reached")
	}
	err = yaml.Unmarshal(configContents, &config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}

func (a *acrProvider) loadConfig(rdr io.Reader) error {
	var err error
	a.config, err = parseConfig(rdr)
	if err != nil {
		klog.Errorf("Failed to load azure credential file: %v", err)
	}

	a.environment, err = auth.ParseAzureEnvironment(a.config.Cloud, a.config.ResourceManagerEndpoint, a.config.IdentitySystem)
	if err != nil {
		return err
	}

	return nil
}

func (a *acrProvider) Enabled() bool {
	if a.file == nil || len(*a.file) == 0 {
		klog.V(5).Infof("Azure config unspecified, disabling")
		return false
	}

	if credentialprovider.AreLegacyCloudCredentialProvidersDisabled() {
		warnOnce.Do(func() {
			klog.V(4).Infof("Azure credential provider is now disabled. Please refer to sig-cloud-provider for guidance on external credential provider integration for Azure")
		})
		return false
	}

	f, err := os.Open(*a.file)
	if err != nil {
		klog.Errorf("Failed to load config from file: %s", *a.file)
		return false
	}
	defer f.Close()

	err = a.loadConfig(f)
	if err != nil {
		klog.Errorf("Failed to load config from file: %s", *a.file)
		return false
	}

	a.servicePrincipalToken, err = auth.GetServicePrincipalToken(a.config, a.environment)
	if err != nil {
		klog.Errorf("Failed to create service principal token: %v", err)
	}
	return true
}

// getFromCache attempts to get credentials from the cache
func (a *acrProvider) getFromCache(loginServer string) (credentialprovider.DockerConfig, bool) {
	cfg := credentialprovider.DockerConfig{}
	obj, exists, err := a.cache.GetByKey(loginServer)
	if err != nil {
		klog.Errorf("error getting ACR credentials from cache: %v", err)
		return cfg, false
	}
	if !exists {
		return cfg, false
	}

	entry := obj.(*cacheEntry)
	cfg[entry.registry] = entry.credentials
	return cfg, true
}

// getFromACR gets credentials from ACR since they are not in the cache
func (a *acrProvider) getFromACR(loginServer string) (credentialprovider.DockerConfig, error) {
	cfg := credentialprovider.DockerConfig{}
	cred, err := getACRDockerEntryFromARMToken(a, loginServer)
	if err != nil {
		return cfg, err
	}

	entry := &cacheEntry{
		expiresAt:   time.Now().Add(10 * time.Minute),
		credentials: *cred,
		registry:    loginServer,
	}
	if err := a.cache.Add(entry); err != nil {
		return cfg, err
	}
	cfg[loginServer] = *cred
	return cfg, nil
}

func (a *acrProvider) Provide(image string) credentialprovider.DockerConfig {
	loginServer := a.parseACRLoginServerFromImage(image)
	if loginServer == "" {
		klog.V(2).Infof("image(%s) is not from ACR, return empty authentication", image)
		return credentialprovider.DockerConfig{}
	}

	cfg := credentialprovider.DockerConfig{}
	if a.config != nil && a.config.UseManagedIdentityExtension {
		var exists bool
		cfg, exists = a.getFromCache(loginServer)
		if exists {
			klog.V(4).Infof("Got ACR credentials from cache for %s", loginServer)
		} else {
			klog.V(2).Infof("unable to get ACR credentials from cache for %s, checking ACR API", loginServer)
			var err error
			cfg, err = a.getFromACR(loginServer)
			if err != nil {
				klog.Errorf("error getting credentials from ACR for %s %v", loginServer, err)
			}
		}
	} else {
		// Add our entry for each of the supported container registry URLs
		for _, url := range containerRegistryUrls {
			cred := &credentialprovider.DockerConfigEntry{
				Username: a.config.AADClientID,
				Password: a.config.AADClientSecret,
				Email:    dummyRegistryEmail,
			}
			cfg[url] = *cred
		}

		// Handle the custom cloud case
		// In clouds where ACR is not yet deployed, the string will be empty
		if a.environment != nil && strings.Contains(a.environment.ContainerRegistryDNSSuffix, ".azurecr.") {
			customAcrSuffix := "*" + a.environment.ContainerRegistryDNSSuffix
			hasBeenAdded := false
			for _, url := range containerRegistryUrls {
				if strings.EqualFold(url, customAcrSuffix) {
					hasBeenAdded = true
					break
				}
			}

			if !hasBeenAdded {
				cred := &credentialprovider.DockerConfigEntry{
					Username: a.config.AADClientID,
					Password: a.config.AADClientSecret,
					Email:    dummyRegistryEmail,
				}
				cfg[customAcrSuffix] = *cred
			}
		}
	}

	// add ACR anonymous repo support: use empty username and password for anonymous access
	defaultConfigEntry := credentialprovider.DockerConfigEntry{
		Username: "",
		Password: "",
		Email:    dummyRegistryEmail,
	}
	cfg["*.azurecr.*"] = defaultConfigEntry
	return cfg
}

func getLoginServer(registry containerregistry.Registry) string {
	return *(*registry.RegistryProperties).LoginServer
}

func getACRDockerEntryFromARMToken(a *acrProvider, loginServer string) (*credentialprovider.DockerConfigEntry, error) {
	if a.servicePrincipalToken == nil {
		token, err := auth.GetServicePrincipalToken(a.config, a.environment)
		if err != nil {
			klog.Errorf("Failed to create service principal token: %v", err)
			return nil, err
		}
		a.servicePrincipalToken = token
	} else {
		// Run EnsureFresh to make sure the token is valid and does not expire
		if err := a.servicePrincipalToken.EnsureFresh(); err != nil {
			klog.Errorf("Failed to ensure fresh service principal token: %v", err)
			return nil, err
		}
	}

	armAccessToken := a.servicePrincipalToken.OAuthToken()

	klog.V(4).Infof("discovering auth redirects for: %s", loginServer)
	directive, err := receiveChallengeFromLoginServer(loginServer)
	if err != nil {
		klog.Errorf("failed to receive challenge: %s", err)
		return nil, err
	}

	klog.V(4).Infof("exchanging an acr refresh_token")
	registryRefreshToken, err := performTokenExchange(
		loginServer, directive, a.config.TenantID, armAccessToken)
	if err != nil {
		klog.Errorf("failed to perform token exchange: %s", err)
		return nil, err
	}

	klog.V(4).Infof("adding ACR docker config entry for: %s", loginServer)
	return &credentialprovider.DockerConfigEntry{
		Username: dockerTokenLoginUsernameGUID,
		Password: registryRefreshToken,
		Email:    dummyRegistryEmail,
	}, nil
}

// parseACRLoginServerFromImage takes image as parameter and returns login server of it.
// Parameter `image` is expected in following format: foo.azurecr.io/bar/imageName:version
// If the provided image is not an acr image, this function will return an empty string.
func (a *acrProvider) parseACRLoginServerFromImage(image string) string {
	match := acrRE.FindAllString(image, -1)
	if len(match) == 1 {
		return match[0]
	}

	// handle the custom cloud case
	if a != nil && a.environment != nil {
		cloudAcrSuffix := a.environment.ContainerRegistryDNSSuffix
		cloudAcrSuffixLength := len(cloudAcrSuffix)
		if cloudAcrSuffixLength > 0 {
			customAcrSuffixIndex := strings.Index(image, cloudAcrSuffix)
			if customAcrSuffixIndex != -1 {
				endIndex := customAcrSuffixIndex + cloudAcrSuffixLength
				return image[0:endIndex]
			}
		}
	}

	return ""
}
