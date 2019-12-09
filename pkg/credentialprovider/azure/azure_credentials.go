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
	"time"

	"github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/spf13/pflag"

	"k8s.io/klog"
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
)

// init registers the various means by which credentials may
// be resolved on Azure.
func init() {
	credentialprovider.RegisterCredentialProvider("azure",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: NewACRProvider(flagConfigFile),
			Lifetime: 1 * time.Minute,
		})
}

// RegistriesClient is a testable interface for the ACR client List operation.
type RegistriesClient interface {
	List(ctx context.Context) ([]containerregistry.Registry, error)
}

// azRegistriesClient implements RegistriesClient.
type azRegistriesClient struct {
	client containerregistry.RegistriesClient
}

func newAzRegistriesClient(subscriptionID, endpoint string, token *adal.ServicePrincipalToken) *azRegistriesClient {
	registryClient := containerregistry.NewRegistriesClient(subscriptionID)
	registryClient.BaseURI = endpoint
	registryClient.Authorizer = autorest.NewBearerAuthorizer(token)

	return &azRegistriesClient{
		client: registryClient,
	}
}

func (az *azRegistriesClient) List(ctx context.Context) ([]containerregistry.Registry, error) {
	iterator, err := az.client.ListComplete(ctx)
	if err != nil {
		return nil, err
	}

	result := make([]containerregistry.Registry, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, err
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

// NewACRProvider parses the specified configFile and returns a DockerConfigProvider
func NewACRProvider(configFile *string) credentialprovider.DockerConfigProvider {
	return &acrProvider{
		file: configFile,
	}
}

type acrProvider struct {
	file                  *string
	config                *auth.AzureAuthConfig
	environment           *azure.Environment
	registryClient        RegistriesClient
	servicePrincipalToken *adal.ServicePrincipalToken
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
		return false
	}

	a.registryClient = newAzRegistriesClient(a.config.SubscriptionID, a.environment.ResourceManagerEndpoint, a.servicePrincipalToken)
	return true
}

func (a *acrProvider) Provide(image string) credentialprovider.DockerConfig {
	klog.V(4).Infof("try to provide secret for image %s", image)
	cfg := credentialprovider.DockerConfig{}

	if a.config.UseManagedIdentityExtension {
		if loginServer := parseACRLoginServerFromImage(image); loginServer == "" {
			klog.V(4).Infof("image(%s) is not from ACR, skip MSI authentication", image)
		} else {
			if cred, err := getACRDockerEntryFromARMToken(a, loginServer); err == nil {
				cfg[loginServer] = *cred
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
	}

	// add ACR anonymous repo support: use empty username and password for anonymous access
	cfg["*.azurecr.*"] = credentialprovider.DockerConfigEntry{
		Username: "",
		Password: "",
		Email:    dummyRegistryEmail,
	}
	return cfg
}

func getLoginServer(registry containerregistry.Registry) string {
	return *(*registry.RegistryProperties).LoginServer
}

func getACRDockerEntryFromARMToken(a *acrProvider, loginServer string) (*credentialprovider.DockerConfigEntry, error) {
	// Run EnsureFresh to make sure the token is valid and does not expire
	if err := a.servicePrincipalToken.EnsureFresh(); err != nil {
		klog.Errorf("Failed to ensure fresh service principal token: %v", err)
		return nil, err
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
func parseACRLoginServerFromImage(image string) string {
	match := acrRE.FindAllString(image, -1)
	if len(match) == 1 {
		return match[0]
	}
	return ""
}
