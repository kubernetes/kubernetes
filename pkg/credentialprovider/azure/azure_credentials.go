package azure

import (
	"fmt"
	"io/ioutil"
	"time"

	yaml "gopkg.in/yaml.v2"

	"github.com/Azure/azure-sdk-for-go/arm/containerregistry"
	azureapi "github.com/Azure/go-autorest/autorest/azure"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

var (
	flagConfigFile = pflag.String("azure-container-registry-config", "",
		"Path to the file container Azure container registry configuration information.")
	flagConfigEmail = pflag.String("azure-container-registry-email", "",
		"Email address to use for the azure container registry docker login information.")
	flagConfigTTL = pflag.Duration("azure-container-registry-credential-cache-ttl", 1*time.Minute,
		"The length of time that credentials are cached before they are re-computed.")
)

// init registers the various means by which credentials may
// be resolved on GCP.
func init() {
	credentialprovider.RegisterCredentialProvider("azure",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: NewACRProvider(*flagConfigFile, *flagConfigEmail),
			Lifetime: *flagConfigTTL,
		})
}

type RegistriesClient interface {
	List() (containerregistry.RegistryListResult, error)
}

func NewACRProvider(configFile, email string) credentialprovider.DockerConfigProvider {
	return &acrProvider{
		file:  configFile,
		email: email,
	}
}

type acrProvider struct {
	file           string
	email          string
	config         azure.Config
	environment    azureapi.Environment
	registryClient RegistriesClient
}

func (a *acrProvider) loadConfig(contents []byte) error {
	err := yaml.Unmarshal(contents, &a.config)
	if err != nil {
		return err
	}

	if a.config.Cloud == "" {
		a.environment = azureapi.PublicCloud
	} else {
		a.environment, err = azureapi.EnvironmentFromName(a.config.Cloud)
		if err != nil {
			return err
		}
	}
	return nil
}

func (a *acrProvider) Enabled() bool {
	if len(a.file) == 0 {
		glog.V(5).Infof("Azure config unspecified, disabling")
		return false
	}
	if len(a.email) == 0 {
		glog.Errorf("Azure configuration specified, but --azure-container-registry-email is empty")
		return false
	}
	contents, err := ioutil.ReadFile(a.file)
	if err != nil {
		glog.Errorf("Failed to load azure credential file: %v", err)
		return false
	}
	if err := a.loadConfig(contents); err != nil {
		glog.Errorf("Failed to load azure credential file: %v", err)
		return false
	}

	oauthConfig, err := a.environment.OAuthConfigForTenant(a.config.TenantID)
	if err != nil {
		glog.Errorf("Failed to load azure credential file: %v", err)
		return false
	}

	servicePrincipalToken, err := azureapi.NewServicePrincipalToken(
		*oauthConfig,
		a.config.AADClientID,
		a.config.AADClientSecret,
		a.environment.ServiceManagementEndpoint)
	if err != nil {
		glog.Errorf("Failed to load azure credential file: %v", err)
		return false
	}

	registryClient := containerregistry.NewRegistriesClient(a.config.SubscriptionID)
	registryClient.BaseURI = a.environment.ResourceManagerEndpoint
	registryClient.Authorizer = servicePrincipalToken
	a.registryClient = registryClient

	return true
}

func (a *acrProvider) Provide() credentialprovider.DockerConfig {
	cfg := credentialprovider.DockerConfig{}
	entry := credentialprovider.DockerConfigEntry{
		Username: a.config.AADClientID,
		Password: a.config.AADClientSecret,
		Email:    a.email,
	}

	res, err := a.registryClient.List()
	if err != nil {
		glog.Errorf("Failed to list registries: %v", err)
		return cfg
	}
	for ix := range *res.Value {
		// TODO: I don't think this will work for national clouds
		cfg[fmt.Sprintf("%s.azurecr.io", *(*res.Value)[ix].Name)] = entry
	}
	return cfg
}

func (a *acrProvider) LazyProvide() *credentialprovider.DockerConfigEntry {
	return nil
}
