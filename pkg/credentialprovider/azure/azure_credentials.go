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
	"io"
	"os"
	"time"

	"github.com/Azure/azure-sdk-for-go/arm/containerregistry"
	"github.com/Azure/go-autorest/autorest"
	azureapi "github.com/Azure/go-autorest/autorest/azure"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

var flagConfigFile = pflag.String("azure-container-registry-config", "",
	"Path to the file container Azure container registry configuration information.")

const dummyRegistryEmail = "name@contoso.com"

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
	List() (containerregistry.RegistryListResult, error)
}

// NewACRProvider parses the specified configFile and returns a DockerConfigProvider
func NewACRProvider(configFile *string) credentialprovider.DockerConfigProvider {
	return &acrProvider{
		file: configFile,
	}
}

type acrProvider struct {
	file           *string
	config         *azure.Config
	environment    *azureapi.Environment
	registryClient RegistriesClient
}

func (a *acrProvider) loadConfig(rdr io.Reader) error {
	var err error
	a.config, a.environment, err = azure.ParseConfig(rdr)
	if err != nil {
		glog.Errorf("Failed to load azure credential file: %v", err)
	}
	return nil
}

func (a *acrProvider) Enabled() bool {
	if a.file == nil || len(*a.file) == 0 {
		glog.V(5).Infof("Azure config unspecified, disabling")
		return false
	}

	f, err := os.Open(*a.file)
	if err != nil {
		glog.Errorf("Failed to load config from file: %s", *a.file)
		return false
	}
	defer f.Close()

	err = a.loadConfig(f)
	if err != nil {
		glog.Errorf("Failed to load config from file: %s", *a.file)
		return false
	}

	servicePrincipalToken, err := azure.GetServicePrincipalToken(a.config, a.environment)
	if err != nil {
		glog.Errorf("Failed to create service principal token: %v", err)
		return false
	}

	registryClient := containerregistry.NewRegistriesClient(a.config.SubscriptionID)
	registryClient.BaseURI = a.environment.ResourceManagerEndpoint
	registryClient.Authorizer = autorest.NewBearerAuthorizer(servicePrincipalToken)
	a.registryClient = registryClient

	return true
}

func (a *acrProvider) Provide() credentialprovider.DockerConfig {
	cfg := credentialprovider.DockerConfig{}
	entry := credentialprovider.DockerConfigEntry{
		Username: a.config.AADClientID,
		Password: a.config.AADClientSecret,
		Email:    dummyRegistryEmail,
	}

	res, err := a.registryClient.List()
	if err != nil {
		glog.Errorf("Failed to list registries: %v", err)
		return cfg
	}
	for ix := range *res.Value {
		loginServer := getLoginServer((*res.Value)[ix])
		glog.V(4).Infof("Adding Azure Container Registry docker credential for %s", loginServer)
		cfg[loginServer] = entry
	}
	return cfg
}

func getLoginServer(registry containerregistry.Registry) string {
	return *(*registry.RegistryProperties).LoginServer
}

func (a *acrProvider) LazyProvide() *credentialprovider.DockerConfigEntry {
	return nil
}
