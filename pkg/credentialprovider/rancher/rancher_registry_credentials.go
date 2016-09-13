package rancher_credentials

import (
	"os"
	"time"

	"github.com/golang/glog"
	"github.com/rancher/go-rancher/client"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

// rancher provider
type rancherProvider struct {
	credGetter credentialsGetter
}

// credentials getter from Rancher private registry
type rancherCredentialsGetter struct {
	client *client.RancherClient
}

type rConfig struct {
	Global configGlobal
}

// An interface for testing purposes.
type credentialsGetter interface {
	getCredentials() []registryCredential
}

type configGlobal struct {
	CattleURL       string `gcfg:"cattle-url"`
	CattleAccessKey string `gcfg:"cattle-access-key"`
	CattleSecretKey string `gcfg:"cattle-secret-key"`
}

type registryCredential struct {
	credential *client.RegistryCredential
	serverIP   string
}

func init() {
	client, err := getRancherClient()
	if err != nil {
		glog.Errorf("Failed to get rancher client: %v", err)
	}

	rancherGetter := &rancherCredentialsGetter{
		client: client,
	}

	credentialprovider.RegisterCredentialProvider("rancher-registry-creds",
		&credentialprovider.CachingDockerConfigProvider{
			Provider: &rancherProvider{rancherGetter},
			Lifetime: 30 * time.Minute,
		})
}

// Assuming it's always enabled
func (p *rancherProvider) Enabled() bool {
	return p.credGetter != nil
}

// LazyProvide implements DockerConfigProvider. Should never be called.
func (p *rancherProvider) LazyProvide() *credentialprovider.DockerConfigEntry {
	return nil
}

// Provide implements DockerConfigProvider.Provide, refreshing Rancher tokens on demand
func (p *rancherProvider) Provide() credentialprovider.DockerConfig {
	cfg := credentialprovider.DockerConfig{}
	for _, cred := range p.credGetter.getCredentials() {
		entry := credentialprovider.DockerConfigEntry{
			Username: cred.credential.PublicValue,
			Password: cred.credential.SecretValue,
			Email:    cred.credential.Email,
		}
		cfg[cred.serverIP] = entry
	}

	return cfg
}

func (g *rancherCredentialsGetter) getCredentials() []registryCredential {
	var registryCreds []registryCredential
	credColl, err := g.client.RegistryCredential.List(client.NewListOpts())
	if err != nil {
		glog.Errorf("Failed to pull registry credentials from rancher %v", err)
		return registryCreds
	}
	for _, cred := range credColl.Data {
		registry := &client.Registry{}
		if err = g.client.GetLink(cred.Resource, "registry", registry); err != nil {
			glog.Errorf("Failed to pull registry from rancher %v", err)
			return registryCreds
		}
		registryCred := registryCredential{
			credential: &cred,
			serverIP:   registry.ServerAddress,
		}
		registryCreds = append(registryCreds, registryCred)
	}
	return registryCreds
}

func getRancherClient() (*client.RancherClient, error) {
	url := os.Getenv("CATTLE_URL")
	accessKey := os.Getenv("CATTLE_ACCESS_KEY")
	secretKey := os.Getenv("CATTLE_SECRET_KEY")

	if url == "" || accessKey == "" || secretKey == "" {
		return nil, nil
	}

	conf := rConfig{
		Global: configGlobal{
			CattleURL:       url,
			CattleAccessKey: accessKey,
			CattleSecretKey: secretKey,
		},
	}

	return client.NewRancherClient(&client.ClientOpts{
		Url:       conf.Global.CattleURL,
		AccessKey: conf.Global.CattleAccessKey,
		SecretKey: conf.Global.CattleSecretKey,
	})
}
