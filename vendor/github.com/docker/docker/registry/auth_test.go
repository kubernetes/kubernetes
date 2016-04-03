package registry

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/docker/docker/cliconfig"
)

func TestEncodeAuth(t *testing.T) {
	newAuthConfig := &cliconfig.AuthConfig{Username: "ken", Password: "test", Email: "test@example.com"}
	authStr := cliconfig.EncodeAuth(newAuthConfig)
	decAuthConfig := &cliconfig.AuthConfig{}
	var err error
	decAuthConfig.Username, decAuthConfig.Password, err = cliconfig.DecodeAuth(authStr)
	if err != nil {
		t.Fatal(err)
	}
	if newAuthConfig.Username != decAuthConfig.Username {
		t.Fatal("Encode Username doesn't match decoded Username")
	}
	if newAuthConfig.Password != decAuthConfig.Password {
		t.Fatal("Encode Password doesn't match decoded Password")
	}
	if authStr != "a2VuOnRlc3Q=" {
		t.Fatal("AuthString encoding isn't correct.")
	}
}

func setupTempConfigFile() (*cliconfig.ConfigFile, error) {
	root, err := ioutil.TempDir("", "docker-test-auth")
	if err != nil {
		return nil, err
	}
	root = filepath.Join(root, cliconfig.ConfigFileName)
	configFile := cliconfig.NewConfigFile(root)

	for _, registry := range []string{"testIndex", INDEXSERVER} {
		configFile.AuthConfigs[registry] = cliconfig.AuthConfig{
			Username: "docker-user",
			Password: "docker-pass",
			Email:    "docker@docker.io",
		}
	}

	return configFile, nil
}

func TestSameAuthDataPostSave(t *testing.T) {
	configFile, err := setupTempConfigFile()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(configFile.Filename())

	err = configFile.Save()
	if err != nil {
		t.Fatal(err)
	}

	authConfig := configFile.AuthConfigs["testIndex"]
	if authConfig.Username != "docker-user" {
		t.Fail()
	}
	if authConfig.Password != "docker-pass" {
		t.Fail()
	}
	if authConfig.Email != "docker@docker.io" {
		t.Fail()
	}
	if authConfig.Auth != "" {
		t.Fail()
	}
}

func TestResolveAuthConfigIndexServer(t *testing.T) {
	configFile, err := setupTempConfigFile()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(configFile.Filename())

	indexConfig := configFile.AuthConfigs[INDEXSERVER]

	officialIndex := &IndexInfo{
		Official: true,
	}
	privateIndex := &IndexInfo{
		Official: false,
	}

	resolved := ResolveAuthConfig(configFile, officialIndex)
	assertEqual(t, resolved, indexConfig, "Expected ResolveAuthConfig to return INDEXSERVER")

	resolved = ResolveAuthConfig(configFile, privateIndex)
	assertNotEqual(t, resolved, indexConfig, "Expected ResolveAuthConfig to not return INDEXSERVER")
}

func TestResolveAuthConfigFullURL(t *testing.T) {
	configFile, err := setupTempConfigFile()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(configFile.Filename())

	registryAuth := cliconfig.AuthConfig{
		Username: "foo-user",
		Password: "foo-pass",
		Email:    "foo@example.com",
	}
	localAuth := cliconfig.AuthConfig{
		Username: "bar-user",
		Password: "bar-pass",
		Email:    "bar@example.com",
	}
	officialAuth := cliconfig.AuthConfig{
		Username: "baz-user",
		Password: "baz-pass",
		Email:    "baz@example.com",
	}
	configFile.AuthConfigs[INDEXSERVER] = officialAuth

	expectedAuths := map[string]cliconfig.AuthConfig{
		"registry.example.com": registryAuth,
		"localhost:8000":       localAuth,
		"registry.com":         localAuth,
	}

	validRegistries := map[string][]string{
		"registry.example.com": {
			"https://registry.example.com/v1/",
			"http://registry.example.com/v1/",
			"registry.example.com",
			"registry.example.com/v1/",
		},
		"localhost:8000": {
			"https://localhost:8000/v1/",
			"http://localhost:8000/v1/",
			"localhost:8000",
			"localhost:8000/v1/",
		},
		"registry.com": {
			"https://registry.com/v1/",
			"http://registry.com/v1/",
			"registry.com",
			"registry.com/v1/",
		},
	}

	for configKey, registries := range validRegistries {
		configured, ok := expectedAuths[configKey]
		if !ok || configured.Email == "" {
			t.Fail()
		}
		index := &IndexInfo{
			Name: configKey,
		}
		for _, registry := range registries {
			configFile.AuthConfigs[registry] = configured
			resolved := ResolveAuthConfig(configFile, index)
			if resolved.Email != configured.Email {
				t.Errorf("%s -> %q != %q\n", registry, resolved.Email, configured.Email)
			}
			delete(configFile.AuthConfigs, registry)
			resolved = ResolveAuthConfig(configFile, index)
			if resolved.Email == configured.Email {
				t.Errorf("%s -> %q == %q\n", registry, resolved.Email, configured.Email)
			}
		}
	}
}
