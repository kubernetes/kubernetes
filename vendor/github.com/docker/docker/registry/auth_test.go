// +build !solaris

// TODO: Support Solaris

package registry

import (
	"testing"

	"github.com/docker/docker/api/types"
	registrytypes "github.com/docker/docker/api/types/registry"
)

func buildAuthConfigs() map[string]types.AuthConfig {
	authConfigs := map[string]types.AuthConfig{}

	for _, registry := range []string{"testIndex", IndexServer} {
		authConfigs[registry] = types.AuthConfig{
			Username: "docker-user",
			Password: "docker-pass",
		}
	}

	return authConfigs
}

func TestSameAuthDataPostSave(t *testing.T) {
	authConfigs := buildAuthConfigs()
	authConfig := authConfigs["testIndex"]
	if authConfig.Username != "docker-user" {
		t.Fail()
	}
	if authConfig.Password != "docker-pass" {
		t.Fail()
	}
	if authConfig.Auth != "" {
		t.Fail()
	}
}

func TestResolveAuthConfigIndexServer(t *testing.T) {
	authConfigs := buildAuthConfigs()
	indexConfig := authConfigs[IndexServer]

	officialIndex := &registrytypes.IndexInfo{
		Official: true,
	}
	privateIndex := &registrytypes.IndexInfo{
		Official: false,
	}

	resolved := ResolveAuthConfig(authConfigs, officialIndex)
	assertEqual(t, resolved, indexConfig, "Expected ResolveAuthConfig to return IndexServer")

	resolved = ResolveAuthConfig(authConfigs, privateIndex)
	assertNotEqual(t, resolved, indexConfig, "Expected ResolveAuthConfig to not return IndexServer")
}

func TestResolveAuthConfigFullURL(t *testing.T) {
	authConfigs := buildAuthConfigs()

	registryAuth := types.AuthConfig{
		Username: "foo-user",
		Password: "foo-pass",
	}
	localAuth := types.AuthConfig{
		Username: "bar-user",
		Password: "bar-pass",
	}
	officialAuth := types.AuthConfig{
		Username: "baz-user",
		Password: "baz-pass",
	}
	authConfigs[IndexServer] = officialAuth

	expectedAuths := map[string]types.AuthConfig{
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
		if !ok {
			t.Fail()
		}
		index := &registrytypes.IndexInfo{
			Name: configKey,
		}
		for _, registry := range registries {
			authConfigs[registry] = configured
			resolved := ResolveAuthConfig(authConfigs, index)
			if resolved.Username != configured.Username || resolved.Password != configured.Password {
				t.Errorf("%s -> %v != %v\n", registry, resolved, configured)
			}
			delete(authConfigs, registry)
			resolved = ResolveAuthConfig(authConfigs, index)
			if resolved.Username == configured.Username || resolved.Password == configured.Password {
				t.Errorf("%s -> %v == %v\n", registry, resolved, configured)
			}
		}
	}
}
