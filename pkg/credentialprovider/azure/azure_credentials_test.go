package azure

import (
	"testing"

	"github.com/Azure/azure-sdk-for-go/arm/containerregistry"
	"github.com/Azure/go-autorest/autorest/to"
)

type fakeClient struct {
	results containerregistry.RegistryListResult
}

func (f *fakeClient) List() (containerregistry.RegistryListResult, error) {
	return f.results, nil
}

func Test(t *testing.T) {
	configStr := `
    {
        "aadClientId": "foo",
        "aadClientSecret": "bar"
    }`
	emailStr := "name@company.com"

	result := containerregistry.RegistryListResult{
		Value: &[]containerregistry.Registry{
			containerregistry.Registry{
				Name: to.StringPtr("foo"),
			},
			containerregistry.Registry{
				Name: to.StringPtr("bar"),
			},
			containerregistry.Registry{
				Name: to.StringPtr("baz"),
			},
		},
	}
	fakeClient := &fakeClient{
		results: result,
	}

	provider := &acrProvider{
		email:          emailStr,
		registryClient: fakeClient,
	}
	provider.loadConfig([]byte(configStr))

	creds := provider.Provide()

	if len(creds) != len(*result.Value) {
		t.Errorf("Unexpected list: %v, expected length %d", creds, len(*result.Value))
	}
	for _, cred := range creds {
		if cred.Email != emailStr {
			t.Errorf("expected: %s, saw: %s", emailStr, cred.Email)
		}
		if cred.Username != "foo" {
			t.Errorf("expected 'foo' for username, saw: %v", cred.Username)
		}
		if cred.Password != "bar" {
			t.Errorf("expected 'bar' for password, saw: %v", cred.Username)
		}
	}
	for _, val := range *result.Value {
		registryName := *val.Name + ".azurecr.io"
		if _, found := creds[registryName]; !found {
			t.Errorf("Missing expected registry: %s", registryName)
		}
	}
}
