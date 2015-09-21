package gophercloud

import (
	"testing"
)

func TestProviderRegistry(t *testing.T) {
	c := TestContext()

	_, err := c.ProviderByName("aProvider")
	if err == nil {
		t.Error("Expected error when looking for a provider by non-existant name")
		return
	}

	err = c.RegisterProvider("aProvider", Provider{})
	if err != ErrConfiguration {
		t.Error("Unexpected error/nil when registering a provider w/out an auth endpoint\n  %s", err)
		return
	}

	_ = c.RegisterProvider("aProvider", Provider{AuthEndpoint: "http://localhost/auth"})
	_, err = c.ProviderByName("aProvider")
	if err != nil {
		t.Error(err)
		return
	}
}
