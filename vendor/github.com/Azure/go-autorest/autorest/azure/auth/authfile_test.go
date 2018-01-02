package auth

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

var (
	expectedFile = File{
		ClientID:                "client-id-123",
		ClientSecret:            "client-secret-456",
		SubscriptionID:          "sub-id-789",
		TenantID:                "tenant-id-123",
		ActiveDirectoryEndpoint: "https://login.microsoftonline.com",
		ResourceManagerEndpoint: "https://management.azure.com/",
		GraphResourceID:         "https://graph.windows.net/",
		SQLManagementEndpoint:   "https://management.core.windows.net:8443/",
		GalleryEndpoint:         "https://gallery.azure.com/",
		ManagementEndpoint:      "https://management.core.windows.net/",
	}
)

func TestGetClientSetup(t *testing.T) {
	os.Setenv("AZURE_AUTH_LOCATION", filepath.Join(getCredsPath(), "credsutf16le.json"))
	setup, err := GetClientSetup("https://management.azure.com")
	if err != nil {
		t.Logf("GetClientSetup failed, got error %v", err)
		t.Fail()
	}

	if setup.BaseURI != "https://management.azure.com/" {
		t.Logf("auth.BaseURI not set correctly, expected 'https://management.azure.com/', got '%s'", setup.BaseURI)
		t.Fail()
	}

	if !reflect.DeepEqual(expectedFile, setup.File) {
		t.Logf("auth.File not set correctly, expected %v, got %v", expectedFile, setup.File)
		t.Fail()
	}

	if setup.BearerAuthorizer == nil {
		t.Log("auth.Authorizer not set correctly, got nil")
		t.Fail()
	}
}

func TestDecodeAndUnmarshal(t *testing.T) {
	tests := []string{
		"credsutf8.json",
		"credsutf16le.json",
		"credsutf16be.json",
	}
	creds := getCredsPath()
	for _, test := range tests {
		b, err := ioutil.ReadFile(filepath.Join(creds, test))
		if err != nil {
			t.Logf("error reading file '%s': %s", test, err)
			t.Fail()
		}
		decoded, err := decode(b)
		if err != nil {
			t.Logf("error decoding file '%s': %s", test, err)
			t.Fail()
		}
		var got File
		err = json.Unmarshal(decoded, &got)
		if err != nil {
			t.Logf("error unmarshaling file '%s': %s", test, err)
			t.Fail()
		}
		if !reflect.DeepEqual(expectedFile, got) {
			t.Logf("unmarshaled map expected %v, got %v", expectedFile, got)
			t.Fail()
		}
	}
}

func getCredsPath() string {
	gopath := os.Getenv("GOPATH")
	return filepath.Join(gopath, "src", "github.com", "Azure", "go-autorest", "testdata")
}

func areMapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if a[k] != b[k] {
			return false
		}
	}
	return true
}
