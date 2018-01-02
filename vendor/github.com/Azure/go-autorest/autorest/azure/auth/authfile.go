package auth

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"unicode/utf16"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/dimchansky/utfbom"
)

// ClientSetup includes authentication details and cloud specific
// parameters for ARM clients
type ClientSetup struct {
	*autorest.BearerAuthorizer
	File
	BaseURI string
}

// File represents the authentication file
type File struct {
	ClientID                string `json:"clientId,omitempty"`
	ClientSecret            string `json:"clientSecret,omitempty"`
	SubscriptionID          string `json:"subscriptionId,omitempty"`
	TenantID                string `json:"tenantId,omitempty"`
	ActiveDirectoryEndpoint string `json:"activeDirectoryEndpointUrl,omitempty"`
	ResourceManagerEndpoint string `json:"resourceManagerEndpointUrl,omitempty"`
	GraphResourceID         string `json:"activeDirectoryGraphResourceId,omitempty"`
	SQLManagementEndpoint   string `json:"sqlManagementEndpointUrl,omitempty"`
	GalleryEndpoint         string `json:"galleryEndpointUrl,omitempty"`
	ManagementEndpoint      string `json:"managementEndpointUrl,omitempty"`
}

// GetClientSetup provides an authorizer, base URI, subscriptionID and
// tenantID parameters from an Azure CLI auth file
func GetClientSetup(baseURI string) (auth ClientSetup, err error) {
	fileLocation := os.Getenv("AZURE_AUTH_LOCATION")
	if fileLocation == "" {
		return auth, errors.New("auth file not found. Environment variable AZURE_AUTH_LOCATION is not set")
	}

	contents, err := ioutil.ReadFile(fileLocation)
	if err != nil {
		return
	}

	// Auth file might be encoded
	decoded, err := decode(contents)
	if err != nil {
		return
	}

	err = json.Unmarshal(decoded, &auth.File)
	if err != nil {
		return
	}

	resource, err := getResourceForToken(auth.File, baseURI)
	if err != nil {
		return
	}
	auth.BaseURI = resource

	config, err := adal.NewOAuthConfig(auth.ActiveDirectoryEndpoint, auth.TenantID)
	if err != nil {
		return
	}

	spToken, err := adal.NewServicePrincipalToken(*config, auth.ClientID, auth.ClientSecret, resource)
	if err != nil {
		return
	}

	auth.BearerAuthorizer = autorest.NewBearerAuthorizer(spToken)
	return
}

func decode(b []byte) ([]byte, error) {
	reader, enc := utfbom.Skip(bytes.NewReader(b))

	switch enc {
	case utfbom.UTF16LittleEndian:
		u16 := make([]uint16, (len(b)/2)-1)
		err := binary.Read(reader, binary.LittleEndian, &u16)
		if err != nil {
			return nil, err
		}
		return []byte(string(utf16.Decode(u16))), nil
	case utfbom.UTF16BigEndian:
		u16 := make([]uint16, (len(b)/2)-1)
		err := binary.Read(reader, binary.BigEndian, &u16)
		if err != nil {
			return nil, err
		}
		return []byte(string(utf16.Decode(u16))), nil
	}
	return ioutil.ReadAll(reader)
}

func getResourceForToken(f File, baseURI string) (string, error) {
	// Compare dafault base URI from the SDK to the endpoints from the public cloud
	// Base URI and token resource are the same string. This func finds the authentication
	// file field that matches the SDK base URI. The SDK defines the public cloud
	// endpoint as its default base URI
	if !strings.HasSuffix(baseURI, "/") {
		baseURI += "/"
	}
	switch baseURI {
	case azure.PublicCloud.ServiceManagementEndpoint:
		return f.ManagementEndpoint, nil
	case azure.PublicCloud.ResourceManagerEndpoint:
		return f.ResourceManagerEndpoint, nil
	case azure.PublicCloud.ActiveDirectoryEndpoint:
		return f.ActiveDirectoryEndpoint, nil
	case azure.PublicCloud.GalleryEndpoint:
		return f.GalleryEndpoint, nil
	case azure.PublicCloud.GraphEndpoint:
		return f.GraphResourceID, nil
	}
	return "", fmt.Errorf("auth: base URI not found in endpoints")
}
