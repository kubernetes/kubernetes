package openshiftkubeapiserver

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/url"

	"github.com/openshift/library-go/pkg/oauth/oauthdiscovery"
)

const (
	// Discovery endpoint for OAuth 2.0 Authorization Server Metadata
	// See IETF Draft:
	// https://tools.ietf.org/html/draft-ietf-oauth-discovery-04#section-2
	oauthMetadataEndpoint = "/.well-known/oauth-authorization-server"
)

func validateURL(urlString string) error {
	urlObj, err := url.Parse(urlString)
	if err != nil {
		return fmt.Errorf("%q is an invalid URL: %v", urlString, err)
	}
	if len(urlObj.Scheme) == 0 {
		return fmt.Errorf("must contain a valid scheme")
	}
	if len(urlObj.Host) == 0 {
		return fmt.Errorf("must contain a valid host")
	}
	return nil
}

func loadOAuthMetadataFile(metadataFile string) ([]byte, error) {
	data, err := ioutil.ReadFile(metadataFile)
	if err != nil {
		return nil, fmt.Errorf("unable to read External OAuth Metadata file: %v", err)
	}

	oauthMetadata := &oauthdiscovery.OauthAuthorizationServerMetadata{}
	if err := json.Unmarshal(data, oauthMetadata); err != nil {
		return nil, fmt.Errorf("unable to decode External OAuth Metadata file: %v", err)
	}

	if err := validateURL(oauthMetadata.Issuer); err != nil {
		return nil, fmt.Errorf("error validating External OAuth Metadata Issuer field: %v", err)
	}

	if err := validateURL(oauthMetadata.AuthorizationEndpoint); err != nil {
		return nil, fmt.Errorf("error validating External OAuth Metadata AuthorizationEndpoint field: %v", err)
	}

	if err := validateURL(oauthMetadata.TokenEndpoint); err != nil {
		return nil, fmt.Errorf("error validating External OAuth Metadata TokenEndpoint field: %v", err)
	}

	return data, nil
}
