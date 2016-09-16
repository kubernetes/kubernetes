package azure

import (
	"net/url"
)

// OAuthConfig represents the endpoints needed
// in OAuth operations
type OAuthConfig struct {
	AuthorizeEndpoint  url.URL
	TokenEndpoint      url.URL
	DeviceCodeEndpoint url.URL
}
