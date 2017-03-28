package tokens

import "github.com/gophercloud/gophercloud"

// CreateURL generates the URL used to create new Tokens.
func CreateURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("tokens")
}

// GetURL generates the URL used to Validate Tokens.
func GetURL(client *gophercloud.ServiceClient, token string) string {
	return client.ServiceURL("tokens", token)
}
