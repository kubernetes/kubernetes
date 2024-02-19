package client // import "github.com/docker/docker/client"

// APIClient is an interface that clients that talk with a docker server must implement.
type APIClient interface {
	CommonAPIClient
	apiClientExperimental
}

// Ensure that Client always implements APIClient.
var _ APIClient = &Client{}
