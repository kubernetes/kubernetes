package proxy

import (
	"net/http"
	"net/url"
	"strings"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/client/auth"
	"github.com/docker/distribution/registry/client/auth/challenge"
)

const challengeHeader = "Docker-Distribution-Api-Version"

type userpass struct {
	username string
	password string
}

type credentials struct {
	creds map[string]userpass
}

func (c credentials) Basic(u *url.URL) (string, string) {
	up := c.creds[u.String()]

	return up.username, up.password
}

func (c credentials) RefreshToken(u *url.URL, service string) string {
	return ""
}

func (c credentials) SetRefreshToken(u *url.URL, service, token string) {
}

// configureAuth stores credentials for challenge responses
func configureAuth(username, password, remoteURL string) (auth.CredentialStore, error) {
	creds := map[string]userpass{}

	authURLs, err := getAuthURLs(remoteURL)
	if err != nil {
		return nil, err
	}

	for _, url := range authURLs {
		context.GetLogger(context.Background()).Infof("Discovered token authentication URL: %s", url)
		creds[url] = userpass{
			username: username,
			password: password,
		}
	}

	return credentials{creds: creds}, nil
}

func getAuthURLs(remoteURL string) ([]string, error) {
	authURLs := []string{}

	resp, err := http.Get(remoteURL + "/v2/")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	for _, c := range challenge.ResponseChallenges(resp) {
		if strings.EqualFold(c.Scheme, "bearer") {
			authURLs = append(authURLs, c.Parameters["realm"])
		}
	}

	return authURLs, nil
}

func ping(manager challenge.Manager, endpoint, versionHeader string) error {
	resp, err := http.Get(endpoint)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := manager.AddResponse(resp); err != nil {
		return err
	}

	return nil
}
