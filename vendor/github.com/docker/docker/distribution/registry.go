package distribution

import (
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/client"
	"github.com/docker/distribution/registry/client/auth"
	"github.com/docker/distribution/registry/client/transport"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/registry"
	"github.com/docker/go-connections/sockets"
	"golang.org/x/net/context"
)

// ImageTypes represents the schema2 config types for images
var ImageTypes = []string{
	schema2.MediaTypeImageConfig,
	// Handle unexpected values from https://github.com/docker/distribution/issues/1621
	// (see also https://github.com/docker/docker/issues/22378,
	// https://github.com/docker/docker/issues/30083)
	"application/octet-stream",
	"application/json",
	"text/html",
	// Treat defaulted values as images, newer types cannot be implied
	"",
}

// PluginTypes represents the schema2 config types for plugins
var PluginTypes = []string{
	schema2.MediaTypePluginConfig,
}

var mediaTypeClasses map[string]string

func init() {
	// initialize media type classes with all know types for
	// plugin
	mediaTypeClasses = map[string]string{}
	for _, t := range ImageTypes {
		mediaTypeClasses[t] = "image"
	}
	for _, t := range PluginTypes {
		mediaTypeClasses[t] = "plugin"
	}
}

// NewV2Repository returns a repository (v2 only). It creates an HTTP transport
// providing timeout settings and authentication support, and also verifies the
// remote API version.
func NewV2Repository(ctx context.Context, repoInfo *registry.RepositoryInfo, endpoint registry.APIEndpoint, metaHeaders http.Header, authConfig *types.AuthConfig, actions ...string) (repo distribution.Repository, foundVersion bool, err error) {
	repoName := repoInfo.Name.Name()
	// If endpoint does not support CanonicalName, use the RemoteName instead
	if endpoint.TrimHostname {
		repoName = reference.Path(repoInfo.Name)
	}

	direct := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
		DualStack: true,
	}

	// TODO(dmcgowan): Call close idle connections when complete, use keep alive
	base := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		Dial:                direct.Dial,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     endpoint.TLSConfig,
		// TODO(dmcgowan): Call close idle connections when complete and use keep alive
		DisableKeepAlives: true,
	}

	proxyDialer, err := sockets.DialerFromEnvironment(direct)
	if err == nil {
		base.Dial = proxyDialer.Dial
	}

	modifiers := registry.DockerHeaders(dockerversion.DockerUserAgent(ctx), metaHeaders)
	authTransport := transport.NewTransport(base, modifiers...)

	challengeManager, foundVersion, err := registry.PingV2Registry(endpoint.URL, authTransport)
	if err != nil {
		transportOK := false
		if responseErr, ok := err.(registry.PingResponseError); ok {
			transportOK = true
			err = responseErr.Err
		}
		return nil, foundVersion, fallbackError{
			err:         err,
			confirmedV2: foundVersion,
			transportOK: transportOK,
		}
	}

	if authConfig.RegistryToken != "" {
		passThruTokenHandler := &existingTokenHandler{token: authConfig.RegistryToken}
		modifiers = append(modifiers, auth.NewAuthorizer(challengeManager, passThruTokenHandler))
	} else {
		scope := auth.RepositoryScope{
			Repository: repoName,
			Actions:    actions,
			Class:      repoInfo.Class,
		}

		creds := registry.NewStaticCredentialStore(authConfig)
		tokenHandlerOptions := auth.TokenHandlerOptions{
			Transport:   authTransport,
			Credentials: creds,
			Scopes:      []auth.Scope{scope},
			ClientID:    registry.AuthClientID,
		}
		tokenHandler := auth.NewTokenHandlerWithOptions(tokenHandlerOptions)
		basicHandler := auth.NewBasicHandler(creds)
		modifiers = append(modifiers, auth.NewAuthorizer(challengeManager, tokenHandler, basicHandler))
	}
	tr := transport.NewTransport(base, modifiers...)

	repoNameRef, err := reference.WithName(repoName)
	if err != nil {
		return nil, foundVersion, fallbackError{
			err:         err,
			confirmedV2: foundVersion,
			transportOK: true,
		}
	}

	repo, err = client.NewRepository(ctx, repoNameRef, endpoint.URL.String(), tr)
	if err != nil {
		err = fallbackError{
			err:         err,
			confirmedV2: foundVersion,
			transportOK: true,
		}
	}
	return
}

type existingTokenHandler struct {
	token string
}

func (th *existingTokenHandler) Scheme() string {
	return "bearer"
}

func (th *existingTokenHandler) AuthorizeRequest(req *http.Request, params map[string]string) error {
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", th.token))
	return nil
}
