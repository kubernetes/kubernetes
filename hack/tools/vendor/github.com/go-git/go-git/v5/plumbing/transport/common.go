// Package transport includes the implementation for different transport
// protocols.
//
// `Client` can be used to fetch and send packfiles to a git server.
// The `client` package provides higher level functions to instantiate the
// appropriate `Client` based on the repository URL.
//
// go-git supports HTTP and SSH (see `Protocols`), but you can also install
// your own protocols (see the `client` package).
//
// Each protocol has its own implementation of `Client`, but you should
// generally not use them directly, use `client.NewClient` instead.
package transport

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strconv"
	"strings"

	giturl "github.com/go-git/go-git/v5/internal/url"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/capability"
)

var (
	ErrRepositoryNotFound     = errors.New("repository not found")
	ErrEmptyRemoteRepository  = errors.New("remote repository is empty")
	ErrAuthenticationRequired = errors.New("authentication required")
	ErrAuthorizationFailed    = errors.New("authorization failed")
	ErrEmptyUploadPackRequest = errors.New("empty git-upload-pack given")
	ErrInvalidAuthMethod      = errors.New("invalid auth method")
	ErrAlreadyConnected       = errors.New("session already established")
)

const (
	UploadPackServiceName  = "git-upload-pack"
	ReceivePackServiceName = "git-receive-pack"
)

// Transport can initiate git-upload-pack and git-receive-pack processes.
// It is implemented both by the client and the server, making this a RPC.
type Transport interface {
	// NewUploadPackSession starts a git-upload-pack session for an endpoint.
	NewUploadPackSession(*Endpoint, AuthMethod) (UploadPackSession, error)
	// NewReceivePackSession starts a git-receive-pack session for an endpoint.
	NewReceivePackSession(*Endpoint, AuthMethod) (ReceivePackSession, error)
}

type Session interface {
	// AdvertisedReferences retrieves the advertised references for a
	// repository.
	// If the repository does not exist, returns ErrRepositoryNotFound.
	// If the repository exists, but is empty, returns ErrEmptyRemoteRepository.
	AdvertisedReferences() (*packp.AdvRefs, error)
	io.Closer
}

type AuthMethod interface {
	fmt.Stringer
	Name() string
}

// UploadPackSession represents a git-upload-pack session.
// A git-upload-pack session has two steps: reference discovery
// (AdvertisedReferences) and uploading pack (UploadPack).
type UploadPackSession interface {
	Session
	// UploadPack takes a git-upload-pack request and returns a response,
	// including a packfile. Don't be confused by terminology, the client
	// side of a git-upload-pack is called git-fetch-pack, although here
	// the same interface is used to make it RPC-like.
	UploadPack(context.Context, *packp.UploadPackRequest) (*packp.UploadPackResponse, error)
}

// ReceivePackSession represents a git-receive-pack session.
// A git-receive-pack session has two steps: reference discovery
// (AdvertisedReferences) and receiving pack (ReceivePack).
// In that order.
type ReceivePackSession interface {
	Session
	// ReceivePack sends an update references request and a packfile
	// reader and returns a ReportStatus and error. Don't be confused by
	// terminology, the client side of a git-receive-pack is called
	// git-send-pack, although here the same interface is used to make it
	// RPC-like.
	ReceivePack(context.Context, *packp.ReferenceUpdateRequest) (*packp.ReportStatus, error)
}

// Endpoint represents a Git URL in any supported protocol.
type Endpoint struct {
	// Protocol is the protocol of the endpoint (e.g. git, https, file).
	Protocol string
	// User is the user.
	User string
	// Password is the password.
	Password string
	// Host is the host.
	Host string
	// Port is the port to connect, if 0 the default port for the given protocol
	// wil be used.
	Port int
	// Path is the repository path.
	Path string
}

var defaultPorts = map[string]int{
	"http":  80,
	"https": 443,
	"git":   9418,
	"ssh":   22,
}

// String returns a string representation of the Git URL.
func (u *Endpoint) String() string {
	var buf bytes.Buffer
	if u.Protocol != "" {
		buf.WriteString(u.Protocol)
		buf.WriteByte(':')
	}

	if u.Protocol != "" || u.Host != "" || u.User != "" || u.Password != "" {
		buf.WriteString("//")

		if u.User != "" || u.Password != "" {
			buf.WriteString(url.PathEscape(u.User))
			if u.Password != "" {
				buf.WriteByte(':')
				buf.WriteString(url.PathEscape(u.Password))
			}

			buf.WriteByte('@')
		}

		if u.Host != "" {
			buf.WriteString(u.Host)

			if u.Port != 0 {
				port, ok := defaultPorts[strings.ToLower(u.Protocol)]
				if !ok || ok && port != u.Port {
					fmt.Fprintf(&buf, ":%d", u.Port)
				}
			}
		}
	}

	if u.Path != "" && u.Path[0] != '/' && u.Host != "" {
		buf.WriteByte('/')
	}

	buf.WriteString(u.Path)
	return buf.String()
}

func NewEndpoint(endpoint string) (*Endpoint, error) {
	if e, ok := parseSCPLike(endpoint); ok {
		return e, nil
	}

	if e, ok := parseFile(endpoint); ok {
		return e, nil
	}

	return parseURL(endpoint)
}

func parseURL(endpoint string) (*Endpoint, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, err
	}

	if !u.IsAbs() {
		return nil, plumbing.NewPermanentError(fmt.Errorf(
			"invalid endpoint: %s", endpoint,
		))
	}

	var user, pass string
	if u.User != nil {
		user = u.User.Username()
		pass, _ = u.User.Password()
	}

	return &Endpoint{
		Protocol: u.Scheme,
		User:     user,
		Password: pass,
		Host:     u.Hostname(),
		Port:     getPort(u),
		Path:     getPath(u),
	}, nil
}

func getPort(u *url.URL) int {
	p := u.Port()
	if p == "" {
		return 0
	}

	i, err := strconv.Atoi(p)
	if err != nil {
		return 0
	}

	return i
}

func getPath(u *url.URL) string {
	var res string = u.Path
	if u.RawQuery != "" {
		res += "?" + u.RawQuery
	}

	if u.Fragment != "" {
		res += "#" + u.Fragment
	}

	return res
}

func parseSCPLike(endpoint string) (*Endpoint, bool) {
	if giturl.MatchesScheme(endpoint) || !giturl.MatchesScpLike(endpoint) {
		return nil, false
	}

	user, host, portStr, path := giturl.FindScpLikeComponents(endpoint)
	port, err := strconv.Atoi(portStr)
	if err != nil {
		port = 22
	}

	return &Endpoint{
		Protocol: "ssh",
		User:     user,
		Host:     host,
		Port:     port,
		Path:     path,
	}, true
}

func parseFile(endpoint string) (*Endpoint, bool) {
	if giturl.MatchesScheme(endpoint) {
		return nil, false
	}

	path := endpoint
	return &Endpoint{
		Protocol: "file",
		Path:     path,
	}, true
}

// UnsupportedCapabilities are the capabilities not supported by any client
// implementation
var UnsupportedCapabilities = []capability.Capability{
	capability.MultiACK,
	capability.MultiACKDetailed,
	capability.ThinPack,
}

// FilterUnsupportedCapabilities it filter out all the UnsupportedCapabilities
// from a capability.List, the intended usage is on the client implementation
// to filter the capabilities from an AdvRefs message.
func FilterUnsupportedCapabilities(list *capability.List) {
	for _, c := range UnsupportedCapabilities {
		list.Delete(c)
	}
}
