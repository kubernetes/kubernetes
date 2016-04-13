package registry

import (
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/distribution/registry/api/v2"
	"github.com/docker/distribution/registry/client/transport"
	"github.com/docker/docker/autogen/dockerversion"
	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/docker/docker/pkg/tlsconfig"
	"github.com/docker/docker/pkg/useragent"
)

var (
	ErrAlreadyExists = errors.New("Image already exists")
	ErrDoesNotExist  = errors.New("Image does not exist")
	errLoginRequired = errors.New("Authentication is required.")
)

type TimeoutType uint32

const (
	NoTimeout TimeoutType = iota
	ReceiveTimeout
	ConnectTimeout
)

// dockerUserAgent is the User-Agent the Docker client uses to identify itself.
// It is populated on init(), comprising version information of different components.
var dockerUserAgent string

func init() {
	httpVersion := make([]useragent.VersionInfo, 0, 6)
	httpVersion = append(httpVersion, useragent.VersionInfo{"docker", dockerversion.VERSION})
	httpVersion = append(httpVersion, useragent.VersionInfo{"go", runtime.Version()})
	httpVersion = append(httpVersion, useragent.VersionInfo{"git-commit", dockerversion.GITCOMMIT})
	if kernelVersion, err := kernel.GetKernelVersion(); err == nil {
		httpVersion = append(httpVersion, useragent.VersionInfo{"kernel", kernelVersion.String()})
	}
	httpVersion = append(httpVersion, useragent.VersionInfo{"os", runtime.GOOS})
	httpVersion = append(httpVersion, useragent.VersionInfo{"arch", runtime.GOARCH})

	dockerUserAgent = useragent.AppendVersions("", httpVersion...)
}

func hasFile(files []os.FileInfo, name string) bool {
	for _, f := range files {
		if f.Name() == name {
			return true
		}
	}
	return false
}

// DockerHeaders returns request modifiers that ensure requests have
// the User-Agent header set to dockerUserAgent and that metaHeaders
// are added.
func DockerHeaders(metaHeaders http.Header) []transport.RequestModifier {
	modifiers := []transport.RequestModifier{
		transport.NewHeaderRequestModifier(http.Header{"User-Agent": []string{dockerUserAgent}}),
	}
	if metaHeaders != nil {
		modifiers = append(modifiers, transport.NewHeaderRequestModifier(metaHeaders))
	}
	return modifiers
}

func HTTPClient(transport http.RoundTripper) *http.Client {
	return &http.Client{
		Transport:     transport,
		CheckRedirect: AddRequiredHeadersToRedirectedRequests,
	}
}

func trustedLocation(req *http.Request) bool {
	var (
		trusteds = []string{"docker.com", "docker.io"}
		hostname = strings.SplitN(req.Host, ":", 2)[0]
	)
	if req.URL.Scheme != "https" {
		return false
	}

	for _, trusted := range trusteds {
		if hostname == trusted || strings.HasSuffix(hostname, "."+trusted) {
			return true
		}
	}
	return false
}

func AddRequiredHeadersToRedirectedRequests(req *http.Request, via []*http.Request) error {
	if via != nil && via[0] != nil {
		if trustedLocation(req) && trustedLocation(via[0]) {
			req.Header = via[0].Header
			return nil
		}
		for k, v := range via[0].Header {
			if k != "Authorization" {
				for _, vv := range v {
					req.Header.Add(k, vv)
				}
			}
		}
	}
	return nil
}

func shouldV2Fallback(err errcode.Error) bool {
	logrus.Debugf("v2 error: %T %v", err, err)
	switch err.Code {
	case v2.ErrorCodeUnauthorized, v2.ErrorCodeManifestUnknown:
		return true
	}
	return false
}

type ErrNoSupport struct{ Err error }

func (e ErrNoSupport) Error() string {
	if e.Err == nil {
		return "not supported"
	}
	return e.Err.Error()
}

func ContinueOnError(err error) bool {
	switch v := err.(type) {
	case errcode.Errors:
		return ContinueOnError(v[0])
	case ErrNoSupport:
		return ContinueOnError(v.Err)
	case errcode.Error:
		return shouldV2Fallback(v)
	}
	return false
}

func NewTransport(tlsConfig *tls.Config) *http.Transport {
	if tlsConfig == nil {
		var cfg = tlsconfig.ServerDefault
		tlsConfig = &cfg
	}
	return &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		Dial: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
			DualStack: true,
		}).Dial,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     tlsConfig,
		// TODO(dmcgowan): Call close idle connections when complete and use keep alive
		DisableKeepAlives: true,
	}
}
