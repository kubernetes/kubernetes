package middleware

import (
	"fmt"
	"net/http"
	"runtime"

	"github.com/docker/docker/api/types/versions"
	"golang.org/x/net/context"
)

// VersionMiddleware is a middleware that
// validates the client and server versions.
type VersionMiddleware struct {
	serverVersion  string
	defaultVersion string
	minVersion     string
}

// NewVersionMiddleware creates a new VersionMiddleware
// with the default versions.
func NewVersionMiddleware(s, d, m string) VersionMiddleware {
	return VersionMiddleware{
		serverVersion:  s,
		defaultVersion: d,
		minVersion:     m,
	}
}

type versionUnsupportedError struct {
	version, minVersion, maxVersion string
}

func (e versionUnsupportedError) Error() string {
	if e.minVersion != "" {
		return fmt.Sprintf("client version %s is too old. Minimum supported API version is %s, please upgrade your client to a newer version", e.version, e.minVersion)
	}
	return fmt.Sprintf("client version %s is too new. Maximum supported API version is %s", e.version, e.maxVersion)
}

func (e versionUnsupportedError) InvalidParameter() {}

// WrapHandler returns a new handler function wrapping the previous one in the request chain.
func (v VersionMiddleware) WrapHandler(handler func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error) func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	return func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
		w.Header().Set("Server", fmt.Sprintf("Docker/%s (%s)", v.serverVersion, runtime.GOOS))
		w.Header().Set("API-Version", v.defaultVersion)
		w.Header().Set("OSType", runtime.GOOS)

		apiVersion := vars["version"]
		if apiVersion == "" {
			apiVersion = v.defaultVersion
		}
		if versions.LessThan(apiVersion, v.minVersion) {
			return versionUnsupportedError{version: apiVersion, minVersion: v.minVersion}
		}
		if versions.GreaterThan(apiVersion, v.defaultVersion) {
			return versionUnsupportedError{version: apiVersion, maxVersion: v.defaultVersion}
		}
		// nolint: golint
		ctx = context.WithValue(ctx, "api-version", apiVersion)
		return handler(ctx, w, r, vars)
	}

}
