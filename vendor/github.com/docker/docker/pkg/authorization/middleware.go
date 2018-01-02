package authorization

import (
	"net/http"
	"sync"

	"github.com/docker/docker/pkg/plugingetter"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

// Middleware uses a list of plugins to
// handle authorization in the API requests.
type Middleware struct {
	mu      sync.Mutex
	plugins []Plugin
}

// NewMiddleware creates a new Middleware
// with a slice of plugins names.
func NewMiddleware(names []string, pg plugingetter.PluginGetter) *Middleware {
	SetPluginGetter(pg)
	return &Middleware{
		plugins: newPlugins(names),
	}
}

func (m *Middleware) getAuthzPlugins() []Plugin {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.plugins
}

// SetPlugins sets the plugin used for authorization
func (m *Middleware) SetPlugins(names []string) {
	m.mu.Lock()
	m.plugins = newPlugins(names)
	m.mu.Unlock()
}

// RemovePlugin removes a single plugin from this authz middleware chain
func (m *Middleware) RemovePlugin(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	plugins := m.plugins[:0]
	for _, authPlugin := range m.plugins {
		if authPlugin.Name() != name {
			plugins = append(plugins, authPlugin)
		}
	}
	m.plugins = plugins
}

// WrapHandler returns a new handler function wrapping the previous one in the request chain.
func (m *Middleware) WrapHandler(handler func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error) func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	return func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
		plugins := m.getAuthzPlugins()
		if len(plugins) == 0 {
			return handler(ctx, w, r, vars)
		}

		user := ""
		userAuthNMethod := ""

		// Default authorization using existing TLS connection credentials
		// FIXME: Non trivial authorization mechanisms (such as advanced certificate validations, kerberos support
		// and ldap) will be extracted using AuthN feature, which is tracked under:
		// https://github.com/docker/docker/pull/20883
		if r.TLS != nil && len(r.TLS.PeerCertificates) > 0 {
			user = r.TLS.PeerCertificates[0].Subject.CommonName
			userAuthNMethod = "TLS"
		}

		authCtx := NewCtx(plugins, user, userAuthNMethod, r.Method, r.RequestURI)

		if err := authCtx.AuthZRequest(w, r); err != nil {
			logrus.Errorf("AuthZRequest for %s %s returned error: %s", r.Method, r.RequestURI, err)
			return err
		}

		rw := NewResponseModifier(w)

		var errD error

		if errD = handler(ctx, rw, r, vars); errD != nil {
			logrus.Errorf("Handler for %s %s returned error: %s", r.Method, r.RequestURI, errD)
		}

		// There's a chance that the authCtx.plugins was updated. One of the reasons
		// this can happen is when an authzplugin is disabled.
		plugins = m.getAuthzPlugins()
		if len(plugins) == 0 {
			logrus.Debug("There are no authz plugins in the chain")
			return nil
		}

		authCtx.plugins = plugins

		if err := authCtx.AuthZResponse(rw, r); errD == nil && err != nil {
			logrus.Errorf("AuthZResponse for %s %s returned error: %s", r.Method, r.RequestURI, err)
			return err
		}

		if errD != nil {
			return errD
		}

		return nil
	}
}
