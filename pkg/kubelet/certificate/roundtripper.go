package certificate

import (
	"fmt"
	"net/http"
	"sync"
)

type rotationRoundTripper struct {
	template            *http.Transport
	certificateManager  *manager
	transportAccessLock sync.RWMutex
	transport           *http.Transport
	certVersion         int
}

func newRotationRoundTripper(certificateManager *manager, template *http.Transport) (http.RoundTripper, error) {
	if template == nil {
		return nil, fmt.Errorf("template can not be nil")
	}
	transport := *template
	return &rotationRoundTripper{
		template:           template,
		transport:          &transport,
		certificateManager: certificateManager,
		certVersion:        certificateManager.certVersion,
	}, nil
}

func (rt *rotationRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	rt.transportAccessLock.RLock()
	if !rt.certificateManager.sameVersion(rt.certVersion) {
		rt.transportAccessLock.RUnlock()
		rt.updateTransport()
		rt.transportAccessLock.RLock()
	}
	t := rt.transport
	rt.transportAccessLock.RUnlock()
	return t.RoundTrip(request)
}

func (rt *rotationRoundTripper) updateTransport() {
	rt.transportAccessLock.Lock()
	defer rt.transportAccessLock.Unlock()
	*rt.transport = *rt.template
	rt.certVersion = rt.certificateManager.certVersion
}
