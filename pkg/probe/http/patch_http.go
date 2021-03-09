package http

import (
	"net/http"
	"time"

	"k8s.io/kubernetes/pkg/probe"
)

// Prober is an interface that defines the Probe function for doing HTTP readiness/liveness checks.
type DetailedProber interface {
	ProbeForBody(req *http.Request, timeout time.Duration) (probe.Result, string, string, error)
}

// ProbeForBody returns a ProbeRunner capable of running an HTTP check.
// returns result, details, body, error
func (pr httpProber) ProbeForBody(req *http.Request, timeout time.Duration) (probe.Result, string, string, error) {
	pr.transport.DisableCompression = true // removes Accept-Encoding header
	client := &http.Client{
		Timeout:       timeout,
		Transport:     pr.transport,
		CheckRedirect: RedirectChecker(pr.followNonLocalRedirects),
	}
	return DoHTTPProbe(req, client)
}
