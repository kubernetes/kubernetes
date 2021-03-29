package httprequest

import (
	"net"
	"net/http"
	"strings"

	"github.com/munnerz/goautoneg"
)

// PrefersHTML returns true if the request was made by something that looks like a browser, or can receive HTML
func PrefersHTML(req *http.Request) bool {
	accepts := goautoneg.ParseAccept(req.Header.Get("Accept"))
	acceptsHTML := false
	acceptsJSON := false
	for _, accept := range accepts {
		if accept.Type == "text" && accept.SubType == "html" {
			acceptsHTML = true
		} else if accept.Type == "application" && accept.SubType == "json" {
			acceptsJSON = true
		}
	}

	// If HTML is accepted, return true
	if acceptsHTML {
		return true
	}

	// If JSON was specifically requested, return false
	// This gives browsers a way to make requests and add an "Accept" header to request JSON
	if acceptsJSON {
		return false
	}

	// In Intranet/Compatibility mode, IE sends an Accept header that does not contain "text/html".
	if strings.HasPrefix(req.UserAgent(), "Mozilla") {
		return true
	}

	return false
}

// SchemeHost returns the scheme and host used to make this request.
// Suitable for use to compute scheme/host in returned 302 redirect Location.
// Note the returned host is not normalized, and may or may not contain a port.
// Returned values are based on the following information:
//
// Host:
// * X-Forwarded-Host/X-Forwarded-Port headers
// * Host field on the request (parsed from Host header)
// * Host in the request's URL (parsed from Request-Line)
//
// Scheme:
// * X-Forwarded-Proto header
// * Existence of TLS information on the request implies https
// * Scheme in the request's URL (parsed from Request-Line)
// * Port (if included in calculated Host value, 443 implies https)
// * Otherwise, defaults to "http"
func SchemeHost(req *http.Request) (string /*scheme*/, string /*host*/) {
	forwarded := func(attr string) string {
		// Get the X-Forwarded-<attr> value
		value := req.Header.Get("X-Forwarded-" + attr)
		// Take the first comma-separated value, if multiple exist
		value = strings.SplitN(value, ",", 2)[0]
		// Trim whitespace
		return strings.TrimSpace(value)
	}

	hasExplicitHost := func(h string) bool {
		_, _, err := net.SplitHostPort(h)
		return err == nil
	}

	forwardedHost := forwarded("Host")
	host := ""
	hostHadExplicitPort := false
	switch {
	case len(forwardedHost) > 0:
		host = forwardedHost
		hostHadExplicitPort = hasExplicitHost(host)

		// If both X-Forwarded-Host and X-Forwarded-Port are sent, use the explicit port info
		if forwardedPort := forwarded("Port"); len(forwardedPort) > 0 {
			if h, _, err := net.SplitHostPort(forwardedHost); err == nil {
				host = net.JoinHostPort(h, forwardedPort)
			} else {
				host = net.JoinHostPort(forwardedHost, forwardedPort)
			}
		}

	case len(req.Host) > 0:
		host = req.Host
		hostHadExplicitPort = hasExplicitHost(host)

	case len(req.URL.Host) > 0:
		host = req.URL.Host
		hostHadExplicitPort = hasExplicitHost(host)
	}

	port := ""
	if _, p, err := net.SplitHostPort(host); err == nil {
		port = p
	}

	forwardedProto := forwarded("Proto")
	scheme := ""
	switch {
	case len(forwardedProto) > 0:
		scheme = forwardedProto
	case req.TLS != nil:
		scheme = "https"
	case len(req.URL.Scheme) > 0:
		scheme = req.URL.Scheme
	case port == "443":
		scheme = "https"
	default:
		scheme = "http"
	}

	if !hostHadExplicitPort {
		if (scheme == "https" && port == "443") || (scheme == "http" && port == "80") {
			if hostWithoutPort, _, err := net.SplitHostPort(host); err == nil {
				host = hostWithoutPort
			}
		}
	}

	return scheme, host
}
