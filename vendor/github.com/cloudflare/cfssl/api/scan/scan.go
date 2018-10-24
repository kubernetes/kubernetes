package scan

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/scan"
)

// scanHandler is an HTTP handler that accepts GET parameters for host (required)
// family and scanner, and uses these to perform scans, returning a JSON blob result.
func scanHandler(w http.ResponseWriter, r *http.Request) error {
	if err := r.ParseForm(); err != nil {
		log.Warningf("failed to parse body: %v", err)
		return errors.NewBadRequest(err)
	}

	family := r.Form.Get("family")
	scanner := r.Form.Get("scanner")
	ip := r.Form.Get("ip")
	timeoutStr := r.Form.Get("timeout")
	var timeout time.Duration
	var err error
	if timeoutStr != "" {
		if timeout, err = time.ParseDuration(timeoutStr); err != nil {
			return errors.NewBadRequest(err)
		}
		if timeout < time.Second || timeout > 5*time.Minute {
			return errors.NewBadRequestString("invalid timeout given")
		}
	} else {
		timeout = time.Minute
	}

	host := r.Form.Get("host")
	if host == "" {
		log.Warningf("no host given")
		return errors.NewBadRequestString("no host given")
	}

	results, err := scan.Default.RunScans(host, ip, family, scanner, timeout)
	if err != nil {
		return errors.NewBadRequest(err)
	}

	return json.NewEncoder(w).Encode(api.NewSuccessResponse(results))
}

// NewHandler returns a new http.Handler that handles a scan request.
func NewHandler(caBundleFile string) (http.Handler, error) {
	return api.HTTPHandler{
		Handler: api.HandlerFunc(scanHandler),
		Methods: []string{"GET"},
	}, scan.LoadRootCAs(caBundleFile)
}

// scanInfoHandler is an HTTP handler that returns a JSON blob result describing
// the possible families and scans to be run.
func scanInfoHandler(w http.ResponseWriter, r *http.Request) error {
	log.Info("setting up scaninfo handler")
	response := api.NewSuccessResponse(scan.Default)
	enc := json.NewEncoder(w)
	return enc.Encode(response)
}

// NewInfoHandler returns a new http.Handler that handles a request for scan info.
func NewInfoHandler() http.Handler {
	return api.HTTPHandler{
		Handler: api.HandlerFunc(scanInfoHandler),
		Methods: []string{"GET"},
	}
}
