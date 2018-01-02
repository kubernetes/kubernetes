package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httputil"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/auth"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/whitelist"
	metrics "github.com/cloudflare/go-metrics"
)

// A SignatureResponse contains only a certificate, as there is no other
// useful data for the CA to return at this time.
type SignatureResponse struct {
	Certificate string `json:"certificate"`
}

type filter func(string, *signer.SignRequest) bool

var filters = map[string][]filter{}

type signerStats struct {
	Counter metrics.Counter
	Rate    metrics.Meter
}

var stats struct {
	Registry         metrics.Registry
	Requests         map[string]signerStats
	TotalRequestRate metrics.Meter
	ErrorPercent     metrics.GaugeFloat64
	ErrorRate        metrics.Meter
}

func initStats() {
	stats.Registry = metrics.NewRegistry()

	stats.Requests = map[string]signerStats{}

	// signers is defined in ca.go
	for k := range signers {
		stats.Requests[k] = signerStats{
			Counter: metrics.NewRegisteredCounter("requests:"+k, stats.Registry),
			Rate:    metrics.NewRegisteredMeter("request-rate:"+k, stats.Registry),
		}
	}

	stats.TotalRequestRate = metrics.NewRegisteredMeter("total-request-rate", stats.Registry)
	stats.ErrorPercent = metrics.NewRegisteredGaugeFloat64("error-percent", stats.Registry)
	stats.ErrorRate = metrics.NewRegisteredMeter("error-rate", stats.Registry)
}

// incError increments the error count and updates the error percentage.
func incErrors() {
	stats.ErrorRate.Mark(1)
	eCtr := float64(stats.ErrorRate.Count())
	rCtr := float64(stats.TotalRequestRate.Count())
	stats.ErrorPercent.Update(eCtr / rCtr * 100)
}

// incRequests increments the request count and updates the error percentage.
func incRequests() {
	stats.TotalRequestRate.Mark(1)
	eCtr := float64(stats.ErrorRate.Count())
	rCtr := float64(stats.TotalRequestRate.Count())
	stats.ErrorPercent.Update(eCtr / rCtr * 100)
}

func fail(w http.ResponseWriter, req *http.Request, status, code int, msg, ad string) {
	incErrors()

	if ad != "" {
		ad = " (" + ad + ")"
	}
	log.Errorf("[HTTP %d] %d - %s%s", status, code, msg, ad)

	dumpReq, err := httputil.DumpRequest(req, true)
	if err != nil {
		fmt.Printf("%v#v\n", req)
	} else {
		fmt.Printf("%s\n", dumpReq)
	}

	res := api.NewErrorResponse(msg, code)
	w.WriteHeader(status)
	jenc := json.NewEncoder(w)
	jenc.Encode(res)
}

func dispatchRequest(w http.ResponseWriter, req *http.Request) {
	incRequests()

	if req.Method != "POST" {
		fail(w, req, http.StatusMethodNotAllowed, 1, "only POST is permitted", "")
		return
	}

	body, err := ioutil.ReadAll(req.Body)
	if err != nil {
		fail(w, req, http.StatusInternalServerError, 1, err.Error(), "while reading request body")
		return
	}
	defer req.Body.Close()

	var authReq auth.AuthenticatedRequest
	err = json.Unmarshal(body, &authReq)
	if err != nil {
		fail(w, req, http.StatusBadRequest, 1, err.Error(), "while unmarshaling request body")
		return
	}

	var sigRequest signer.SignRequest
	err = json.Unmarshal(authReq.Request, &sigRequest)
	if err != nil {
		fail(w, req, http.StatusBadRequest, 1, err.Error(), "while unmarshalling authenticated request")
		return
	}

	if sigRequest.Label == "" {
		sigRequest.Label = defaultLabel
	}

	acl := whitelists[sigRequest.Label]
	if acl != nil {
		ip, err := whitelist.HTTPRequestLookup(req)
		if err != nil {
			fail(w, req, http.StatusInternalServerError, 1, err.Error(), "while getting request IP")
			return
		}

		if !acl.Permitted(ip) {
			fail(w, req, http.StatusForbidden, 1, "not authorised", "because IP is not whitelisted")
			return
		}
	}

	s, ok := signers[sigRequest.Label]
	if !ok {
		fail(w, req, http.StatusBadRequest, 1, "bad request", "request is for non-existent label "+sigRequest.Label)
		return
	}

	stats.Requests[sigRequest.Label].Counter.Inc(1)
	stats.Requests[sigRequest.Label].Rate.Mark(1)

	// Sanity checks to ensure that we have a valid policy. This
	// should have been checked in NewAuthSignHandler.
	policy := s.Policy()
	if policy == nil {
		fail(w, req, http.StatusInternalServerError, 1, "invalid policy", "signer was initialised without a signing policy")
		return
	}
	profile := policy.Default

	if policy.Profiles != nil && sigRequest.Profile != "" {
		profile = policy.Profiles[sigRequest.Profile]
		if profile == nil {
			fail(w, req, http.StatusBadRequest, 1, "invalid profile", "failed to look up profile with name: "+sigRequest.Profile)
			return
		}
	}

	if profile == nil {
		fail(w, req, http.StatusInternalServerError, 1, "invalid profile", "signer was initialised without any valid profiles")
		return
	}

	if profile.Provider == nil {
		fail(w, req, http.StatusUnauthorized, 1, "authorisation required", "received unauthenticated request")
		return
	}

	if !profile.Provider.Verify(&authReq) {
		fail(w, req, http.StatusBadRequest, 1, "invalid token", "received authenticated request with invalid token")
		return
	}

	if sigRequest.Request == "" {
		fail(w, req, http.StatusBadRequest, 1, "invalid request", "empty request")
		return
	}

	cert, err := s.Sign(sigRequest)
	if err != nil {
		fail(w, req, http.StatusBadRequest, 1, "bad request", "signature failed: "+err.Error())
		return
	}

	x509Cert, err := helpers.ParseCertificatePEM(cert)
	if err != nil {
		fail(w, req, http.StatusInternalServerError, 1, "bad certificate", err.Error())
	}

	log.Infof("signature: requester=%s, label=%s, profile=%s, serialno=%s",
		req.RemoteAddr, sigRequest.Label, sigRequest.Profile, x509Cert.SerialNumber)

	res := api.NewSuccessResponse(&SignatureResponse{Certificate: string(cert)})
	jenc := json.NewEncoder(w)
	err = jenc.Encode(res)
	if err != nil {
		log.Errorf("error writing response: %v", err)
	}
}

func metricsDisallowed(w http.ResponseWriter, req *http.Request) {
	log.Warning("attempt to access metrics endpoint from external address ", req.RemoteAddr)
	http.NotFound(w, req)
}

func dumpMetrics(w http.ResponseWriter, req *http.Request) {
	log.Info("whitelisted requested for metrics endpoint")
	var statsOut = struct {
		Metrics metrics.Registry `json:"metrics"`
		Signers []string         `json:"signers"`
	}{stats.Registry, make([]string, 0, len(signers))}

	for signer := range signers {
		statsOut.Signers = append(statsOut.Signers, signer)
	}

	out, err := json.Marshal(statsOut)
	if err != nil {
		log.Errorf("failed to dump metrics: %v", err)
	}

	w.Write(out)
}
