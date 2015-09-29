package health

import (
	"expvar"
	"fmt"
	"log"
	"net/http"

	"github.com/coreos/pkg/httputil"
)

// Checkables should return nil when the thing they are checking is healthy, and an error otherwise.
type Checkable interface {
	Healthy() error
}

// Checker provides a way to make an endpoint which can be probed for system health.
type Checker struct {
	// Checks are the Checkables to be checked when probing.
	Checks []Checkable

	// Unhealthyhandler is called when one or more of the checks are unhealthy.
	// If not provided DefaultUnhealthyHandler is called.
	UnhealthyHandler UnhealthyHandler

	// HealthyHandler is called when all checks are healthy.
	// If not provided, DefaultHealthyHandler is called.
	HealthyHandler http.HandlerFunc
}

func (c Checker) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	unhealthyHandler := c.UnhealthyHandler
	if unhealthyHandler == nil {
		unhealthyHandler = DefaultUnhealthyHandler
	}

	successHandler := c.HealthyHandler
	if successHandler == nil {
		successHandler = DefaultHealthyHandler
	}

	if r.Method != "GET" {
		w.Header().Set("Allow", "GET")
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if err := Check(c.Checks); err != nil {
		unhealthyHandler(w, r, err)
		return
	}

	successHandler(w, r)
}

type UnhealthyHandler func(w http.ResponseWriter, r *http.Request, err error)

type StatusResponse struct {
	Status  string                 `json:"status"`
	Details *StatusResponseDetails `json:"details,omitempty"`
}

type StatusResponseDetails struct {
	Code    int    `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

func Check(checks []Checkable) (err error) {
	errs := []error{}
	for _, c := range checks {
		if e := c.Healthy(); e != nil {
			errs = append(errs, e)
		}
	}

	switch len(errs) {
	case 0:
		err = nil
	case 1:
		err = errs[0]
	default:
		err = fmt.Errorf("multiple health check failure: %v", errs)
	}

	return
}

func DefaultHealthyHandler(w http.ResponseWriter, r *http.Request) {
	err := httputil.WriteJSONResponse(w, http.StatusOK, StatusResponse{
		Status: "ok",
	})
	if err != nil {
		// TODO(bobbyrullo): replace with logging from new logging pkg,
		// once it lands.
		log.Printf("Failed to write JSON response: %v", err)
	}
}

func DefaultUnhealthyHandler(w http.ResponseWriter, r *http.Request, err error) {
	writeErr := httputil.WriteJSONResponse(w, http.StatusInternalServerError, StatusResponse{
		Status: "error",
		Details: &StatusResponseDetails{
			Code:    http.StatusInternalServerError,
			Message: err.Error(),
		},
	})
	if writeErr != nil {
		// TODO(bobbyrullo): replace with logging from new logging pkg,
		// once it lands.
		log.Printf("Failed to write JSON response: %v", err)
	}
}

// ExpvarHandler is copied from https://golang.org/src/expvar/expvar.go, where it's sadly unexported.
func ExpvarHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	fmt.Fprintf(w, "{\n")
	first := true
	expvar.Do(func(kv expvar.KeyValue) {
		if !first {
			fmt.Fprintf(w, ",\n")
		}
		first = false
		fmt.Fprintf(w, "%q: %s", kv.Key, kv.Value)
	})
	fmt.Fprintf(w, "\n}\n")
}
