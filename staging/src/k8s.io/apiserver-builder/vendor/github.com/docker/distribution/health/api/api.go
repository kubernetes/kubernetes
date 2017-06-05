package api

import (
	"errors"
	"net/http"

	"github.com/docker/distribution/health"
)

var (
	updater = health.NewStatusUpdater()
)

// DownHandler registers a manual_http_status that always returns an Error
func DownHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		updater.Update(errors.New("Manual Check"))
	} else {
		w.WriteHeader(http.StatusNotFound)
	}
}

// UpHandler registers a manual_http_status that always returns nil
func UpHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		updater.Update(nil)
	} else {
		w.WriteHeader(http.StatusNotFound)
	}
}

// init sets up the two endpoints to bring the service up and down
func init() {
	health.Register("manual_http_status", updater)
	http.HandleFunc("/debug/health/down", DownHandler)
	http.HandleFunc("/debug/health/up", UpHandler)
}
