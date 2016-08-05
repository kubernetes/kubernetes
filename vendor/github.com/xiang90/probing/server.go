package probing

import (
	"encoding/json"
	"net/http"
	"time"
)

func NewHandler() http.Handler {
	return &httpHealth{}
}

type httpHealth struct {
}

type Health struct {
	OK  bool
	Now time.Time
}

func (h *httpHealth) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	health := Health{OK: true, Now: time.Now()}
	e := json.NewEncoder(w)
	e.Encode(health)
}
