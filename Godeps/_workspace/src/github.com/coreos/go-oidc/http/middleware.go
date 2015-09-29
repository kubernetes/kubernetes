package http

import (
	"net/http"
)

type LoggingMiddleware struct {
	Next http.Handler
}

func (l *LoggingMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	log.Infof("HTTP %s %v", r.Method, r.URL)
	l.Next.ServeHTTP(w, r)
}
