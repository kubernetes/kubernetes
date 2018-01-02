package testutil

import (
	"io"
	"net"
	"net/http"
	"reflect"
	"testing"

	log "github.com/Sirupsen/logrus"
)

func Expect(t *testing.T, a interface{}, b interface{}) {
	if a != b {
		t.Errorf("Expected %v (type %v) - Got %v (type %v)", b, reflect.TypeOf(b), a, reflect.TypeOf(a))
	}
}

func Refute(t *testing.T, a interface{}, b interface{}) {
	if a == b {
		t.Errorf("Did not expect %v (type %v) - Got %v (type %v)", b, reflect.TypeOf(b), a, reflect.TypeOf(a))
	}
}

// NewTestingServer - testing server and teardown func
func NewTestingServer(status int, body string) (*http.Server, func()) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		w.Header().Set("Content-Type", "application/json")
		if body != "" {
			io.WriteString(w, body)
		}
	}

	server := http.Server{
		// Addr:    port,
		Handler: http.HandlerFunc(handler),
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.WithFields(log.Fields{
			"error": err,
		}).Fatal("failed to create new testing server")
	}
	server.Addr = listener.Addr().String()

	stoppable := Handle(listener)
	go server.Serve(stoppable)

	teardown := func() {
		stoppable.Stop <- true
	}

	return &server, teardown
}
