package logrus_sentry

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Sirupsen/logrus"
	"github.com/getsentry/raven-go"
)

const (
	message     = "error message"
	server_name = "testserver.internal"
	logger_name = "test.logger"
)

func getTestLogger() *logrus.Logger {
	l := logrus.New()
	l.Out = ioutil.Discard
	return l
}

func WithTestDSN(t *testing.T, tf func(string, <-chan *raven.Packet)) {
	pch := make(chan *raven.Packet, 1)
	s := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		defer req.Body.Close()
		d := json.NewDecoder(req.Body)
		p := &raven.Packet{}
		err := d.Decode(p)
		if err != nil {
			t.Fatal(err.Error())
		}

		pch <- p
	}))
	defer s.Close()

	fragments := strings.SplitN(s.URL, "://", 2)
	dsn := fmt.Sprintf(
		"%s://public:secret@%s/sentry/project-id",
		fragments[0],
		fragments[1],
	)
	tf(dsn, pch)
}

func TestSpecialFields(t *testing.T) {
	WithTestDSN(t, func(dsn string, pch <-chan *raven.Packet) {
		logger := getTestLogger()

		hook, err := NewSentryHook(dsn, []logrus.Level{
			logrus.ErrorLevel,
		})

		if err != nil {
			t.Fatal(err.Error())
		}
		logger.Hooks.Add(hook)
		logger.WithFields(logrus.Fields{
			"server_name": server_name,
			"logger":      logger_name,
		}).Error(message)

		packet := <-pch
		if packet.Logger != logger_name {
			t.Errorf("logger should have been %s, was %s", logger_name, packet.Logger)
		}

		if packet.ServerName != server_name {
			t.Errorf("server_name should have been %s, was %s", server_name, packet.ServerName)
		}
	})
}

func TestSentryHandler(t *testing.T) {
	WithTestDSN(t, func(dsn string, pch <-chan *raven.Packet) {
		logger := getTestLogger()
		hook, err := NewSentryHook(dsn, []logrus.Level{
			logrus.ErrorLevel,
		})
		if err != nil {
			t.Fatal(err.Error())
		}
		logger.Hooks.Add(hook)

		logger.Error(message)
		packet := <-pch
		if packet.Message != message {
			t.Errorf("message should have been %s, was %s", message, packet.Message)
		}
	})
}
