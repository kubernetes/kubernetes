package martini

import (
	"bytes"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"
)

func Test_Recovery(t *testing.T) {
	buff := bytes.NewBufferString("")
	recorder := httptest.NewRecorder()

	setENV(Dev)
	m := New()
	// replace log for testing
	m.Map(log.New(buff, "[martini] ", 0))
	m.Use(func(res http.ResponseWriter, req *http.Request) {
		res.Header().Set("Content-Type", "unpredictable")
	})
	m.Use(Recovery())
	m.Use(func(res http.ResponseWriter, req *http.Request) {
		panic("here is a panic!")
	})
	m.ServeHTTP(recorder, (*http.Request)(nil))
	expect(t, recorder.Code, http.StatusInternalServerError)
	expect(t, recorder.HeaderMap.Get("Content-Type"), "text/html")
	refute(t, recorder.Body.Len(), 0)
	refute(t, len(buff.String()), 0)
}

func Test_Recovery_ResponseWriter(t *testing.T) {
	recorder := httptest.NewRecorder()
	recorder2 := httptest.NewRecorder()

	setENV(Dev)
	m := New()
	m.Use(Recovery())
	m.Use(func(c Context) {
		c.MapTo(recorder2, (*http.ResponseWriter)(nil))
		panic("here is a panic!")
	})
	m.ServeHTTP(recorder, (*http.Request)(nil))

	expect(t, recorder2.Code, http.StatusInternalServerError)
	expect(t, recorder2.HeaderMap.Get("Content-Type"), "text/html")
	refute(t, recorder2.Body.Len(), 0)
}
