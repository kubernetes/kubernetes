package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/emicklei/go-restful"
)

var (
	Result string
)

func TestRouteExtractParameter(t *testing.T) {
	// setup service
	ws := new(restful.WebService)
	ws.Consumes(restful.MIME_XML)
	ws.Route(ws.GET("/test/{param}").To(DummyHandler))
	restful.Add(ws)

	// setup request + writer
	bodyReader := strings.NewReader("<Sample><Value>42</Value></Sample>")
	httpRequest, _ := http.NewRequest("GET", "/test/THIS", bodyReader)
	httpRequest.Header.Set("Content-Type", restful.MIME_XML)
	httpWriter := httptest.NewRecorder()

	// run
	restful.DefaultContainer.ServeHTTP(httpWriter, httpRequest)

	if Result != "THIS" {
		t.Fatalf("Result is actually: %s", Result)
	}
}

func DummyHandler(rq *restful.Request, rp *restful.Response) {
	Result = rq.PathParameter("param")
}
