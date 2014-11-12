package main

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful"
)

// This example show how to test one particular RouteFunction (getIt)
// It uses the httptest.ResponseRecorder to capture output

func getIt(req *restful.Request, resp *restful.Response) {
	resp.WriteHeader(404)
}

func TestCallFunction(t *testing.T) {
	httpReq, _ := http.NewRequest("GET", "/", nil)
	req := restful.NewRequest(httpReq)

	recorder := new(httptest.ResponseRecorder)
	resp := restful.NewResponse(recoder)

	getIt(req, resp)
	if recorder.Code != 404 {
		t.Logf("Missing or wrong status code:%d", recorder.Code)
	}
}
