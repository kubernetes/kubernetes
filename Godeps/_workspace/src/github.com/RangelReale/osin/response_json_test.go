package osin

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestResponseJSON(t *testing.T) {
	req, err := http.NewRequest("GET", "http://localhost:14000/appauth", nil)
	if err != nil {
		t.Fatal(err)
	}

	w := httptest.NewRecorder()

	r := NewResponse(NewTestingStorage())
	r.Output["access_token"] = "1234"
	r.Output["token_type"] = "5678"

	err = OutputJSON(r, w, req)
	if err != nil {
		t.Fatalf("Error outputting json: %s", err)
	}

	//fmt.Printf("%d - %s - %+v", w.Code, w.Body.String(), w.HeaderMap)

	if w.Code != 200 {
		t.Fatalf("Invalid response code for output: %d", w.Code)
	}

	if w.HeaderMap.Get("Content-Type") != "application/json" {
		t.Fatalf("Result from json must be application/json")
	}

	// parse output json
	output := make(map[string]interface{})
	if err := json.Unmarshal(w.Body.Bytes(), &output); err != nil {
		t.Fatalf("Could not decode output json: %s", err)
	}

	if d, ok := output["access_token"]; !ok || d != "1234" {
		t.Fatalf("Invalid or not found output data: access_token=%s", d)
	}

	if d, ok := output["token_type"]; !ok || d != "5678" {
		t.Fatalf("Invalid or not found output data: token_type=%s", d)
	}
}

func TestErrorResponseJSON(t *testing.T) {
	req, err := http.NewRequest("GET", "http://localhost:14000/appauth", nil)
	if err != nil {
		t.Fatal(err)
	}

	w := httptest.NewRecorder()

	r := NewResponse(NewTestingStorage())
	r.ErrorStatusCode = 500
	r.SetError(E_INVALID_REQUEST, "")

	err = OutputJSON(r, w, req)
	if err != nil {
		t.Fatalf("Error outputting json: %s", err)
	}

	//fmt.Printf("%d - %s - %+v", w.Code, w.Body.String(), w.HeaderMap)

	if w.Code != 500 {
		t.Fatalf("Invalid response code for error output: %d", w.Code)
	}

	if w.HeaderMap.Get("Content-Type") != "application/json" {
		t.Fatalf("Result from json must be application/json")
	}

	// parse output json
	output := make(map[string]interface{})
	if err := json.Unmarshal(w.Body.Bytes(), &output); err != nil {
		t.Fatalf("Could not decode output json: %s", err)
	}

	if d, ok := output["error"]; !ok || d != E_INVALID_REQUEST {
		t.Fatalf("Invalid or not found output data: error=%s", d)
	}
}

func TestRedirectResponseJSON(t *testing.T) {
	req, err := http.NewRequest("GET", "http://localhost:14000/appauth", nil)
	if err != nil {
		t.Fatal(err)
	}

	w := httptest.NewRecorder()

	r := NewResponse(NewTestingStorage())
	r.SetRedirect("http://localhost:14000")

	err = OutputJSON(r, w, req)
	if err != nil {
		t.Fatalf("Error outputting json: %s", err)
	}

	//fmt.Printf("%d - %s - %+v", w.Code, w.Body.String(), w.HeaderMap)

	if w.Code != 302 {
		t.Fatalf("Invalid response code for redirect output: %d", w.Code)
	}

	if w.HeaderMap.Get("Location") != "http://localhost:14000" {
		t.Fatalf("Invalid response location url: %s", w.HeaderMap.Get("Location"))
	}
}
