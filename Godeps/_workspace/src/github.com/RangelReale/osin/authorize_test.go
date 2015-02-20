package osin

import (
	"net/http"
	"net/url"
	"testing"
)

func TestAuthorizeCode(t *testing.T) {
	sconfig := NewServerConfig()
	sconfig.AllowedAuthorizeTypes = AllowedAuthorizeType{CODE}
	server := NewServer(sconfig, NewTestingStorage())
	server.AuthorizeTokenGen = &TestingAuthorizeTokenGen{}
	resp := server.NewResponse()

	req, err := http.NewRequest("GET", "http://localhost:14000/appauth", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Form = make(url.Values)
	req.Form.Set("response_type", string(CODE))
	req.Form.Set("client_id", "1234")
	req.Form.Set("state", "a")

	if ar := server.HandleAuthorizeRequest(resp, req); ar != nil {
		ar.Authorized = true
		server.FinishAuthorizeRequest(resp, req, ar)
	}

	//fmt.Printf("%+v", resp)

	if resp.IsError && resp.InternalError != nil {
		t.Fatalf("Error in response: %s", resp.InternalError)
	}

	if resp.IsError {
		t.Fatalf("Should not be an error")
	}

	if resp.Type != REDIRECT {
		t.Fatalf("Response should be a redirect")
	}

	if d := resp.Output["code"]; d != "1" {
		t.Fatalf("Unexpected authorization code: %s", d)
	}
}

func TestAuthorizeToken(t *testing.T) {
	sconfig := NewServerConfig()
	sconfig.AllowedAuthorizeTypes = AllowedAuthorizeType{TOKEN}
	server := NewServer(sconfig, NewTestingStorage())
	server.AuthorizeTokenGen = &TestingAuthorizeTokenGen{}
	server.AccessTokenGen = &TestingAccessTokenGen{}
	resp := server.NewResponse()

	req, err := http.NewRequest("GET", "http://localhost:14000/appauth", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Form = make(url.Values)
	req.Form.Set("response_type", string(TOKEN))
	req.Form.Set("client_id", "1234")
	req.Form.Set("state", "a")

	if ar := server.HandleAuthorizeRequest(resp, req); ar != nil {
		ar.Authorized = true
		server.FinishAuthorizeRequest(resp, req, ar)
	}

	//fmt.Printf("%+v", resp)

	if resp.IsError && resp.InternalError != nil {
		t.Fatalf("Error in response: %s", resp.InternalError)
	}

	if resp.IsError {
		t.Fatalf("Should not be an error")
	}

	if resp.Type != REDIRECT || !resp.RedirectInFragment {
		t.Fatalf("Response should be a redirect with fragment")
	}

	if d := resp.Output["access_token"]; d != "1" {
		t.Fatalf("Unexpected access token: %s", d)
	}
}
