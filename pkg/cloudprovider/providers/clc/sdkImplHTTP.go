/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package clc

import (
	"bytes"

	// put this back if cert checking is to be disabled again
	// tls "crypto/tls"

	"encoding/json"

	"fmt"
	"io"
	"net/http"
	"net/http/httputil"

	"github.com/golang/glog"
)

//// requests honor this state, no need to pass in with every call
var bCloseConnections = true
var bDebugRequests = true
var bDebugResponses = true

func SetCloseConnectionMode(b bool) {
	bCloseConnections = b
}

func SetDebugRequestMode(b bool) {
	bDebugRequests = b
}

func SetDebugResponseMode(b bool) {
	bDebugResponses = b
}

//// most funcs here return HttpError, which is an error

const ( // HttpError codes when the error occurred here, not in the remote call.  Hijacking the 000 range for this.
	HTTP_ERROR_UNKNOWN   = 0
	HTTP_ERROR_NOCREDS   = 1
	HTTP_ERROR_CLIENT    = 2
	HTTP_ERROR_NOREQUEST = 3
	HTTP_ERROR_JSON      = 4
)

// no request message body sent.  Response body returned if ret is not nil
func simpleGET(server, uri string, creds Credentials, ret interface{}) (int, error) {
	return invokeHTTP("GET", server, uri, creds, nil, ret)
}

// no request message body sent.  Response body returned if ret is not nil (typically it is nil).
func simpleDELETE(server, uri string, creds Credentials, ret interface{}) (int, error) {
	return invokeHTTP("DELETE", server, uri, creds, nil, ret)
}

// body must be a json-annotated struct, and is marshalled into the request body
func marshalledPOST(server, uri string, creds Credentials, body interface{}, ret interface{}) (int, error) {
	b := new(bytes.Buffer)
	err := json.NewEncoder(b).Encode(body)
	if err != nil {
		return HTTP_ERROR_JSON, clcError(fmt.Sprintf("JSON marshalling failed, err=%s", err.Error()))
	}

	return invokeHTTP("POST", server, uri, creds, b, ret)
}

// body must be a json-annotated struct, and is marshalled into the request body
func marshalledPUT(server, uri string, creds Credentials, body interface{}, ret interface{}) (int, error) {
	b := new(bytes.Buffer)
	err := json.NewEncoder(b).Encode(body)
	if err != nil {
		return HTTP_ERROR_JSON, clcError(fmt.Sprintf("JSON marshalling failed, err=%s", err.Error()))
	}

	return invokeHTTP("PUT", server, uri, creds, b, ret)
}

// body is a JSON string, sent directly as the request body
func simplePOST(server, uri string, creds Credentials, body string, ret interface{}) (int, error) {
	b := bytes.NewBufferString(body)
	return invokeHTTP("POST", server, uri, creds, b, ret)
}

// method to be "GET", "POST", etc.
// server name "api.ctl.io" or "api.loadbalancer.ctl.io"
// uri always starts with /   (we assemble https://<server><uri>)
// creds required for anything except the login call
// body may be be nil

// return int is HTTP response code.  Or an HTTP_ERROR series-000 value, especially if no request was made
// So the possible returns are: (HTTP_ERROR_XXX, err) (non-2xx, err), (2xx, json-err), (2xx, nil)
//      caller couldn't marshal a payload                                                        (0, err)
// 		failed to issue a request.  Does anyone really need the integer code of why not?         (0, err)
//		HTTP response had a failure code.  Retain this code, a 404 might not really be an error. (4xx, err)
//		Success from the HTTP standpoint, but we couldn't unmarshal the JSON response payload    (2xx, err)
//		Success all the way.  Return payload if any is in the ret interface                      (2xx, nil)
// Caller can still just look for (err != nil) - the main reason for the code is that 404 might not be an err for DELETE calls

func invokeHTTP(method, server, uri string, creds Credentials, body io.Reader, ret interface{}) (int, error) {
	if (creds == nil) || !creds.IsValid() {
		return HTTP_ERROR_NOCREDS, clcError("username and/or password not provided")
	}

	full_url := ("https://" + server + uri)
	req, err := http.NewRequest(method, full_url, body)
	if err != nil {
		return HTTP_ERROR_NOREQUEST, err
	} else if body != nil {
		req.Header.Add("Content-Type", "application/json") // incoming body to be a marshaled object already
	}

	req.Header.Add("Host", server) // the reason we take server and uri separately
	req.Header.Add("Accept", "application/json")
	creds.AddAuthHeader(req)

	if bCloseConnections {
		req.Header.Add("Connection", "close")
	}

	if bDebugRequests { // do not call invokeHTTP to perform auth, because this might log the username/password message body
		glog.Info("---- initiating SDK HTTP request ----")
		v, _ := httputil.DumpRequestOut(req, true)
		glog.Info(string(v))
	}

	// this should be the normal code
	resp, err := http.DefaultClient.Do(req) // execute the call

	// instead, we have this which tolerates bad certs [fixme both here and in CredsLogin]
	// tlscfg := &tls.Config{InsecureSkipVerify: true} // true means to skip the verification
	// transp := &http.Transport{TLSClientConfig: tlscfg}
	// client := &http.Client{Transport: transp}
	// resp, err := client.Do(req)
	// end of tolerating bad certs.  Do not keep this code - it allows MITM etc. attacks

	defer resp.Body.Close() // avoid CLOSE_WAIT state

	if bDebugResponses {
		vv, _ := httputil.DumpResponse(resp, true)
		glog.Info(string(vv))
	}

	if err != nil { // failed HTTP call
		return HTTP_ERROR_CLIENT, err
	}

	if resp.StatusCode == 401 { // Unauthorized.  Not a failure yet, perhaps we can reauth.  This is why we need a whole Credentials and not just the token

		creds.CredsReauth()
		if creds.IsValid() {
			req.Header.Del("Authorization")
			creds.AddAuthHeader(req)
			resp, err = http.DefaultClient.Do(req) // not :=
		}
	}

	if !isSuccess(resp.StatusCode) {
		// stat := fmt.Sprintf("received HTTP response code %d\n", resp.StatusCode)

		if !bDebugRequests {
			glog.Info("dumping this request, after the fact")
			v, _ := httputil.DumpRequestOut(req, true)
			glog.Info(string(v))
		}

		if !bDebugResponses {
			vv, _ := httputil.DumpResponse(resp, true)
			glog.Info(string(vv))
		}

		return resp.StatusCode, clcError(fmt.Sprintf("HTTP call failed, status=%d", resp.StatusCode))
	}

	if ret != nil { // permit methods without a response body, or calls that ignore the body and just look for status
		err = json.NewDecoder(resp.Body).Decode(ret)

		if err != nil { // status is a 200-series success code, because HTTP returned a proper payload.  We just couldn't interpret it.
			return resp.StatusCode, clcError(fmt.Sprintf("JSON decode failed: err=%s", err))
		}
	}

	return resp.StatusCode, nil // success
}

func isSuccess(httpStatus int) bool {
	return (httpStatus >= 200) && (httpStatus <= 299)
}
