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
	// tls "crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/golang/glog"
)

type Credentials interface {
	GetUsername() string
	GetAccount() string
	GetLocation() string
	IsValid() bool
	CredsLogout(clearUser bool) error
	CredsReauth() error
	CredsLogin(username string, password string) error
	CredsFromEnv() error
	AddAuthHeader(req *http.Request)
}

//// Credentials is returned from the login func, and used by everything else
type implCreds struct { // implements the Credentials methods
	AuthServer string
	AuthURI    string

	Username      string
	Password      string // kept because we need reauth, especially when a token expires
	AccountAlias  string
	LocationAlias string // do we need this?
	BearerToken   string
}

func MakeEmptyCreds(srv string, uri string) Credentials {
	return &implCreds{
		AuthServer:    srv,
		AuthURI:       uri,
		Username:      "",
		Password:      "",
		AccountAlias:  "",
		LocationAlias: "",
		BearerToken:   "",
	}
}

func (creds *implCreds) GetUsername() string {
	return creds.Username
}

func (creds *implCreds) GetAccount() string {
	return creds.AccountAlias
}

func (creds *implCreds) GetLocation() string {
	return creds.LocationAlias
}

func (creds *implCreds) IsValid() bool {
	return (creds.AccountAlias != "") && (creds.BearerToken != "")
}

func (creds *implCreds) AddAuthHeader(req *http.Request) {
	req.Header.Add("Authorization", ("Bearer " + creds.BearerToken))
}

func (creds *implCreds) CredsLogout(clearUser bool) error {
	creds.AccountAlias = ""
	creds.LocationAlias = ""
	creds.BearerToken = ""

	if clearUser {
		creds.Username = ""
		creds.Password = ""
	}

	return nil // currently logout involves no outside call. We don't really invalidate the token, we just forget our copy of it
}

func (creds *implCreds) CredsReauth() error {
	user := creds.Username
	pass := creds.Password

	e := creds.CredsLogout(true)
	if e != nil {
		return e
	}

	return creds.CredsLogin(user, pass)
}

func (creds *implCreds) CredsFromEnv() error {
	envUsername := os.Getenv("CLC_API_USERNAME")
	envAccount := os.Getenv("CLC_API_ACCOUNT")
	envLocation := os.Getenv("CLC_API_LOCATION")
	envToken := os.Getenv("CLC_API_TOKEN")
	envPassword := os.Getenv("CLC_API_PASSWORD")

	if envUsername == "" {
		return clcError("cannot auth from env, username missing")
	}

	if (envAccount == "") || (envToken == "") { // cannot work as is
		if envPassword == "" {
			return clcError("cannot auth from env, token and password both missing")
		}

		return creds.CredsLogin(envUsername, envPassword) // may or may not work
	}

	creds.Username = envUsername
	creds.Password = envPassword
	creds.AccountAlias = envAccount
	creds.LocationAlias = envLocation
	creds.BearerToken = envToken
	return nil // no error
}

type AuthLoginRequestJSON struct { // actually this is unused, as we simply sprintf the string
	Username string `json:"username"`
	Password string `json:"password"`
}

type AuthLoginResponseJSON struct {
	Username      string   `json:"username"`
	AccountAlias  string   `json:"accountAlias"`
	LocationAlias string   `json:"locationAlias"`
	Roles         []string `json:"roles"`
	BearerToken   string   `json:"bearerToken"`
}

// overwrites all previous creds info, including username
func (creds *implCreds) CredsLogin(username string, password string) error {
	e := creds.CredsLogout(true)
	if e != nil {
		return e
	}

	if username == "" {
		return clcError("cannot log in, username not provided")
	} else if password == "" {
		return clcError("cannot log in, password not provided")
	}

	body := fmt.Sprintf("{\"username\":\"%s\",\"password\":\"%s\"}", username, password)
	b := bytes.NewBufferString(body)

	authresp := AuthLoginResponseJSON{}

	// but don't call invokeHTTP here, because we're a layer underneath that.
	full_url := ("https://" + creds.AuthServer + creds.AuthURI)
	req, err := http.NewRequest("POST", full_url, b)
	if err != nil {
		return clcError(fmt.Sprintf("could not create HTTP request: url=%s, err=%s", full_url, err.Error()))
	}

	req.Header.Add("Content-Type", "application/json") // incoming body to be a marshaled object already
	req.Header.Add("Host", creds.AuthServer)           // the reason we take server and uri separately
	req.Header.Add("Accept", "application/json")
	// do not send Authorization - we are asking to become authorized
	req.Header.Add("Connection", "close")

	// this should be the normal code
	resp,err := http.DefaultClient.Do(req)	// execute the call

	// instead, we have this which tolerates bad certs [fixme both here and in invokeHTTP]
	// tlscfg := &tls.Config{InsecureSkipVerify: true} // true means to skip the verification
	// transp := &http.Transport{TLSClientConfig: tlscfg}
	// client := &http.Client{Transport: transp}
	// resp, err := client.Do(req)
	// end of tolerating bad certs.  Do not keep this code - it allows MITM etc. attacks

	if err != nil { // failed HTTP call
		return clcError(fmt.Sprintf("authorization call failed, err=%s", err.Error()))
	}

	if (resp.StatusCode < 200) || (resp.StatusCode >= 300) {
		return clcError(fmt.Sprintf("authorization call failed, HTTP status=%d", resp.StatusCode))
	}

	err = json.NewDecoder(resp.Body).Decode(&authresp)
	if err != nil {
		return clcError(fmt.Sprintf("JSON decode failed, err=%s", err.Error()))
	} // otherwise we now have an auth response in authresp

	glog.Info(fmt.Sprintf("assigning new token, do this:  export CLC_API_TOKEN=%s\n", authresp.BearerToken))
	glog.Info(fmt.Sprintf("also CLC_API_USERNAME=%s  CLC_API_ACCOUNT=%s  CLC_API_LOCATION=%s\n", authresp.Username, authresp.AccountAlias, authresp.LocationAlias))

	creds.Username = authresp.Username
	creds.Password = password
	creds.AccountAlias = authresp.AccountAlias
	creds.LocationAlias = authresp.LocationAlias
	creds.BearerToken = authresp.BearerToken

	return nil // no error
}

// whole clc package uses this, it's just here in creds because this is the bottom of the dependency stack
func clcError(content string) error { // caller probably to use fmt.Sprintf to make the input string
	if content == "" {
		content = "<error text not available>"
	}

	full_err := "CLC: " + content

	glog.Info(full_err)
	return errors.New(full_err)
}
