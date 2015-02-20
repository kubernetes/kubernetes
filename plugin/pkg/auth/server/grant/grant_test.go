/*
Copyright 2014 Google Inc. All rights reserved.

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

package grant

import (
	"errors"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/csrf"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	oauthclient "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/client"
)

type testAuth struct {
	User    user.Info
	Success bool
	Err     error
}

func (t *testAuth) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	return t.User, t.Success, t.Err
}

// auth builders
func goodAuth(username string) *testAuth {
	return &testAuth{Success: true, User: &user.DefaultInfo{Name: username}}
}
func badAuth(err error) *testAuth {
	return &testAuth{Success: false, User: nil, Err: err}
}

// oauth client builders
func goodClient(clientID string, redirectURIs []string) *oauthclient.Fake {
	return &oauthclient.Fake{Client: client("myclient", []string{"myredirect"})}
}
func badClient(err error) *oauthclient.Fake {
	return &oauthclient.Fake{ClientGetErr: err}
}

// oauth api object builders
func client(clientID string, redirectURIs []string) oapi.OAuthClient {
	client := oapi.OAuthClient{Secret: "mysecret", RedirectURIs: redirectURIs}
	client.Name = clientID
	return client
}

func clientAuthorization(scopes []string) oapi.OAuthClientAuthorization {
	auth := oapi.OAuthClientAuthorization{
		UserName:   "existingUserName",
		UserUID:    "existingUserUID",
		ClientName: "existingClientName",
		Scopes:     scopes,
	}
	auth.Name = "existingID"
	return auth
}

func TestGrant(t *testing.T) {
	testCases := map[string]struct {
		CSRF   csrf.CSRF
		Auth   *testAuth
		Client *oauthclient.Fake

		Path       string
		PostValues url.Values

		ExpectStatusCode        int
		ExpectCreatedAuthScopes []string
		ExpectUpdatedAuthScopes []string
		ExpectRedirect          string
		ExpectContains          []string
		ExpectThen              string
	}{
		"display form": {
			CSRF:   &csrf.FakeCSRF{Token: "test"},
			Auth:   goodAuth("username"),
			Client: goodClient("myclient", []string{"myredirect"}),
			Path:   "/grant?client_id=myclient&scopes=myscope1%20myscope2&redirect_uri=/myredirect&then=/authorize",

			ExpectStatusCode: 200,
			ExpectContains: []string{
				`action="/grant"`,
				`name="csrf" value="test"`,
				`name="client_id" value="myclient"`,
				`name="scopes" value="myscope1 myscope2"`,
				`name="redirect_uri" value="/myredirect"`,
				`name="then" value="/authorize"`,
			},
		},

		"Unauthenticated with redirect": {
			CSRF: &csrf.FakeCSRF{Token: "test"},
			Auth: badAuth(nil),
			Path: "/grant?then=/authorize",

			ExpectStatusCode: 302,
			ExpectRedirect:   "/authorize",
		},

		"Unauthenticated without redirect": {
			CSRF: &csrf.FakeCSRF{Token: "test"},
			Auth: badAuth(nil),
			Path: "/grant",

			ExpectStatusCode: 200,
			ExpectContains:   []string{"reauthenticate"},
		},

		"Auth error with redirect": {
			CSRF: &csrf.FakeCSRF{Token: "test"},
			Auth: badAuth(errors.New("Auth error")),
			Path: "/grant?then=/authorize",

			ExpectStatusCode: 302,
			ExpectRedirect:   "/authorize",
		},

		"Auth error without redirect": {
			CSRF: &csrf.FakeCSRF{Token: "test"},
			Auth: badAuth(errors.New("Auth error")),
			Path: "/grant",

			ExpectStatusCode: 200,
			ExpectContains:   []string{"reauthenticate"},
		},

		"error when POST fails CSRF": {
			CSRF:   &csrf.FakeCSRF{Token: "test"},
			Auth:   goodAuth("username"),
			Client: goodClient("myclient", []string{"myredirect"}),
			Path:   "/grant",
			PostValues: url.Values{
				clientIDParam:    {"myclient"},
				scopesParam:      {"myscope1 myscope2"},
				redirectURIParam: {"/myredirect"},
				thenParam:        {"/authorize"},
				csrfParam:        {"wrong"},
			},

			ExpectStatusCode: 200,
			ExpectContains:   []string{"CSRF"},
		},

		"error displaying form with invalid client": {
			CSRF:   &csrf.FakeCSRF{Token: "test"},
			Auth:   goodAuth("username"),
			Client: badClient(errors.New("bad client")),
			Path:   "/grant",

			ExpectStatusCode: 200,
			ExpectContains:   []string{"find client"},
		},

		"error submitting form with invalid client": {
			CSRF:   &csrf.FakeCSRF{Token: "test"},
			Auth:   goodAuth("username"),
			Client: badClient(errors.New("bad client")),
			Path:   "/grant",
			PostValues: url.Values{
				clientIDParam:    {"myclient"},
				scopesParam:      {"myscope1 myscope2"},
				redirectURIParam: {"/myredirect"},
				thenParam:        {"/authorize"},
				csrfParam:        {"test"},
			},

			ExpectStatusCode: 200,
			ExpectContains:   []string{"find client"},
		},

		"grant denied with redirect": {
			CSRF:   &csrf.FakeCSRF{Token: "test"},
			Auth:   goodAuth("username"),
			Client: goodClient("myclient", []string{"myredirect"}),
			Path:   "/grant",
			PostValues: url.Values{
				denyParam:        {"Reject"},
				clientIDParam:    {"myclient"},
				scopesParam:      {"myscope1 myscope2"},
				redirectURIParam: {"/myredirect"},
				thenParam:        {"/authorize?error=existing_error&other_param=other_value"},
				csrfParam:        {"test"},
			},

			ExpectStatusCode: 302,
			ExpectRedirect:   "/authorize?error=grant_denied&other_param=other_value",
		},

		"grant denied without redirect": {
			CSRF:   &csrf.FakeCSRF{Token: "test"},
			Auth:   goodAuth("username"),
			Client: goodClient("myclient", []string{"myredirect"}),
			Path:   "/grant",
			PostValues: url.Values{
				denyParam:        {"Reject"},
				clientIDParam:    {"myclient"},
				scopesParam:      {"myscope1 myscope2"},
				redirectURIParam: {"/myredirect"},
				csrfParam:        {"test"},
			},

			ExpectStatusCode: 200,
			ExpectContains: []string{
				"denied",
				"no redirect",
			},
		},

		"successful create grant with redirect": {
			CSRF: &csrf.FakeCSRF{Token: "test"},
			Auth: goodAuth("username"),
			Client: &oauthclient.Fake{
				Client: client("myclient", []string{"myredirect"}),
				ClientAuthorizationGetErr: errors.New("missing auth"),
			},
			Path: "/grant",
			PostValues: url.Values{
				approveParam:     {"Approve"},
				clientIDParam:    {"myclient"},
				scopesParam:      {"myscope1 myscope2"},
				redirectURIParam: {"/myredirect"},
				thenParam:        {"/authorize"},
				csrfParam:        {"test"},
			},

			ExpectStatusCode:        302,
			ExpectCreatedAuthScopes: []string{"myscope1", "myscope2"},
			ExpectRedirect:          "/authorize",
		},

		"successful create grant without redirect": {
			CSRF: &csrf.FakeCSRF{Token: "test"},
			Auth: goodAuth("username"),
			Client: &oauthclient.Fake{
				Client: client("myclient", []string{"myredirect"}),
				ClientAuthorizationGetErr: errors.New("missing auth"),
			},
			Path: "/grant",
			PostValues: url.Values{
				approveParam:     {"Approve"},
				clientIDParam:    {"myclient"},
				scopesParam:      {"myscope1 myscope2"},
				redirectURIParam: {"/myredirect"},
				csrfParam:        {"test"},
			},

			ExpectStatusCode:        200,
			ExpectCreatedAuthScopes: []string{"myscope1", "myscope2"},
			ExpectContains: []string{
				"granted",
				"no redirect",
			},
		},

		"successful update grant with identical scopes": {
			CSRF: &csrf.FakeCSRF{Token: "test"},
			Auth: goodAuth("username"),
			Client: &oauthclient.Fake{
				Client:              client("myclient", []string{"myredirect"}),
				ClientAuthorization: clientAuthorization([]string{"myscope2", "myscope1"}),
			},
			Path: "/grant",
			PostValues: url.Values{
				approveParam:     {"Approve"},
				clientIDParam:    {"myclient"},
				scopesParam:      {"myscope1 myscope2"},
				redirectURIParam: {"/myredirect"},
				thenParam:        {"/authorize"},
				csrfParam:        {"test"},
			},

			ExpectStatusCode:        302,
			ExpectUpdatedAuthScopes: []string{"myscope1", "myscope2"},
			ExpectRedirect:          "/authorize",
		},

		"successful update grant with additional scopes": {
			CSRF: &csrf.FakeCSRF{Token: "test"},
			Auth: goodAuth("username"),
			Client: &oauthclient.Fake{
				Client:              client("myclient", []string{"myredirect"}),
				ClientAuthorization: clientAuthorization([]string{"existingscope2", "existingscope1"}),
			},
			Path: "/grant",
			PostValues: url.Values{
				approveParam:     {"Approve"},
				clientIDParam:    {"myclient"},
				scopesParam:      {"newscope1 existingscope1"},
				redirectURIParam: {"/myredirect"},
				thenParam:        {"/authorize"},
				csrfParam:        {"test"},
			},

			ExpectStatusCode:        302,
			ExpectUpdatedAuthScopes: []string{"existingscope1", "existingscope2", "newscope1"},
			ExpectRedirect:          "/authorize",
		},
	}

	for k, testCase := range testCases {
		server := httptest.NewServer(NewGrant(testCase.CSRF, testCase.Auth, DefaultGrantFormRenderer, testCase.Client))

		var resp *http.Response
		if testCase.PostValues != nil {
			r, err := postForm(server.URL+testCase.Path, testCase.PostValues)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", k, err)
				continue
			}
			resp = r
		} else {
			r, err := getUrl(server.URL + testCase.Path)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", k, err)
				continue
			}
			resp = r
		}
		defer resp.Body.Close()

		if testCase.ExpectStatusCode != 0 && testCase.ExpectStatusCode != resp.StatusCode {
			t.Errorf("%s: unexpected response: %#v", k, resp)
			continue
		}

		if len(testCase.ExpectCreatedAuthScopes) > 0 {
			createdAuths := []*oapi.OAuthClientAuthorization{}
			visitActions(testCase.Client.Actions, "create-client-authorization", func(value interface{}) {
				auth, ok := value.(*oapi.OAuthClientAuthorization)
				if !ok {
					t.Errorf("Incorrect type")
				}
				if ok {
					createdAuths = append(createdAuths, auth)
				}
			})
			if len(createdAuths) != 1 {
				t.Errorf("%s: expected created auth, got %#v", k, createdAuths)
				continue
			}
			if !reflect.DeepEqual(testCase.ExpectCreatedAuthScopes, createdAuths[0].Scopes) {
				t.Errorf("%s: expected created scopes %v, got %v", k, testCase.ExpectCreatedAuthScopes, createdAuths[0].Scopes)
			}
		}

		if len(testCase.ExpectUpdatedAuthScopes) > 0 {
			updatedAuths := []*oapi.OAuthClientAuthorization{}
			visitActions(testCase.Client.Actions, "update-client-authorization", func(value interface{}) {
				auth, ok := value.(*oapi.OAuthClientAuthorization)
				if !ok {
					t.Errorf("Incorrect type")
				}
				if ok {
					updatedAuths = append(updatedAuths, auth)
				}
			})
			if len(updatedAuths) != 1 {
				t.Errorf("%s: expected updated auth, got %#v", k, updatedAuths)
				continue
			}
			if !reflect.DeepEqual(testCase.ExpectUpdatedAuthScopes, updatedAuths[0].Scopes) {
				t.Errorf("%s: expected updated scopes %v, got %v", k, testCase.ExpectUpdatedAuthScopes, updatedAuths[0].Scopes)
			}
		}

		if testCase.ExpectRedirect != "" {
			uri, err := resp.Location()
			if err != nil {
				t.Errorf("%s: unexpected error: %v", k, err)
				continue
			}
			if uri.String() != server.URL+testCase.ExpectRedirect {
				t.Errorf("%s: unexpected redirect: %s", k, uri.String())
			}
		}

		if len(testCase.ExpectContains) > 0 {
			data, _ := ioutil.ReadAll(resp.Body)
			body := string(data)
			for i := range testCase.ExpectContains {
				if !strings.Contains(body, testCase.ExpectContains[i]) {
					t.Errorf("%s: did not find expected value %s: %s", k, testCase.ExpectContains[i], body)
					continue
				}
			}
		}
	}
}

func visitActions(actions []oauthclient.FakeAction, name string, visitor func(interface{})) {
	for _, a := range actions {
		if a.Action == name {
			visitor(a.Value)
		}
	}
}

func postForm(url string, body url.Values) (resp *http.Response, err error) {
	tr := &http.Transport{}
	req, err := http.NewRequest("POST", url, strings.NewReader(body.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return tr.RoundTrip(req)
}

func getUrl(url string) (resp *http.Response, err error) {
	tr := &http.Transport{}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	return tr.RoundTrip(req)
}
