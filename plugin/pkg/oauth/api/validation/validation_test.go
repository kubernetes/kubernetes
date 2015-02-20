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

package validation

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
)

func TestValidateClientAuthorization(t *testing.T) {
	errs := ValidateClientAuthorization(&oapi.OAuthClientAuthorization{
		ObjectMeta: api.ObjectMeta{Name: "authName"},
		ClientName: "myclientname",
		UserName:   "myusername",
		UserUID:    "myuseruid",
	})
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		A oapi.OAuthClientAuthorization
		T errors.ValidationErrorType
		F string
	}{
		"zero-length name": {
			A: oapi.OAuthClientAuthorization{
				ClientName: "myclientname",
				UserName:   "myusername",
				UserUID:    "myuseruid",
			},
			T: errors.ValidationErrorTypeRequired,
			F: "name",
		},
	}
	for k, v := range errorCases {
		errs := ValidateClientAuthorization(&v.A)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.A)
			continue
		}
		for i := range errs {
			if errs[i].(*errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(*errors.ValidationError).Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}

func TestValidateClient(t *testing.T) {
	errs := ValidateClient(&oapi.OAuthClient{
		ObjectMeta: api.ObjectMeta{Name: "clientName"},
	})
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		Client oapi.OAuthClient
		T      errors.ValidationErrorType
		F      string
	}{
		"zero-length name": {
			Client: oapi.OAuthClient{},
			T:      errors.ValidationErrorTypeRequired,
			F:      "name",
		},
	}
	for k, v := range errorCases {
		errs := ValidateClient(&v.Client)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.Client)
			continue
		}
		for i := range errs {
			if errs[i].(*errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(*errors.ValidationError).Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}

func TestValidateAccessTokens(t *testing.T) {
	errs := ValidateAccessToken(&oapi.OAuthAccessToken{
		ObjectMeta: api.ObjectMeta{Name: "accessTokenName"},
		ClientName: "myclient",
		UserName:   "myusername",
		UserUID:    "myuseruid",
		AuthorizeToken: oapi.OAuthAuthorizeToken{
			ObjectMeta: api.ObjectMeta{Name: "authTokenName"},
			ClientName: "myclient",
			UserName:   "myusername",
			UserUID:    "myuseruid",
		}})
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		Token oapi.OAuthAccessToken
		T     errors.ValidationErrorType
		F     string
	}{
		"zero-length name": {
			Token: oapi.OAuthAccessToken{
				ClientName: "myclient",
				UserName:   "myusername",
				UserUID:    "myuseruid",
			},
			T: errors.ValidationErrorTypeRequired,
			F: "name",
		},
	}
	for k, v := range errorCases {
		errs := ValidateAccessToken(&v.Token)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.Token)
			continue
		}
		for i := range errs {
			if errs[i].(*errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(*errors.ValidationError).Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}

func TestValidateAuthorizeTokens(t *testing.T) {
	errs := ValidateAuthorizeToken(&oapi.OAuthAuthorizeToken{
		ObjectMeta: api.ObjectMeta{Name: "authorizeTokenName"},
		ClientName: "myclient",
		UserName:   "myusername",
		UserUID:    "myuseruid",
	})
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		Token oapi.OAuthAuthorizeToken
		T     errors.ValidationErrorType
		F     string
	}{
		"zero-length name": {
			Token: oapi.OAuthAuthorizeToken{
				ClientName: "myclient",
				UserName:   "myusername",
				UserUID:    "myuseruid",
			},
			T: errors.ValidationErrorTypeRequired,
			F: "name",
		},
		"zero-length client name": {
			Token: oapi.OAuthAuthorizeToken{
				ObjectMeta: api.ObjectMeta{Name: "authorizeTokenName"},
				UserName:   "myusername",
				UserUID:    "myuseruid",
			},
			T: errors.ValidationErrorTypeRequired,
			F: "clientname",
		},
		"zero-length user name": {
			Token: oapi.OAuthAuthorizeToken{
				ObjectMeta: api.ObjectMeta{Name: "authorizeTokenName"},
				ClientName: "myclient",
				UserUID:    "myuseruid",
			},
			T: errors.ValidationErrorTypeRequired,
			F: "username",
		},
		"zero-length user uid": {
			Token: oapi.OAuthAuthorizeToken{
				ObjectMeta: api.ObjectMeta{Name: "authorizeTokenName"},
				ClientName: "myclient",
				UserName:   "myusername",
			},
			T: errors.ValidationErrorTypeRequired,
			F: "useruid",
		},
	}
	for k, v := range errorCases {
		errs := ValidateAuthorizeToken(&v.Token)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.Token)
			continue
		}
		for i := range errs {
			if errs[i].(*errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(*errors.ValidationError).Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}
