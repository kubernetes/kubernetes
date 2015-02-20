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

package api

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func OAuthAccessTokenFixture() *OAuthAccessToken {
	return &OAuthAccessToken{
		ObjectMeta: api.ObjectMeta{
			Name:              "accessName",
			UID:               "accessUID",
			ResourceVersion:   "1",
			CreationTimestamp: util.Unix(1000, 0),
			Labels:            map[string]string{"LabelKey": "LabelValue"},
			Annotations:       map[string]string{"AnnotationKey": "AnnotationValue"},
		},
		ClientName:     "myClientName",
		ExpiresIn:      1000,
		Scopes:         []string{"A", "B"},
		RedirectURI:    "http://localhost",
		UserName:       "accessUserName",
		UserUID:        "accessUserUID",
		AuthorizeToken: *OAuthAuthorizeTokenFixture(),
		RefreshToken:   "myRefreshToken",
	}
}
func OAuthAccessTokenListFixture() *OAuthAccessTokenList {
	return &OAuthAccessTokenList{
		ListMeta: api.ListMeta{
			ResourceVersion: "2",
		},
		Items: []OAuthAccessToken{
			*OAuthAccessTokenFixture(),
			*OAuthAccessTokenFixture(),
		},
	}
}
func OAuthAuthorizeTokenFixture() *OAuthAuthorizeToken {
	return &OAuthAuthorizeToken{
		ObjectMeta: api.ObjectMeta{
			Name:              "authorizeName",
			UID:               "authorizeUID",
			ResourceVersion:   "1",
			CreationTimestamp: util.Unix(1000, 0),
			Labels:            map[string]string{"LabelKey": "LabelValue"},
			Annotations:       map[string]string{"AnnotationKey": "AnnotationValue"},
		},
		ClientName:  "myclient2",
		ExpiresIn:   2000,
		Scopes:      []string{"C", "D"},
		RedirectURI: "http://127.0.0.1",
		State:       "mystate",
		UserName:    "authorizeUserName",
		UserUID:     "authorizeUserUID",
	}
}
func OAuthAuthorizeTokenListFixture() *OAuthAuthorizeTokenList {
	return &OAuthAuthorizeTokenList{
		ListMeta: api.ListMeta{
			ResourceVersion: "2",
		},
		Items: []OAuthAuthorizeToken{
			*OAuthAuthorizeTokenFixture(),
			*OAuthAuthorizeTokenFixture(),
		},
	}
}
func OAuthClientFixture() *OAuthClient {
	return &OAuthClient{
		ObjectMeta: api.ObjectMeta{
			Name:              "clientName",
			UID:               "clientUID",
			ResourceVersion:   "1",
			CreationTimestamp: util.Unix(1000, 0),
			Labels:            map[string]string{"LabelKey": "LabelValue"},
			Annotations:       map[string]string{"AnnotationKey": "AnnotationValue"},
		},
		Secret:       "mySecret",
		RedirectURIs: []string{"http://127.0.0.1", "https://www.example.com"},
	}
}
func OAuthClientListFixture() *OAuthClientList {
	return &OAuthClientList{
		ListMeta: api.ListMeta{
			ResourceVersion: "3",
		},
		Items: []OAuthClient{
			*OAuthClientFixture(),
			*OAuthClientFixture(),
		},
	}
}
func OAuthClientAuthorizationFixture() *OAuthClientAuthorization {
	return &OAuthClientAuthorization{
		ObjectMeta: api.ObjectMeta{
			Name:              "clientAuthName",
			UID:               "clientAuthUID",
			ResourceVersion:   "1",
			CreationTimestamp: util.Unix(1000, 0),
			Labels:            map[string]string{"LabelKey": "LabelValue"},
			Annotations:       map[string]string{"AnnotationKey": "AnnotationValue"},
		},
		ClientName: "myClientName",
		UserName:   "myUserName",
		UserUID:    "myUserUID",
		Scopes:     []string{"A", "B"},
	}
}
func OAuthClientAuthorizationListFixture() *OAuthClientAuthorizationList {
	return &OAuthClientAuthorizationList{
		ListMeta: api.ListMeta{
			ResourceVersion: "3",
		},
		Items: []OAuthClientAuthorization{
			*OAuthClientAuthorizationFixture(),
			*OAuthClientAuthorizationFixture(),
		},
	}
}
