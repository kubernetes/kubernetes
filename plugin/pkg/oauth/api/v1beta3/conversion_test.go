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

package v1beta3_test

import (
	"reflect"
	"testing"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	current "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	onewer "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	ocurrent "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api/v1beta3"
)

func TestAccessToken(t *testing.T) {
	testObject(
		onewer.OAuthAccessTokenFixture(),
		func() runtime.Object { return &onewer.OAuthAccessToken{} },
		func() runtime.Object { return &ocurrent.OAuthAccessToken{} },
		t,
	)
}

func TestAccessTokenList(t *testing.T) {
	testObject(
		onewer.OAuthAccessTokenListFixture(),
		func() runtime.Object { return &onewer.OAuthAccessTokenList{} },
		func() runtime.Object { return &ocurrent.OAuthAccessTokenList{} },
		t,
	)
}

func TestAuthorizeToken(t *testing.T) {
	testObject(
		onewer.OAuthAuthorizeTokenFixture(),
		func() runtime.Object { return &onewer.OAuthAuthorizeToken{} },
		func() runtime.Object { return &ocurrent.OAuthAuthorizeToken{} },
		t,
	)
}

func TestAuthorizeTokenList(t *testing.T) {
	testObject(
		onewer.OAuthAuthorizeTokenListFixture(),
		func() runtime.Object { return &onewer.OAuthAuthorizeTokenList{} },
		func() runtime.Object { return &ocurrent.OAuthAuthorizeTokenList{} },
		t,
	)
}

func TestClient(t *testing.T) {
	testObject(
		onewer.OAuthClientFixture(),
		func() runtime.Object { return &onewer.OAuthClient{} },
		func() runtime.Object { return &ocurrent.OAuthClient{} },
		t,
	)
}

func TestClientList(t *testing.T) {
	testObject(
		onewer.OAuthClientListFixture(),
		func() runtime.Object { return &onewer.OAuthClientList{} },
		func() runtime.Object { return &ocurrent.OAuthClientList{} },
		t,
	)
}

func TestClientAuthorization(t *testing.T) {
	testObject(
		onewer.OAuthClientAuthorizationFixture(),
		func() runtime.Object { return &onewer.OAuthClientAuthorization{} },
		func() runtime.Object { return &ocurrent.OAuthClientAuthorization{} },
		t,
	)
}

func TestClientAuthorizationList(t *testing.T) {
	testObject(
		onewer.OAuthClientAuthorizationListFixture(),
		func() runtime.Object { return &onewer.OAuthClientAuthorizationList{} },
		func() runtime.Object { return &ocurrent.OAuthClientAuthorizationList{} },
		t,
	)
}

var Convert = newer.Scheme.Convert

type maker func() runtime.Object

func testObject(newObj runtime.Object, newObjMaker, currentObjMaker maker, t *testing.T) {
	// Make sure we can convert a newer object to a current object
	oldObj := currentObjMaker()
	err := Convert(newObj, oldObj)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Make sure we didn't lose any information
	newObj2 := newObjMaker()
	err = Convert(oldObj, newObj2)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(newObj, newObj2) {
		t.Fatalf("Round-trip failed.\nNew: \n\t%#v\nCurrent: \n\t%#v\nNew2:\n\t%#v", newObj, oldObj, newObj2)
	}

	// Make sure we can encode the old object
	data, err := current.Codec.Encode(oldObj)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Decode a current serialization into a current object
	oldObj2 := currentObjMaker()
	if err := current.Codec.DecodeInto(data, oldObj2); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(oldObj, oldObj2) {
		t.Fatalf("Encode/decode failed.\nOriginal:\n\t%#v\nCurrent JSON:\n\t%#v\nDecoded:\n\t%#v", oldObj, string(data), oldObj2)
	}

	// Make sure we can decode an old serialization into a newer object
	newObj3, err := current.Codec.Decode(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reflect.TypeOf(newObj3) != reflect.TypeOf(newObj) {
		t.Fatalf("unexpected type: %#v", newObj3)
	}
	if !reflect.DeepEqual(newObj, newObj3) {
		t.Fatalf("Encode/decode failed.\nNew:\n\t%#v\nCurrent JSON:\n\t%#v\nDecoded:\n\t%#v", newObj, string(data), newObj3)
	}
}
