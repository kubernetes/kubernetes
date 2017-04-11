// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

import (
	"testing"
)

func TestSerialization_AuthSerialization(t *testing.T) {
	assertSerializeJSON(t, BasicAuth(), `{"type":"basic"}`)

	assertSerializeJSON(t, APIKeyAuth("api-key", "header"), `{"type":"apiKey","name":"api-key","in":"header"}`)

	assertSerializeJSON(
		t,
		OAuth2Implicit("http://foo.com/authorization"),
		`{"type":"oauth2","flow":"implicit","authorizationUrl":"http://foo.com/authorization"}`)

	assertSerializeJSON(
		t,
		OAuth2Password("http://foo.com/token"),
		`{"type":"oauth2","flow":"password","tokenUrl":"http://foo.com/token"}`)

	assertSerializeJSON(t,
		OAuth2Application("http://foo.com/token"),
		`{"type":"oauth2","flow":"application","tokenUrl":"http://foo.com/token"}`)

	assertSerializeJSON(
		t,
		OAuth2AccessToken("http://foo.com/authorization", "http://foo.com/token"),
		`{"type":"oauth2","flow":"accessCode","authorizationUrl":"http://foo.com/authorization","tokenUrl":"http://foo.com/token"}`)

	auth1 := OAuth2Implicit("http://foo.com/authorization")
	auth1.AddScope("email", "read your email")
	assertSerializeJSON(
		t,
		auth1,
		`{"type":"oauth2","flow":"implicit","authorizationUrl":"http://foo.com/authorization","scopes":{"email":"read your email"}}`)

	auth2 := OAuth2Password("http://foo.com/authorization")
	auth2.AddScope("email", "read your email")
	assertSerializeJSON(
		t,
		auth2,
		`{"type":"oauth2","flow":"password","tokenUrl":"http://foo.com/authorization","scopes":{"email":"read your email"}}`)

	auth3 := OAuth2Application("http://foo.com/token")
	auth3.AddScope("email", "read your email")
	assertSerializeJSON(
		t,
		auth3,
		`{"type":"oauth2","flow":"application","tokenUrl":"http://foo.com/token","scopes":{"email":"read your email"}}`)

	auth4 := OAuth2AccessToken("http://foo.com/authorization", "http://foo.com/token")
	auth4.AddScope("email", "read your email")
	assertSerializeJSON(
		t,
		auth4,
		`{"type":"oauth2","flow":"accessCode","authorizationUrl":"http://foo.com/authorization","tokenUrl":"http://foo.com/token","scopes":{"email":"read your email"}}`)
}

func TestSerialization_AuthDeserialization(t *testing.T) {

	assertParsesJSON(t, `{"type":"basic"}`, BasicAuth())

	assertParsesJSON(
		t,
		`{"in":"header","name":"api-key","type":"apiKey"}`,
		APIKeyAuth("api-key", "header"))

	assertParsesJSON(
		t,
		`{"authorizationUrl":"http://foo.com/authorization","flow":"implicit","type":"oauth2"}`,
		OAuth2Implicit("http://foo.com/authorization"))

	assertParsesJSON(
		t,
		`{"flow":"password","tokenUrl":"http://foo.com/token","type":"oauth2"}`,
		OAuth2Password("http://foo.com/token"))

	assertParsesJSON(
		t,
		`{"flow":"application","tokenUrl":"http://foo.com/token","type":"oauth2"}`,
		OAuth2Application("http://foo.com/token"))

	assertParsesJSON(
		t,
		`{"authorizationUrl":"http://foo.com/authorization","flow":"accessCode","tokenUrl":"http://foo.com/token","type":"oauth2"}`,
		OAuth2AccessToken("http://foo.com/authorization", "http://foo.com/token"))

	auth1 := OAuth2Implicit("http://foo.com/authorization")
	auth1.AddScope("email", "read your email")
	assertParsesJSON(t,
		`{"authorizationUrl":"http://foo.com/authorization","flow":"implicit","scopes":{"email":"read your email"},"type":"oauth2"}`,
		auth1)

	auth2 := OAuth2Password("http://foo.com/token")
	auth2.AddScope("email", "read your email")
	assertParsesJSON(t,
		`{"flow":"password","scopes":{"email":"read your email"},"tokenUrl":"http://foo.com/token","type":"oauth2"}`,
		auth2)

	auth3 := OAuth2Application("http://foo.com/token")
	auth3.AddScope("email", "read your email")
	assertParsesJSON(t,
		`{"flow":"application","scopes":{"email":"read your email"},"tokenUrl":"http://foo.com/token","type":"oauth2"}`,
		auth3)

	auth4 := OAuth2AccessToken("http://foo.com/authorization", "http://foo.com/token")
	auth4.AddScope("email", "read your email")
	assertParsesJSON(
		t,
		`{"authorizationUrl":"http://foo.com/authorization","flow":"accessCode","scopes":{"email":"read your email"},"tokenUrl":"http://foo.com/token","type":"oauth2"}`,
		auth4)

}
