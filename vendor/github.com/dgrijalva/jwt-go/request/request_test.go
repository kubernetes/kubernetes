package request

import (
	"fmt"
	"github.com/dgrijalva/jwt-go"
	"github.com/dgrijalva/jwt-go/test"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
)

var requestTestData = []struct {
	name      string
	claims    jwt.MapClaims
	extractor Extractor
	headers   map[string]string
	query     url.Values
	valid     bool
}{
	{
		"authorization bearer token",
		jwt.MapClaims{"foo": "bar"},
		AuthorizationHeaderExtractor,
		map[string]string{"Authorization": "Bearer %v"},
		url.Values{},
		true,
	},
	{
		"oauth bearer token - header",
		jwt.MapClaims{"foo": "bar"},
		OAuth2Extractor,
		map[string]string{"Authorization": "Bearer %v"},
		url.Values{},
		true,
	},
	{
		"oauth bearer token - url",
		jwt.MapClaims{"foo": "bar"},
		OAuth2Extractor,
		map[string]string{},
		url.Values{"access_token": {"%v"}},
		true,
	},
	{
		"url token",
		jwt.MapClaims{"foo": "bar"},
		ArgumentExtractor{"token"},
		map[string]string{},
		url.Values{"token": {"%v"}},
		true,
	},
}

func TestParseRequest(t *testing.T) {
	// load keys from disk
	privateKey := test.LoadRSAPrivateKeyFromDisk("../test/sample_key")
	publicKey := test.LoadRSAPublicKeyFromDisk("../test/sample_key.pub")
	keyfunc := func(*jwt.Token) (interface{}, error) {
		return publicKey, nil
	}

	// Bearer token request
	for _, data := range requestTestData {
		// Make token from claims
		tokenString := test.MakeSampleToken(data.claims, privateKey)

		// Make query string
		for k, vv := range data.query {
			for i, v := range vv {
				if strings.Contains(v, "%v") {
					data.query[k][i] = fmt.Sprintf(v, tokenString)
				}
			}
		}

		// Make request from test struct
		r, _ := http.NewRequest("GET", fmt.Sprintf("/?%v", data.query.Encode()), nil)
		for k, v := range data.headers {
			if strings.Contains(v, "%v") {
				r.Header.Set(k, fmt.Sprintf(v, tokenString))
			} else {
				r.Header.Set(k, tokenString)
			}
		}
		token, err := ParseFromRequestWithClaims(r, data.extractor, jwt.MapClaims{}, keyfunc)

		if token == nil {
			t.Errorf("[%v] Token was not found: %v", data.name, err)
			continue
		}
		if !reflect.DeepEqual(data.claims, token.Claims) {
			t.Errorf("[%v] Claims mismatch. Expecting: %v  Got: %v", data.name, data.claims, token.Claims)
		}
		if data.valid && err != nil {
			t.Errorf("[%v] Error while verifying token: %v", data.name, err)
		}
		if !data.valid && err == nil {
			t.Errorf("[%v] Invalid token passed validation", data.name)
		}
	}
}
