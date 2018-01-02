package security

import (
	"bytes"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/go-openapi/errors"
	"github.com/stretchr/testify/assert"
)

var bearerAuth = ScopedTokenAuthentication(func(token string, requiredScopes []string) (interface{}, error) {
	if token == "token123" {
		return "admin", nil
	}
	return nil, errors.Unauthenticated("bearer")
})

func TestValidBearerAuth(t *testing.T) {
	ba := BearerAuth("owners_auth", bearerAuth)

	req1, _ := http.NewRequest("GET", "/blah?access_token=token123", nil)

	ok, usr, err := ba.Authenticate(&ScopedAuthRequest{Request: req1})
	assert.True(t, ok)
	assert.Equal(t, "admin", usr)
	assert.NoError(t, err)

	req2, _ := http.NewRequest("GET", "/blah", nil)
	req2.Header.Set("Authorization", "Bearer token123")

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req2})
	assert.True(t, ok)
	assert.Equal(t, "admin", usr)
	assert.NoError(t, err)

	body := url.Values(map[string][]string{})
	body.Set("access_token", "token123")
	req3, _ := http.NewRequest("POST", "/blah", strings.NewReader(body.Encode()))
	req3.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req3})
	assert.True(t, ok)
	assert.Equal(t, "admin", usr)
	assert.NoError(t, err)

	mpbody := bytes.NewBuffer(nil)
	writer := multipart.NewWriter(mpbody)
	writer.WriteField("access_token", "token123")
	writer.Close()
	req4, _ := http.NewRequest("POST", "/blah", mpbody)
	req4.Header.Set("Content-Type", writer.FormDataContentType())

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req4})
	assert.True(t, ok)
	assert.Equal(t, "admin", usr)
	assert.NoError(t, err)
}

func TestInvalidBearerAuth(t *testing.T) {
	ba := BearerAuth("owners_auth", bearerAuth)

	req1, _ := http.NewRequest("GET", "/blah?access_token=token124", nil)

	ok, usr, err := ba.Authenticate(&ScopedAuthRequest{Request: req1})
	assert.True(t, ok)
	assert.Equal(t, nil, usr)
	assert.Error(t, err)

	req2, _ := http.NewRequest("GET", "/blah", nil)
	req2.Header.Set("Authorization", "Bearer token124")

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req2})
	assert.True(t, ok)
	assert.Equal(t, nil, usr)
	assert.Error(t, err)

	body := url.Values(map[string][]string{})
	body.Set("access_token", "token124")
	req3, _ := http.NewRequest("POST", "/blah", strings.NewReader(body.Encode()))
	req3.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req3})
	assert.True(t, ok)
	assert.Equal(t, nil, usr)
	assert.Error(t, err)

	mpbody := bytes.NewBuffer(nil)
	writer := multipart.NewWriter(mpbody)
	writer.WriteField("access_token", "token124")
	writer.Close()
	req4, _ := http.NewRequest("POST", "/blah", mpbody)
	req4.Header.Set("Content-Type", writer.FormDataContentType())

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req4})
	assert.True(t, ok)
	assert.Equal(t, nil, usr)
	assert.Error(t, err)
}

func TestMissingBearerAuth(t *testing.T) {
	ba := BearerAuth("owners_auth", bearerAuth)

	req1, _ := http.NewRequest("GET", "/blah?access_toke=token123", nil)

	ok, usr, err := ba.Authenticate(&ScopedAuthRequest{Request: req1})
	assert.False(t, ok)
	assert.Equal(t, nil, usr)
	assert.NoError(t, err)

	req2, _ := http.NewRequest("GET", "/blah", nil)
	req2.Header.Set("Authorization", "Beare token123")

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req2})
	assert.False(t, ok)
	assert.Equal(t, nil, usr)
	assert.NoError(t, err)

	body := url.Values(map[string][]string{})
	body.Set("access_toke", "token123")
	req3, _ := http.NewRequest("POST", "/blah", strings.NewReader(body.Encode()))
	req3.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req3})
	assert.False(t, ok)
	assert.Equal(t, nil, usr)
	assert.NoError(t, err)

	mpbody := bytes.NewBuffer(nil)
	writer := multipart.NewWriter(mpbody)
	writer.WriteField("access_toke", "token123")
	writer.Close()
	req4, _ := http.NewRequest("POST", "/blah", mpbody)
	req4.Header.Set("Content-Type", writer.FormDataContentType())

	ok, usr, err = ba.Authenticate(&ScopedAuthRequest{Request: req4})
	assert.False(t, ok)
	assert.Equal(t, nil, usr)
	assert.NoError(t, err)
}
