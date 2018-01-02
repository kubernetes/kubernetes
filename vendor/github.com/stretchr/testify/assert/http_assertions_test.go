package assert

import (
	"fmt"
	"net/http"
	"net/url"
	"testing"
)

func httpOK(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func httpRedirect(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusTemporaryRedirect)
}

func httpError(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusInternalServerError)
}

func TestHTTPSuccess(t *testing.T) {
	assert := New(t)

	mockT1 := new(testing.T)
	assert.Equal(HTTPSuccess(mockT1, httpOK, "GET", "/", nil), true)
	assert.False(mockT1.Failed())

	mockT2 := new(testing.T)
	assert.Equal(HTTPSuccess(mockT2, httpRedirect, "GET", "/", nil), false)
	assert.True(mockT2.Failed())

	mockT3 := new(testing.T)
	assert.Equal(HTTPSuccess(mockT3, httpError, "GET", "/", nil), false)
	assert.True(mockT3.Failed())
}

func TestHTTPRedirect(t *testing.T) {
	assert := New(t)

	mockT1 := new(testing.T)
	assert.Equal(HTTPRedirect(mockT1, httpOK, "GET", "/", nil), false)
	assert.True(mockT1.Failed())

	mockT2 := new(testing.T)
	assert.Equal(HTTPRedirect(mockT2, httpRedirect, "GET", "/", nil), true)
	assert.False(mockT2.Failed())

	mockT3 := new(testing.T)
	assert.Equal(HTTPRedirect(mockT3, httpError, "GET", "/", nil), false)
	assert.True(mockT3.Failed())
}

func TestHTTPError(t *testing.T) {
	assert := New(t)

	mockT1 := new(testing.T)
	assert.Equal(HTTPError(mockT1, httpOK, "GET", "/", nil), false)
	assert.True(mockT1.Failed())

	mockT2 := new(testing.T)
	assert.Equal(HTTPError(mockT2, httpRedirect, "GET", "/", nil), false)
	assert.True(mockT2.Failed())

	mockT3 := new(testing.T)
	assert.Equal(HTTPError(mockT3, httpError, "GET", "/", nil), true)
	assert.False(mockT3.Failed())
}

func TestHTTPStatusesWrapper(t *testing.T) {
	assert := New(t)
	mockAssert := New(new(testing.T))

	assert.Equal(mockAssert.HTTPSuccess(httpOK, "GET", "/", nil), true)
	assert.Equal(mockAssert.HTTPSuccess(httpRedirect, "GET", "/", nil), false)
	assert.Equal(mockAssert.HTTPSuccess(httpError, "GET", "/", nil), false)

	assert.Equal(mockAssert.HTTPRedirect(httpOK, "GET", "/", nil), false)
	assert.Equal(mockAssert.HTTPRedirect(httpRedirect, "GET", "/", nil), true)
	assert.Equal(mockAssert.HTTPRedirect(httpError, "GET", "/", nil), false)

	assert.Equal(mockAssert.HTTPError(httpOK, "GET", "/", nil), false)
	assert.Equal(mockAssert.HTTPError(httpRedirect, "GET", "/", nil), false)
	assert.Equal(mockAssert.HTTPError(httpError, "GET", "/", nil), true)
}

func httpHelloName(w http.ResponseWriter, r *http.Request) {
	name := r.FormValue("name")
	w.Write([]byte(fmt.Sprintf("Hello, %s!", name)))
}

func TestHttpBody(t *testing.T) {
	assert := New(t)
	mockT := new(testing.T)

	assert.True(HTTPBodyContains(mockT, httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "Hello, World!"))
	assert.True(HTTPBodyContains(mockT, httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "World"))
	assert.False(HTTPBodyContains(mockT, httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "world"))

	assert.False(HTTPBodyNotContains(mockT, httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "Hello, World!"))
	assert.False(HTTPBodyNotContains(mockT, httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "World"))
	assert.True(HTTPBodyNotContains(mockT, httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "world"))
}

func TestHttpBodyWrappers(t *testing.T) {
	assert := New(t)
	mockAssert := New(new(testing.T))

	assert.True(mockAssert.HTTPBodyContains(httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "Hello, World!"))
	assert.True(mockAssert.HTTPBodyContains(httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "World"))
	assert.False(mockAssert.HTTPBodyContains(httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "world"))

	assert.False(mockAssert.HTTPBodyNotContains(httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "Hello, World!"))
	assert.False(mockAssert.HTTPBodyNotContains(httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "World"))
	assert.True(mockAssert.HTTPBodyNotContains(httpHelloName, "GET", "/", url.Values{"name": []string{"World"}}, "world"))

}
