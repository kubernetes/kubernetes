package credentials

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func initTestServer(expireOn string) *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.RequestURI == "/" {
			fmt.Fprintln(w, "/creds")
		} else {
			fmt.Fprintf(w, `{
  "AccessKeyId" : "accessKey",
  "SecretAccessKey" : "secret",
  "Token" : "token",
  "Expiration" : "%s"
}`, expireOn)
		}
	}))

	return server
}

func TestEC2RoleProvider(t *testing.T) {
	server := initTestServer("2014-12-16T01:51:37Z")
	defer server.Close()

	p := &EC2RoleProvider{Client: http.DefaultClient, Endpoint: server.URL}

	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "accessKey", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "secret", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Equal(t, "token", creds.SessionToken, "Expect session token to match")
}

func TestEC2RoleProviderIsExpired(t *testing.T) {
	server := initTestServer("2014-12-16T01:51:37Z")
	defer server.Close()

	p := &EC2RoleProvider{Client: http.DefaultClient, Endpoint: server.URL}
	defer func() {
		currentTime = time.Now
	}()
	currentTime = func() time.Time {
		return time.Date(2014, 12, 15, 21, 26, 0, 0, time.UTC)
	}

	assert.True(t, p.IsExpired(), "Expect creds to be expired before retrieve.")

	_, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.False(t, p.IsExpired(), "Expect creds to not be expired after retrieve.")

	currentTime = func() time.Time {
		return time.Date(3014, 12, 15, 21, 26, 0, 0, time.UTC)
	}

	assert.True(t, p.IsExpired(), "Expect creds to be expired.")
}

func TestEC2RoleProviderExpiryWindowIsExpired(t *testing.T) {
	server := initTestServer("2014-12-16T01:51:37Z")
	defer server.Close()

	p := &EC2RoleProvider{Client: http.DefaultClient, Endpoint: server.URL, ExpiryWindow: time.Hour * 1}
	defer func() {
		currentTime = time.Now
	}()
	currentTime = func() time.Time {
		return time.Date(2014, 12, 15, 0, 51, 37, 0, time.UTC)
	}

	assert.True(t, p.IsExpired(), "Expect creds to be expired before retrieve.")

	_, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.False(t, p.IsExpired(), "Expect creds to not be expired after retrieve.")

	currentTime = func() time.Time {
		return time.Date(2014, 12, 16, 0, 55, 37, 0, time.UTC)
	}

	assert.True(t, p.IsExpired(), "Expect creds to be expired.")
}

func BenchmarkEC2RoleProvider(b *testing.B) {
	server := initTestServer("2014-12-16T01:51:37Z")
	defer server.Close()

	p := &EC2RoleProvider{Client: http.DefaultClient, Endpoint: server.URL}
	_, err := p.Retrieve()
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := p.Retrieve()
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
