package openshiftkubeapiserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/util/flowcontrol"
)

func TestWatchRateLimit(t *testing.T) {
	delegate := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	fakeClock := clock.NewFakeClock(time.Now())

	watchRateLimiter := newWatchRateLimit(delegate, fakeClock)
	watchRateLimiter.earlyRateLimiter = flowcontrol.NewTokenBucketRateLimiterWithClock(1, 1, fakeClock)
	watchRateLimiter.middleRateLimiter = flowcontrol.NewTokenBucketRateLimiterWithClock(3, 3, fakeClock)
	watchRateLimiter.authorizerAttributesFn = func(ctx context.Context) (authorizer.Attributes, error) {
		return authorizer.AttributesRecord{
			User: &user.DefaultInfo{
				Name:   "",
				UID:    "",
				Groups: nil,
				Extra:  nil,
			},
			Verb:            "watch",
			Namespace:       "foo",
			APIGroup:        "",
			APIVersion:      "",
			Resource:        "pods",
			Subresource:     "",
			Name:            "",
			ResourceRequest: true,
			Path:            "",
		}, nil
	}

	testServer := httptest.NewServer(watchRateLimiter)
	defer testServer.Close()

	// fill the buckets
	fakeClock.SetTime(fakeClock.Now().Add(1 * time.Minute))
	expectOk(t, testServer)
	expectRateLimit(t, testServer)

	// refill early
	fakeClock.SetTime(fakeClock.Now().Add(1 * time.Second))
	expectOk(t, testServer)
	expectRateLimit(t, testServer)

	// move to the middle rate limiter
	fakeClock.SetTime(fakeClock.Now().Add(10 * time.Minute))
	expectOk(t, testServer)
	expectOk(t, testServer)
	expectOk(t, testServer)
	expectRateLimit(t, testServer)

	// refill middle
	fakeClock.SetTime(fakeClock.Now().Add(1 * time.Second))
	expectOk(t, testServer)
	expectOk(t, testServer)
	expectOk(t, testServer)
	expectRateLimit(t, testServer)

	// move to unlimited
	fakeClock.SetTime(fakeClock.Now().Add(10 * time.Minute))
	expectOk(t, testServer)
	expectOk(t, testServer)
	expectOk(t, testServer)
	expectOk(t, testServer)
	expectOk(t, testServer)
}

func expectOk(t *testing.T, testServer *httptest.Server) {
	t.Helper()

	response, err := testServer.Client().Get(testServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatal(response.StatusCode)
	}
}

func expectRateLimit(t *testing.T, testServer *httptest.Server) {
	t.Helper()

	response, err := testServer.Client().Get(testServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusTooManyRequests {
		t.Fatal(response.StatusCode)
	}
}
