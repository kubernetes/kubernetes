package aws

import (
	"math"
	"net"
	"net/http"
	"time"
)

type RetryableFunc func(*http.Request, *http.Response, error) bool
type WaitFunc func(try int)
type DeadlineFunc func() time.Time

type ResilientTransport struct {
	// Timeout is the maximum amount of time a dial will wait for
	// a connect to complete.
	//
	// The default is no timeout.
	//
	// With or without a timeout, the operating system may impose
	// its own earlier timeout. For instance, TCP timeouts are
	// often around 3 minutes.
	DialTimeout time.Duration

	// MaxTries, if non-zero, specifies the number of times we will retry on
	// failure. Retries are only attempted for temporary network errors or known
	// safe failures.
	MaxTries    int
	Deadline    DeadlineFunc
	ShouldRetry RetryableFunc
	Wait        WaitFunc
	transport   *http.Transport
}

// Convenience method for creating an http client
func NewClient(rt *ResilientTransport) *http.Client {
	rt.transport = &http.Transport{
		Dial: func(netw, addr string) (net.Conn, error) {
			c, err := net.DialTimeout(netw, addr, rt.DialTimeout)
			if err != nil {
				return nil, err
			}
			c.SetDeadline(rt.Deadline())
			return c, nil
		},
		DisableKeepAlives: true,
		Proxy:             http.ProxyFromEnvironment,
	}
	// TODO: Would be nice is ResilientTransport allowed clients to initialize
	// with http.Transport attributes.
	return &http.Client{
		Transport: rt,
	}
}

var retryingTransport = &ResilientTransport{
	Deadline: func() time.Time {
		return time.Now().Add(5 * time.Second)
	},
	DialTimeout: 10 * time.Second,
	MaxTries:    3,
	ShouldRetry: awsRetry,
	Wait:        ExpBackoff,
}

// Exported default client
var RetryingClient = NewClient(retryingTransport)

func (t *ResilientTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.tries(req)
}

// Retry a request a maximum of t.MaxTries times.
// We'll only retry if the proper criteria are met.
// If a wait function is specified, wait that amount of time
// In between requests.
func (t *ResilientTransport) tries(req *http.Request) (res *http.Response, err error) {
	for try := 0; try < t.MaxTries; try += 1 {
		res, err = t.transport.RoundTrip(req)

		if !t.ShouldRetry(req, res, err) {
			break
		}
		if res != nil {
			res.Body.Close()
		}
		if t.Wait != nil {
			t.Wait(try)
		}
	}

	return
}

func ExpBackoff(try int) {
	time.Sleep(100 * time.Millisecond *
		time.Duration(math.Exp2(float64(try))))
}

func LinearBackoff(try int) {
	time.Sleep(time.Duration(try*100) * time.Millisecond)
}

// Decide if we should retry a request.
// In general, the criteria for retrying a request is described here
// http://docs.aws.amazon.com/general/latest/gr/api-retries.html
func awsRetry(req *http.Request, res *http.Response, err error) bool {
	retry := false

	// Retry if there's a temporary network error.
	if neterr, ok := err.(net.Error); ok {
		if neterr.Temporary() {
			retry = true
		}
	}

	// Retry if we get a 5xx series error.
	if res != nil {
		if res.StatusCode >= 500 && res.StatusCode < 600 {
			retry = true
		}
	}

	return retry
}
