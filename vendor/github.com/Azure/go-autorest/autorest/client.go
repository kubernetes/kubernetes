package autorest

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/cookiejar"
	"time"
)

const (
	// DefaultPollingDelay is a reasonable delay between polling requests.
	DefaultPollingDelay = 60 * time.Second

	// DefaultPollingDuration is a reasonable total polling duration.
	DefaultPollingDuration = 15 * time.Minute

	// DefaultRetryAttempts is number of attempts for retry status codes (5xx).
	DefaultRetryAttempts = 3

	// DefaultRetryDuration is a resonable delay for retry.
	defaultRetryInterval = 30 * time.Second
)

var statusCodesForRetry = []int{
	http.StatusRequestTimeout,      // 408
	http.StatusInternalServerError, // 500
	http.StatusBadGateway,          // 502
	http.StatusServiceUnavailable,  // 503
	http.StatusGatewayTimeout,      // 504
}

const (
	requestFormat = `HTTP Request Begin ===================================================
%s
===================================================== HTTP Request End
`
	responseFormat = `HTTP Response Begin ===================================================
%s
===================================================== HTTP Response End
`
)

// Response serves as the base for all responses from generated clients. It provides access to the
// last http.Response.
type Response struct {
	*http.Response `json:"-"`
}

// LoggingInspector implements request and response inspectors that log the full request and
// response to a supplied log.
type LoggingInspector struct {
	Logger *log.Logger
}

// WithInspection returns a PrepareDecorator that emits the http.Request to the supplied logger. The
// body is restored after being emitted.
//
// Note: Since it reads the entire Body, this decorator should not be used where body streaming is
// important. It is best used to trace JSON or similar body values.
func (li LoggingInspector) WithInspection() PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			var body, b bytes.Buffer

			defer r.Body.Close()

			r.Body = ioutil.NopCloser(io.TeeReader(r.Body, &body))
			if err := r.Write(&b); err != nil {
				return nil, fmt.Errorf("Failed to write response: %v", err)
			}

			li.Logger.Printf(requestFormat, b.String())

			r.Body = ioutil.NopCloser(&body)
			return p.Prepare(r)
		})
	}
}

// ByInspecting returns a RespondDecorator that emits the http.Response to the supplied logger. The
// body is restored after being emitted.
//
// Note: Since it reads the entire Body, this decorator should not be used where body streaming is
// important. It is best used to trace JSON or similar body values.
func (li LoggingInspector) ByInspecting() RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			var body, b bytes.Buffer
			defer resp.Body.Close()
			resp.Body = ioutil.NopCloser(io.TeeReader(resp.Body, &body))
			if err := resp.Write(&b); err != nil {
				return fmt.Errorf("Failed to write response: %v", err)
			}

			li.Logger.Printf(responseFormat, b.String())

			resp.Body = ioutil.NopCloser(&body)
			return r.Respond(resp)
		})
	}
}

// Client is the base for autorest generated clients. It provides default, "do nothing"
// implementations of an Authorizer, RequestInspector, and ResponseInspector. It also returns the
// standard, undecorated http.Client as a default Sender.
//
// Generated clients should also use Error (see NewError and NewErrorWithError) for errors and
// return responses that compose with Response.
//
// Most customization of generated clients is best achieved by supplying a custom Authorizer, custom
// RequestInspector, and / or custom ResponseInspector. Users may log requests, implement circuit
// breakers (see https://msdn.microsoft.com/en-us/library/dn589784.aspx) or otherwise influence
// sending the request by providing a decorated Sender.
type Client struct {
	Authorizer        Authorizer
	Sender            Sender
	RequestInspector  PrepareDecorator
	ResponseInspector RespondDecorator

	// PollingDelay sets the polling frequency used in absence of a Retry-After HTTP header
	PollingDelay time.Duration

	// PollingDuration sets the maximum polling time after which an error is returned.
	PollingDuration time.Duration

	// RetryAttempts sets the default number of retry attempts for client.
	RetryAttempts int

	// UserAgent, if not empty, will be set as the HTTP User-Agent header on all requests sent
	// through the Do method.
	UserAgent string

	Jar http.CookieJar
}

// NewClientWithUserAgent returns an instance of a Client with the UserAgent set to the passed
// string.
func NewClientWithUserAgent(ua string) Client {
	return Client{
		PollingDelay:    DefaultPollingDelay,
		PollingDuration: DefaultPollingDuration,
		RetryAttempts:   DefaultRetryAttempts,
		UserAgent:       ua,
	}
}

// Do implements the Sender interface by invoking the active Sender after applying authorization.
// If Sender is not set, it uses a new instance of http.Client. In both cases it will, if UserAgent
// is set, apply set the User-Agent header.
func (c Client) Do(r *http.Request) (*http.Response, error) {
	if r.UserAgent() == "" {
		r, _ = Prepare(r,
			WithUserAgent(c.UserAgent))
	}
	r, err := Prepare(r,
		c.WithInspection(),
		c.WithAuthorization())
	if err != nil {
		return nil, NewErrorWithError(err, "autorest/Client", "Do", nil, "Preparing request failed")
	}
	resp, err := SendWithSender(c.sender(), r,
		DoRetryForStatusCodes(c.RetryAttempts, defaultRetryInterval, statusCodesForRetry...))
	Respond(resp,
		c.ByInspecting())
	return resp, err
}

// sender returns the Sender to which to send requests.
func (c Client) sender() Sender {
	if c.Sender == nil {
		j, _ := cookiejar.New(nil)
		return &http.Client{Jar: j}
	}
	return c.Sender
}

// WithAuthorization is a convenience method that returns the WithAuthorization PrepareDecorator
// from the current Authorizer. If not Authorizer is set, it uses the NullAuthorizer.
func (c Client) WithAuthorization() PrepareDecorator {
	return c.authorizer().WithAuthorization()
}

// authorizer returns the Authorizer to use.
func (c Client) authorizer() Authorizer {
	if c.Authorizer == nil {
		return NullAuthorizer{}
	}
	return c.Authorizer
}

// WithInspection is a convenience method that passes the request to the supplied RequestInspector,
// if present, or returns the WithNothing PrepareDecorator otherwise.
func (c Client) WithInspection() PrepareDecorator {
	if c.RequestInspector == nil {
		return WithNothing()
	}
	return c.RequestInspector
}

// ByInspecting is a convenience method that passes the response to the supplied ResponseInspector,
// if present, or returns the ByIgnoring RespondDecorator otherwise.
func (c Client) ByInspecting() RespondDecorator {
	if c.ResponseInspector == nil {
		return ByIgnoring()
	}
	return c.ResponseInspector
}
