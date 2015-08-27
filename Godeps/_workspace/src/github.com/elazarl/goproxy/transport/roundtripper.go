package transport
import "net/http"
type RoundTripper interface {
	// RoundTrip executes a single HTTP transaction, returning
	// the Response for the request req.  RoundTrip should not
	// attempt to interpret the response.  In particular,
	// RoundTrip must return err == nil if it obtained a response,
	// regardless of the response's HTTP status code.  A non-nil
	// err should be reserved for failure to obtain a response.
	// Similarly, RoundTrip should not attempt to handle
	// higher-level protocol details such as redirects,
	// authentication, or cookies.
	//
	// RoundTrip should not modify the request, except for
	// consuming the Body.  The request's URL and Header fields
	// are guaranteed to be initialized.
	RoundTrip(*http.Request) (*http.Response, error)
	DetailedRoundTrip(*http.Request) (*RoundTripDetails, *http.Response, error)
}
