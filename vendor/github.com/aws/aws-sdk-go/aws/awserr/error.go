// Package awserr represents API error interface accessors for the SDK.
package awserr

// An Error wraps lower level errors with code, message and an original error.
// The underlying concrete error type may also satisfy other interfaces which
// can be to used to obtain more specific information about the error.
//
// Calling Error() or String() will always include the full information about
// an error based on its underlying type.
//
// Example:
//
//     output, err := s3manage.Upload(svc, input, opts)
//     if err != nil {
//         if awsErr, ok := err.(awserr.Error); ok {
//             // Get error details
//             log.Println("Error:", err.Code(), err.Message())
//
//             // Prints out full error message, including original error if there was one.
//             log.Println("Error:", err.Error())
//
//             // Get original error
//             if origErr := err.Err(); origErr != nil {
//                 // operate on original error.
//             }
//         } else {
//             fmt.Println(err.Error())
//         }
//     }
//
type Error interface {
	// Satisfy the generic error interface.
	error

	// Returns the short phrase depicting the classification of the error.
	Code() string

	// Returns the error details message.
	Message() string

	// Returns the original error if one was set.  Nil is returned if not set.
	OrigErr() error
}

// New returns an Error object described by the code, message, and origErr.
//
// If origErr satisfies the Error interface it will not be wrapped within a new
// Error object and will instead be returned.
func New(code, message string, origErr error) Error {
	if e, ok := origErr.(Error); ok && e != nil {
		return e
	}
	return newBaseError(code, message, origErr)
}

// A RequestFailure is an interface to extract request failure information from
// an Error such as the request ID of the failed request returned by a service.
// RequestFailures may not always have a requestID value if the request failed
// prior to reaching the service such as a connection error.
//
// Example:
//
//     output, err := s3manage.Upload(svc, input, opts)
//     if err != nil {
//         if reqerr, ok := err.(RequestFailure); ok {
//             log.Printf("Request failed", reqerr.Code(), reqerr.Message(), reqerr.RequestID())
//         } else {
//             log.Printf("Error:", err.Error()
//         }
//     }
//
// Combined with awserr.Error:
//
//    output, err := s3manage.Upload(svc, input, opts)
//    if err != nil {
//        if awsErr, ok := err.(awserr.Error); ok {
//            // Generic AWS Error with Code, Message, and original error (if any)
//            fmt.Println(awsErr.Code(), awsErr.Message(), awsErr.OrigErr())
//
//            if reqErr, ok := err.(awserr.RequestFailure); ok {
//                // A service error occurred
//                fmt.Println(reqErr.StatusCode(), reqErr.RequestID())
//            }
//        } else {
//            fmt.Println(err.Error())
//        }
//    }
//
type RequestFailure interface {
	Error

	// The status code of the HTTP response.
	StatusCode() int

	// The request ID returned by the service for a request failure. This will
	// be empty if no request ID is available such as the request failed due
	// to a connection error.
	RequestID() string
}

// NewRequestFailure returns a new request error wrapper for the given Error
// provided.
func NewRequestFailure(err Error, statusCode int, reqID string) RequestFailure {
	return newRequestError(err, statusCode, reqID)
}
