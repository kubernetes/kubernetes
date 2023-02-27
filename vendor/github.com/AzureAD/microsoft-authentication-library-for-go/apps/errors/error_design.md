# MSAL Error Design

Author: Abhidnya Patil(abhidnya.patil@microsoft.com)

Contributors:

- John Doak(jdoak@microsoft.com)
- Keegan Caruso(Keegan.Caruso@microsoft.com)
- Joel Hendrix(jhendrix@microsoft.com)

## Background

Errors in MSAL are intended for app developers to troubleshoot and not for displaying to end-users.

### Go error handling vs other MSAL languages

Most modern languages use exception based errors. Simply put, you "throw" an exception and it must be caught at some routine in the upper stack or it will eventually crash the program.

Go doesn't use exceptions, instead it relies on multiple return values, one of which can be the builtin error interface type. It is up to the user to decide what to do.

### Go custom error types

Errors can be created in Go by simply using errors.New() or fmt.Errorf() to create an "error".

Custom errors can be created in multiple ways. One of the more robust ways is simply to satisfy the error interface:

```go
type MyCustomErr struct {
  Msg string
}
func (m MyCustomErr) Error() string { // This implements "error"
  return m.Msg
}
```

### MSAL Error Goals

- Provide diagnostics to the user and for tickets that can be used to track down bugs or client misconfigurations
- Detect errors that are transitory and can be retried
- Allow the user to identify certain errors that the program can respond to, such a informing the user for the need to do an enrollment
  
## Implementing Client Side Errors

Client side errors indicate a misconfiguration or passing of bad arguments that is non-recoverable. Retrying isn't possible.

These errors can simply be standard Go errors created by errors.New() or fmt.Errorf(). If down the line we need a custom error, we can introduce it, but for now the error messages just need to be clear on what the issue was.

## Implementing Service Side Errors

Service side errors occur when an external RPC responds either with an HTTP error code or returns a message that includes an error.

These errors can be transitory (please slow down) or permanent (HTTP 404).  To provide our diagnostic goals, we require the ability to differentiate these errors from other errors.

The current implementation includes a specialized type that captures any error from the server:

```go
// CallErr represents an HTTP call error. Has a Verbose() method that allows getting the
// http.Request and Response objects. Implements error.
type CallErr struct {
    Req  *http.Request
    Resp *http.Response
    Err  error
}

// Errors implements error.Error().
func (e CallErr) Error() string {
    return e.Err.Error()
}

// Verbose prints a versbose error message with the request or response.
func (e CallErr) Verbose() string {
    e.Resp.Request = nil // This brings in a bunch of TLS crap we don't need
    e.Resp.TLS = nil     // Same
    return fmt.Sprintf("%s:\nRequest:\n%s\nResponse:\n%s", e.Err, prettyConf.Sprint(e.Req), prettyConf.Sprint(e.Resp))
}
```

A user will always receive the most concise error we provide.  They can tell if it is a server side error using Go error package:

```go
var callErr CallErr
if errors.As(err, &callErr) {
  ...
}
```

We provide a Verbose() function that can retrieve the most verbose message from any error we provide:

```go
fmt.Println(errors.Verbose(err))
```

If further differentiation is required, we can add custom errors that use Go error wrapping on top of CallErr to achieve our diagnostic goals (such as detecting when to retry a call due to transient errors).  

CallErr is always thrown from the comm package (which handles all http requests) and looks similar to:

```go
return nil, errors.CallErr{
    Req:  req,
    Resp: reply,
    Err:  fmt.Errorf("http call(%s)(%s) error: reply status code was %d:\n%s", req.URL.String(), req.Method, reply.StatusCode, ErrorResponse), //ErrorResponse is the json body extracted from the http response
    }
```

## Future Decisions

The ability to retry calls needs to have centralized responsibility. Either the user is doing it or the client is doing it.  

If the user should be responsible, our errors package will include a CanRetry() function that will inform the user if the error provided to them is retryable.  This is based on the http error code and possibly the type of error that was returned.  It would also include a sleep time if the server returned an amount of time to wait.

Otherwise we will do this internally and retries will be left to us.
