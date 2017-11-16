# rootcerts

Functions for loading root certificates for TLS connections.

-----

Go's standard library `crypto/tls` provides a common mechanism for configuring
TLS connections in `tls.Config`. The `RootCAs` field on this struct is a pool
of certificates for the client to use as a trust store when verifying server
certificates.

This library contains utility functions for loading certificates destined for
that field, as well as one other important thing:

When the `RootCAs` field is `nil`, the standard library attempts to load the
host's root CA set.  This behavior is OS-specific, and the Darwin
implementation contains [a bug that prevents trusted certificates from the
System and Login keychains from being loaded][1]. This library contains
Darwin-specific behavior that works around that bug.

[1]: https://github.com/golang/go/issues/14514

## Example Usage

Here's a snippet demonstrating how this library is meant to be used:

```go
func httpClient() (*http.Client, error)
	tlsConfig := &tls.Config{}
	err := rootcerts.ConfigureTLS(tlsConfig, &rootcerts.Config{
		CAFile: os.Getenv("MYAPP_CAFILE"),
		CAPath: os.Getenv("MYAPP_CAPATH"),
	})
	if err != nil {
		return nil, err
	}
	c := cleanhttp.DefaultClient()
	t := cleanhttp.DefaultTransport()
	t.TLSClientConfig = tlsConfig
	c.Transport = t
	return c, nil
}
```
