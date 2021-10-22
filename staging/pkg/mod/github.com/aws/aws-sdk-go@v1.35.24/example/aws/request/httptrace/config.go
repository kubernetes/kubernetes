package main

import (
	"flag"
	"net/http"
	"time"
)

// ClientConfig provides the timeouts from CLI flags and default values for the
// example apps's configuration.
type ClientConfig struct {
	KeepAlive bool
	Timeouts  Timeouts
}

// SetupFlags initializes the CLI flags.
func (c *ClientConfig) SetupFlags(prefix string, flagset *flag.FlagSet) {
	prefix += "client."

	flagset.BoolVar(&c.KeepAlive, prefix+"http-keep-alive", true,
		"Specifies if HTTP keep alive is enabled.")

	c.Timeouts.SetupFlags(prefix, flagset)
}

// Timeouts collection of HTTP client timeout values.
type Timeouts struct {
	IdleConnection time.Duration
	Connect        time.Duration
	TLSHandshake   time.Duration
	ExpectContinue time.Duration
	ResponseHeader time.Duration
}

// SetupFlags initializes the CLI flags.
func (c *Timeouts) SetupFlags(prefix string, flagset *flag.FlagSet) {
	prefix += "timeout."

	flagset.DurationVar(&c.IdleConnection, prefix+"idle-conn", 90*time.Second,
		"The `timeout` of idle connects to the remote host.")

	flagset.DurationVar(&c.Connect, prefix+"connect", 30*time.Second,
		"The `timeout` connecting to the remote host.")

	defTR := http.DefaultTransport.(*http.Transport)

	flagset.DurationVar(&c.TLSHandshake, prefix+"tls", defTR.TLSHandshakeTimeout,
		"The `timeout` waiting for the TLS handshake to complete.")

	c.ExpectContinue = defTR.ExpectContinueTimeout

	flagset.DurationVar(&c.ResponseHeader, prefix+"response-header", defTR.ResponseHeaderTimeout,
		"The `timeout` waiting for the TLS handshake to complete.")
}
