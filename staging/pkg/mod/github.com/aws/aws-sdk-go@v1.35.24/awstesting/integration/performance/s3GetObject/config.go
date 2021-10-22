// +build integration,perftest

package main

import (
	"flag"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type Config struct {
	RequestDuration time.Duration
	RequestCount    int64
	RequestDelay    time.Duration

	Endpoint    string
	Bucket, Key string

	SDK    SDKConfig
	Client ClientConfig
}

func (c *Config) SetupFlags(prefix string, flagset *flag.FlagSet) {
	flagset.DurationVar(&c.RequestDuration, "duration", 0,
		"The duration to make requests for. Use instead of count for specific running duration.")
	flagset.Int64Var(&c.RequestCount, "count", 0,
		"The total `number` of requests to make. Use instead of duration for specific count.")
	flagset.DurationVar(&c.RequestDelay, "delay", 0,
		"The detail between sequential requests.")
	flagset.StringVar(&c.Endpoint, prefix+"endpoint", "",
		"Optional overridden endpoint S3 client will connect to.")
	flagset.StringVar(&c.Bucket, "bucket", "",
		"The S3 bucket `name` to request the object from.")
	flagset.StringVar(&c.Key, "key", "",
		"The S3 object key `name` to request the object from.")

	c.SDK.SetupFlags(prefix, flagset)
	c.Client.SetupFlags(prefix, flagset)
}

func (c *Config) Validate() error {
	var errs Errors

	if c.RequestDuration != 0 && c.RequestCount != 0 {
		errs = append(errs, fmt.Errorf("duration and count canot be used together"))
	}
	if c.RequestDuration == 0 && c.RequestCount == 0 {
		errs = append(errs, fmt.Errorf("duration or count must be provided"))
	}
	if len(c.Bucket) == 0 || len(c.Key) == 0 {
		errs = append(errs, fmt.Errorf("bucket and key are required"))
	}

	if err := c.SDK.Validate(); err != nil {
		errs = append(errs, err)
	}
	if err := c.Client.Validate(); err != nil {
		errs = append(errs, err)
	}

	if len(errs) != 0 {
		return errs
	}

	return nil
}

type SDKConfig struct {
	Anonymous      bool
	ExpectContinue bool
}

func (c *SDKConfig) SetupFlags(prefix string, flagset *flag.FlagSet) {
	prefix += "sdk."

	flagset.BoolVar(&c.Anonymous, prefix+"anonymous", false,
		"Specifies if the SDK will make requests anonymously, and unsigned.")

	c.ExpectContinue = true
	//	flagset.BoolVar(&c.ExpectContinue, prefix+"100-continue", true,
	//		"Specifies if the SDK requests will wait for the 100 continue response before sending request payload.")
}

func (c *SDKConfig) Validate() error {
	return nil
}

type ClientConfig struct {
	KeepAlive bool
	Timeouts  Timeouts
}

func (c *ClientConfig) SetupFlags(prefix string, flagset *flag.FlagSet) {
	prefix += "client."

	flagset.BoolVar(&c.KeepAlive, prefix+"http-keep-alive", true,
		"Specifies if HTTP keep alive is enabled.")

	c.Timeouts.SetupFlags(prefix, flagset)
}

func (c *ClientConfig) Validate() error {
	var errs Errors

	if err := c.Timeouts.Validate(); err != nil {
		errs = append(errs, err)
	}

	if len(errs) != 0 {
		return errs
	}
	return nil
}

type Timeouts struct {
	Connect        time.Duration
	TLSHandshake   time.Duration
	ExpectContinue time.Duration
	ResponseHeader time.Duration
}

func (c *Timeouts) SetupFlags(prefix string, flagset *flag.FlagSet) {
	prefix += "timeout."

	flagset.DurationVar(&c.Connect, prefix+"connect", 30*time.Second,
		"The `timeout` connecting to the remote host.")

	defTR := http.DefaultTransport.(*http.Transport)

	flagset.DurationVar(&c.TLSHandshake, prefix+"tls", defTR.TLSHandshakeTimeout,
		"The `timeout` waiting for the TLS handshake to complete.")

	c.ExpectContinue = defTR.ExpectContinueTimeout
	//	flagset.DurationVar(&c.ExpectContinue, prefix+"expect-continue", defTR.ExpectContinueTimeout,
	//		"The `timeout` waiting for the TLS handshake to complete.")

	flagset.DurationVar(&c.ResponseHeader, prefix+"response-header", defTR.ResponseHeaderTimeout,
		"The `timeout` waiting for the TLS handshake to complete.")
}

func (c *Timeouts) Validate() error {
	return nil
}

type Errors []error

func (es Errors) Error() string {
	var buf strings.Builder
	for _, e := range es {
		buf.WriteString(e.Error())
	}

	return buf.String()
}
