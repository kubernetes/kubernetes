// +build go1.13,integration,perftest

package main

import (
	"flag"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

type Config struct {
	Bucket         string
	Size           int64
	Key            string
	LogVerbose     bool
	UploadPartSize int64

	SDK      SDKConfig
	Client   ClientConfig
	Profiler Profiler
}

func (c *Config) SetupFlags(prefix string, flagset *flag.FlagSet) {
	flagset.StringVar(&c.Bucket, "bucket", "",
		"The S3 bucket `name` to download the object from.")
	flagset.Int64Var(&c.Size, "size", 0,
		"The S3 object size in bytes to be first uploaded then downloaded")
	flagset.StringVar(&c.Key, "key", "", "The S3 object key to download")
	flagset.BoolVar(&c.LogVerbose, "verbose", false,
		"The output log will include verbose request information")
	flagset.Int64Var(&c.UploadPartSize, "upload-part-size", 0, "the upload part size when uploading a file to s3")

	c.SDK.SetupFlags(prefix, flagset)
	c.Client.SetupFlags(prefix, flagset)
	c.Profiler.SetupFlags(prefix, flagset)
}

func (c *Config) Validate() error {
	var errs Errors

	if len(c.Bucket) == 0 || (c.Size <= 0 && len(c.Key) == 0) {
		errs = append(errs, fmt.Errorf("bucket and filename/size are required"))
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
	PartSize       int64
	Concurrency    int
	BufferProvider s3manager.WriterReadFromProvider
}

func (c *SDKConfig) SetupFlags(prefix string, flagset *flag.FlagSet) {
	prefix += "sdk."

	flagset.Int64Var(&c.PartSize, prefix+"part-size", s3manager.DefaultDownloadPartSize,
		"Specifies the `size` of parts of the object to download.")
	flagset.IntVar(&c.Concurrency, prefix+"concurrency", s3manager.DefaultDownloadConcurrency,
		"Specifies the number of parts to download `at once`.")
}

func (c *SDKConfig) Validate() error {
	return nil
}

type ClientConfig struct {
	KeepAlive bool
	Timeouts  Timeouts

	MaxIdleConns        int
	MaxIdleConnsPerHost int

	// Go 1.13
	ReadBufferSize  int
	WriteBufferSize int
}

func (c *ClientConfig) SetupFlags(prefix string, flagset *flag.FlagSet) {
	prefix += "client."

	flagset.BoolVar(&c.KeepAlive, prefix+"http-keep-alive", true,
		"Specifies if HTTP keep alive is enabled.")

	defTR := http.DefaultTransport.(*http.Transport)

	flagset.IntVar(&c.MaxIdleConns, prefix+"idle-conns", defTR.MaxIdleConns,
		"Specifies max idle connection pool size.")

	flagset.IntVar(&c.MaxIdleConnsPerHost, prefix+"idle-conns-host", http.DefaultMaxIdleConnsPerHost,
		"Specifies max idle connection pool per host, will be truncated by idle-conns.")

	flagset.IntVar(&c.ReadBufferSize, prefix+"read-buffer", defTR.ReadBufferSize, "size of the transport read buffer used")
	flagset.IntVar(&c.WriteBufferSize, prefix+"writer-buffer", defTR.WriteBufferSize, "size of the transport write buffer used")

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

	flagset.DurationVar(&c.ExpectContinue, prefix+"expect-continue", defTR.ExpectContinueTimeout,
		"The `timeout` waiting for the TLS handshake to complete.")

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
