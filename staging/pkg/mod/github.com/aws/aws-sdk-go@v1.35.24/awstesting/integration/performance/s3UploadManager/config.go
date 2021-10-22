// +build integration,perftest

package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/service/s3/s3manager"
)

type Config struct {
	Bucket     string
	Filename   string
	Size       int64
	TempDir    string
	LogVerbose bool

	SDK    SDKConfig
	Client ClientConfig
}

func (c *Config) SetupFlags(prefix string, flagset *flag.FlagSet) {
	flagset.StringVar(&c.Bucket, "bucket", "",
		"The S3 bucket `name` to upload the object to.")
	flagset.StringVar(&c.Filename, "file", "",
		"The `path` of the local file to upload.")
	flagset.Int64Var(&c.Size, "size", 0,
		"The S3 object size in bytes to upload")
	flagset.StringVar(&c.TempDir, "temp", os.TempDir(), "location to create temporary files")
	flagset.BoolVar(&c.LogVerbose, "verbose", false,
		"The output log will include verbose request information")

	c.SDK.SetupFlags(prefix, flagset)
	c.Client.SetupFlags(prefix, flagset)
}

func (c *Config) Validate() error {
	var errs Errors

	if len(c.Bucket) == 0 || (c.Size <= 0 && c.Filename == "") {
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
	PartSize            int64
	Concurrency         int
	WithUnsignedPayload bool
	WithContentMD5      bool
	ExpectContinue      bool
	BufferProvider      s3manager.ReadSeekerWriteToProvider
}

func (c *SDKConfig) SetupFlags(prefix string, flagset *flag.FlagSet) {
	prefix += "sdk."

	flagset.Int64Var(&c.PartSize, prefix+"part-size", s3manager.DefaultUploadPartSize,
		"Specifies the `size` of parts of the object to upload.")
	flagset.IntVar(&c.Concurrency, prefix+"concurrency", s3manager.DefaultUploadConcurrency,
		"Specifies the number of parts to upload `at once`.")
	flagset.BoolVar(&c.WithUnsignedPayload, prefix+"unsigned", false,
		"Specifies if the SDK will use UNSIGNED_PAYLOAD for part SHA256 in request signature.")
	flagset.BoolVar(&c.WithContentMD5, prefix+"content-md5", true,
		"Specifies if the SDK should compute the content md5 header for S3 uploads.")

	flagset.BoolVar(&c.ExpectContinue, prefix+"100-continue", true,
		"Specifies if the SDK requests will wait for the 100 continue response before sending request payload.")
}

func (c *SDKConfig) Validate() error {
	return nil
}

type ClientConfig struct {
	KeepAlive bool
	Timeouts  Timeouts

	MaxIdleConns        int
	MaxIdleConnsPerHost int
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
