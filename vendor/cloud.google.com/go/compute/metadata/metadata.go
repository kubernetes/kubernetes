// Copyright 2014 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package metadata provides access to Google Compute Engine (GCE)
// metadata and API service accounts.
//
// This package is a wrapper around the GCE metadata service,
// as documented at https://cloud.google.com/compute/docs/metadata/overview.
package metadata // import "cloud.google.com/go/compute/metadata"

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	// metadataIP is the documented metadata server IP address.
	metadataIP = "169.254.169.254"

	// metadataHostEnv is the environment variable specifying the
	// GCE metadata hostname.  If empty, the default value of
	// metadataIP ("169.254.169.254") is used instead.
	// This is variable name is not defined by any spec, as far as
	// I know; it was made up for the Go package.
	metadataHostEnv = "GCE_METADATA_HOST"

	userAgent = "gcloud-golang/0.1"
)

type cachedValue struct {
	k    string
	trim bool
	mu   sync.Mutex
	v    string
}

var (
	projID  = &cachedValue{k: "project/project-id", trim: true}
	projNum = &cachedValue{k: "project/numeric-project-id", trim: true}
	instID  = &cachedValue{k: "instance/id", trim: true}
)

var defaultClient = &Client{
	hc:     newDefaultHTTPClient(),
	logger: slog.New(noOpHandler{}),
}

func newDefaultHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Dial: (&net.Dialer{
				Timeout:   2 * time.Second,
				KeepAlive: 30 * time.Second,
			}).Dial,
			IdleConnTimeout: 60 * time.Second,
		},
		Timeout: 5 * time.Second,
	}
}

// NotDefinedError is returned when requested metadata is not defined.
//
// The underlying string is the suffix after "/computeMetadata/v1/".
//
// This error is not returned if the value is defined to be the empty
// string.
type NotDefinedError string

func (suffix NotDefinedError) Error() string {
	return fmt.Sprintf("metadata: GCE metadata %q not defined", string(suffix))
}

func (c *cachedValue) get(ctx context.Context, cl *Client) (v string, err error) {
	defer c.mu.Unlock()
	c.mu.Lock()
	if c.v != "" {
		return c.v, nil
	}
	if c.trim {
		v, err = cl.getTrimmed(ctx, c.k)
	} else {
		v, err = cl.GetWithContext(ctx, c.k)
	}
	if err == nil {
		c.v = v
	}
	return
}

var (
	onGCEOnce sync.Once
	onGCE     bool
)

// OnGCE reports whether this process is running on Google Compute Platforms.
// NOTE: True returned from `OnGCE` does not guarantee that the metadata server
// is accessible from this process and have all the metadata defined.
func OnGCE() bool {
	return OnGCEWithContext(context.Background())
}

// OnGCEWithContext reports whether this process is running on Google Compute Platforms.
// This function's return value is memoized for better performance.
// NOTE: True returned from `OnGCEWithContext` does not guarantee that the metadata server
// is accessible from this process and have all the metadata defined.
func OnGCEWithContext(ctx context.Context) bool {
	onGCEOnce.Do(func() {
		onGCE = defaultClient.OnGCEWithContext(ctx)
	})
	return onGCE
}

// Subscribe calls Client.SubscribeWithContext on the default client.
//
// Deprecated: Please use the context aware variant [SubscribeWithContext].
func Subscribe(suffix string, fn func(v string, ok bool) error) error {
	return defaultClient.SubscribeWithContext(context.Background(), suffix, func(ctx context.Context, v string, ok bool) error { return fn(v, ok) })
}

// SubscribeWithContext calls Client.SubscribeWithContext on the default client.
func SubscribeWithContext(ctx context.Context, suffix string, fn func(ctx context.Context, v string, ok bool) error) error {
	return defaultClient.SubscribeWithContext(ctx, suffix, fn)
}

// Get calls Client.GetWithContext on the default client.
//
// Deprecated: Please use the context aware variant [GetWithContext].
func Get(suffix string) (string, error) {
	return defaultClient.GetWithContext(context.Background(), suffix)
}

// GetWithContext calls Client.GetWithContext on the default client.
func GetWithContext(ctx context.Context, suffix string) (string, error) {
	return defaultClient.GetWithContext(ctx, suffix)
}

// ProjectID returns the current instance's project ID string.
//
// Deprecated: Please use the context aware variant [ProjectIDWithContext].
func ProjectID() (string, error) {
	return defaultClient.ProjectIDWithContext(context.Background())
}

// ProjectIDWithContext returns the current instance's project ID string.
func ProjectIDWithContext(ctx context.Context) (string, error) {
	return defaultClient.ProjectIDWithContext(ctx)
}

// NumericProjectID returns the current instance's numeric project ID.
//
// Deprecated: Please use the context aware variant [NumericProjectIDWithContext].
func NumericProjectID() (string, error) {
	return defaultClient.NumericProjectIDWithContext(context.Background())
}

// NumericProjectIDWithContext returns the current instance's numeric project ID.
func NumericProjectIDWithContext(ctx context.Context) (string, error) {
	return defaultClient.NumericProjectIDWithContext(ctx)
}

// InternalIP returns the instance's primary internal IP address.
//
// Deprecated: Please use the context aware variant [InternalIPWithContext].
func InternalIP() (string, error) {
	return defaultClient.InternalIPWithContext(context.Background())
}

// InternalIPWithContext returns the instance's primary internal IP address.
func InternalIPWithContext(ctx context.Context) (string, error) {
	return defaultClient.InternalIPWithContext(ctx)
}

// ExternalIP returns the instance's primary external (public) IP address.
//
// Deprecated: Please use the context aware variant [ExternalIPWithContext].
func ExternalIP() (string, error) {
	return defaultClient.ExternalIPWithContext(context.Background())
}

// ExternalIPWithContext returns the instance's primary external (public) IP address.
func ExternalIPWithContext(ctx context.Context) (string, error) {
	return defaultClient.ExternalIPWithContext(ctx)
}

// Email calls Client.EmailWithContext on the default client.
//
// Deprecated: Please use the context aware variant [EmailWithContext].
func Email(serviceAccount string) (string, error) {
	return defaultClient.EmailWithContext(context.Background(), serviceAccount)
}

// EmailWithContext calls Client.EmailWithContext on the default client.
func EmailWithContext(ctx context.Context, serviceAccount string) (string, error) {
	return defaultClient.EmailWithContext(ctx, serviceAccount)
}

// Hostname returns the instance's hostname. This will be of the form
// "<instanceID>.c.<projID>.internal".
//
// Deprecated: Please use the context aware variant [HostnameWithContext].
func Hostname() (string, error) {
	return defaultClient.HostnameWithContext(context.Background())
}

// HostnameWithContext returns the instance's hostname. This will be of the form
// "<instanceID>.c.<projID>.internal".
func HostnameWithContext(ctx context.Context) (string, error) {
	return defaultClient.HostnameWithContext(ctx)
}

// InstanceTags returns the list of user-defined instance tags,
// assigned when initially creating a GCE instance.
//
// Deprecated: Please use the context aware variant [InstanceTagsWithContext].
func InstanceTags() ([]string, error) {
	return defaultClient.InstanceTagsWithContext(context.Background())
}

// InstanceTagsWithContext returns the list of user-defined instance tags,
// assigned when initially creating a GCE instance.
func InstanceTagsWithContext(ctx context.Context) ([]string, error) {
	return defaultClient.InstanceTagsWithContext(ctx)
}

// InstanceID returns the current VM's numeric instance ID.
//
// Deprecated: Please use the context aware variant [InstanceIDWithContext].
func InstanceID() (string, error) {
	return defaultClient.InstanceIDWithContext(context.Background())
}

// InstanceIDWithContext returns the current VM's numeric instance ID.
func InstanceIDWithContext(ctx context.Context) (string, error) {
	return defaultClient.InstanceIDWithContext(ctx)
}

// InstanceName returns the current VM's instance ID string.
//
// Deprecated: Please use the context aware variant [InstanceNameWithContext].
func InstanceName() (string, error) {
	return defaultClient.InstanceNameWithContext(context.Background())
}

// InstanceNameWithContext returns the current VM's instance ID string.
func InstanceNameWithContext(ctx context.Context) (string, error) {
	return defaultClient.InstanceNameWithContext(ctx)
}

// Zone returns the current VM's zone, such as "us-central1-b".
//
// Deprecated: Please use the context aware variant [ZoneWithContext].
func Zone() (string, error) {
	return defaultClient.ZoneWithContext(context.Background())
}

// ZoneWithContext returns the current VM's zone, such as "us-central1-b".
func ZoneWithContext(ctx context.Context) (string, error) {
	return defaultClient.ZoneWithContext(ctx)
}

// InstanceAttributes calls Client.InstanceAttributesWithContext on the default client.
//
// Deprecated: Please use the context aware variant [InstanceAttributesWithContext.
func InstanceAttributes() ([]string, error) {
	return defaultClient.InstanceAttributesWithContext(context.Background())
}

// InstanceAttributesWithContext calls Client.ProjectAttributesWithContext on the default client.
func InstanceAttributesWithContext(ctx context.Context) ([]string, error) {
	return defaultClient.InstanceAttributesWithContext(ctx)
}

// ProjectAttributes calls Client.ProjectAttributesWithContext on the default client.
//
// Deprecated: Please use the context aware variant [ProjectAttributesWithContext].
func ProjectAttributes() ([]string, error) {
	return defaultClient.ProjectAttributesWithContext(context.Background())
}

// ProjectAttributesWithContext calls Client.ProjectAttributesWithContext on the default client.
func ProjectAttributesWithContext(ctx context.Context) ([]string, error) {
	return defaultClient.ProjectAttributesWithContext(ctx)
}

// InstanceAttributeValue calls Client.InstanceAttributeValueWithContext on the default client.
//
// Deprecated: Please use the context aware variant [InstanceAttributeValueWithContext].
func InstanceAttributeValue(attr string) (string, error) {
	return defaultClient.InstanceAttributeValueWithContext(context.Background(), attr)
}

// InstanceAttributeValueWithContext calls Client.InstanceAttributeValueWithContext on the default client.
func InstanceAttributeValueWithContext(ctx context.Context, attr string) (string, error) {
	return defaultClient.InstanceAttributeValueWithContext(ctx, attr)
}

// ProjectAttributeValue calls Client.ProjectAttributeValueWithContext on the default client.
//
// Deprecated: Please use the context aware variant [ProjectAttributeValueWithContext].
func ProjectAttributeValue(attr string) (string, error) {
	return defaultClient.ProjectAttributeValueWithContext(context.Background(), attr)
}

// ProjectAttributeValueWithContext calls Client.ProjectAttributeValueWithContext on the default client.
func ProjectAttributeValueWithContext(ctx context.Context, attr string) (string, error) {
	return defaultClient.ProjectAttributeValueWithContext(ctx, attr)
}

// Scopes calls Client.ScopesWithContext on the default client.
//
// Deprecated: Please use the context aware variant [ScopesWithContext].
func Scopes(serviceAccount string) ([]string, error) {
	return defaultClient.ScopesWithContext(context.Background(), serviceAccount)
}

// ScopesWithContext calls Client.ScopesWithContext on the default client.
func ScopesWithContext(ctx context.Context, serviceAccount string) ([]string, error) {
	return defaultClient.ScopesWithContext(ctx, serviceAccount)
}

func strsContains(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}

// A Client provides metadata.
type Client struct {
	hc     *http.Client
	logger *slog.Logger
}

// Options for configuring a [Client].
type Options struct {
	// Client is the HTTP client used to make requests. Optional.
	Client *http.Client
	// Logger is used to log information about HTTP request and responses.
	// If not provided, nothing will be logged. Optional.
	Logger *slog.Logger
}

// NewClient returns a Client that can be used to fetch metadata.
// Returns the client that uses the specified http.Client for HTTP requests.
// If nil is specified, returns the default client.
func NewClient(c *http.Client) *Client {
	return NewWithOptions(&Options{
		Client: c,
	})
}

// NewWithOptions returns a Client that is configured with the provided Options.
func NewWithOptions(opts *Options) *Client {
	if opts == nil {
		return defaultClient
	}
	client := opts.Client
	if client == nil {
		client = newDefaultHTTPClient()
	}
	logger := opts.Logger
	if logger == nil {
		logger = slog.New(noOpHandler{})
	}
	return &Client{hc: client, logger: logger}
}

// NOTE: metadataRequestStrategy is assigned to a variable for test stubbing purposes.
var metadataRequestStrategy = func(ctx context.Context, httpClient *http.Client, resc chan bool) {
	req, _ := http.NewRequest("GET", "http://"+metadataIP, nil)
	req.Header.Set("User-Agent", userAgent)
	res, err := httpClient.Do(req.WithContext(ctx))
	if err != nil {
		resc <- false
		return
	}
	defer res.Body.Close()
	resc <- res.Header.Get("Metadata-Flavor") == "Google"
}

// NOTE: dnsRequestStrategy is assigned to a variable for test stubbing purposes.
var dnsRequestStrategy = func(ctx context.Context, resc chan bool) {
	resolver := &net.Resolver{}
	addrs, err := resolver.LookupHost(ctx, "metadata.google.internal.")
	if err != nil || len(addrs) == 0 {
		resc <- false
		return
	}
	resc <- strsContains(addrs, metadataIP)
}

// OnGCEWithContext reports whether this process is running on Google Compute Platforms.
// NOTE: True returned from `OnGCEWithContext` does not guarantee that the metadata server
// is accessible from this process and have all the metadata defined.
func (c *Client) OnGCEWithContext(ctx context.Context) bool {
	// The user explicitly said they're on GCE, so trust them.
	if os.Getenv(metadataHostEnv) != "" {
		return true
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	resc := make(chan bool, 2)

	// Try two strategies in parallel.
	// See https://github.com/googleapis/google-cloud-go/issues/194
	go metadataRequestStrategy(ctx, c.hc, resc)
	go dnsRequestStrategy(ctx, resc)

	tryHarder := systemInfoSuggestsGCE()
	if tryHarder {
		res := <-resc
		if res {
			// The first strategy succeeded, so let's use it.
			return true
		}

		// Wait for either the DNS or metadata server probe to
		// contradict the other one and say we are running on
		// GCE. Give it a lot of time to do so, since the system
		// info already suggests we're running on a GCE BIOS.
		// Ensure cancellations from the calling context are respected.
		waitContext, cancelWait := context.WithTimeout(ctx, 5*time.Second)
		defer cancelWait()
		select {
		case res = <-resc:
			return res
		case <-waitContext.Done():
			// Too slow. Who knows what this system is.
			return false
		}
	}

	// There's no hint from the system info that we're running on
	// GCE, so use the first probe's result as truth, whether it's
	// true or false. The goal here is to optimize for speed for
	// users who are NOT running on GCE. We can't assume that
	// either a DNS lookup or an HTTP request to a blackholed IP
	// address is fast. Worst case this should return when the
	// metaClient's Transport.ResponseHeaderTimeout or
	// Transport.Dial.Timeout fires (in two seconds).
	return <-resc
}

// getETag returns a value from the metadata service as well as the associated ETag.
// This func is otherwise equivalent to Get.
func (c *Client) getETag(ctx context.Context, suffix string) (value, etag string, err error) {
	// Using a fixed IP makes it very difficult to spoof the metadata service in
	// a container, which is an important use-case for local testing of cloud
	// deployments. To enable spoofing of the metadata service, the environment
	// variable GCE_METADATA_HOST is first inspected to decide where metadata
	// requests shall go.
	host := os.Getenv(metadataHostEnv)
	if host == "" {
		// Using 169.254.169.254 instead of "metadata" here because Go
		// binaries built with the "netgo" tag and without cgo won't
		// know the search suffix for "metadata" is
		// ".google.internal", and this IP address is documented as
		// being stable anyway.
		host = metadataIP
	}
	suffix = strings.TrimLeft(suffix, "/")
	u := "http://" + host + "/computeMetadata/v1/" + suffix
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Metadata-Flavor", "Google")
	req.Header.Set("User-Agent", userAgent)
	var res *http.Response
	var reqErr error
	var body []byte
	retryer := newRetryer()
	for {
		c.logger.DebugContext(ctx, "metadata request", "request", httpRequest(req, nil))
		res, reqErr = c.hc.Do(req)
		var code int
		if res != nil {
			code = res.StatusCode
			body, err = io.ReadAll(res.Body)
			if err != nil {
				res.Body.Close()
				return "", "", err
			}
			c.logger.DebugContext(ctx, "metadata response", "response", httpResponse(res, body))
			res.Body.Close()
		}
		if delay, shouldRetry := retryer.Retry(code, reqErr); shouldRetry {
			if res != nil && res.Body != nil {
				res.Body.Close()
			}
			if err := sleep(ctx, delay); err != nil {
				return "", "", err
			}
			continue
		}
		break
	}
	if reqErr != nil {
		return "", "", reqErr
	}
	if res.StatusCode == http.StatusNotFound {
		return "", "", NotDefinedError(suffix)
	}
	if res.StatusCode != 200 {
		return "", "", &Error{Code: res.StatusCode, Message: string(body)}
	}
	return string(body), res.Header.Get("Etag"), nil
}

// Get returns a value from the metadata service.
// The suffix is appended to "http://${GCE_METADATA_HOST}/computeMetadata/v1/".
//
// If the GCE_METADATA_HOST environment variable is not defined, a default of
// 169.254.169.254 will be used instead.
//
// If the requested metadata is not defined, the returned error will
// be of type NotDefinedError.
//
// Deprecated: Please use the context aware variant [Client.GetWithContext].
func (c *Client) Get(suffix string) (string, error) {
	return c.GetWithContext(context.Background(), suffix)
}

// GetWithContext returns a value from the metadata service.
// The suffix is appended to "http://${GCE_METADATA_HOST}/computeMetadata/v1/".
//
// If the GCE_METADATA_HOST environment variable is not defined, a default of
// 169.254.169.254 will be used instead.
//
// If the requested metadata is not defined, the returned error will
// be of type NotDefinedError.
//
// NOTE: Without an extra deadline in the context this call can take in the
// worst case, with internal backoff retries, up to 15 seconds (e.g. when server
// is responding slowly). Pass context with additional timeouts when needed.
func (c *Client) GetWithContext(ctx context.Context, suffix string) (string, error) {
	val, _, err := c.getETag(ctx, suffix)
	return val, err
}

func (c *Client) getTrimmed(ctx context.Context, suffix string) (s string, err error) {
	s, err = c.GetWithContext(ctx, suffix)
	s = strings.TrimSpace(s)
	return
}

func (c *Client) lines(ctx context.Context, suffix string) ([]string, error) {
	j, err := c.GetWithContext(ctx, suffix)
	if err != nil {
		return nil, err
	}
	s := strings.Split(strings.TrimSpace(j), "\n")
	for i := range s {
		s[i] = strings.TrimSpace(s[i])
	}
	return s, nil
}

// ProjectID returns the current instance's project ID string.
//
// Deprecated: Please use the context aware variant [Client.ProjectIDWithContext].
func (c *Client) ProjectID() (string, error) { return c.ProjectIDWithContext(context.Background()) }

// ProjectIDWithContext returns the current instance's project ID string.
func (c *Client) ProjectIDWithContext(ctx context.Context) (string, error) { return projID.get(ctx, c) }

// NumericProjectID returns the current instance's numeric project ID.
//
// Deprecated: Please use the context aware variant [Client.NumericProjectIDWithContext].
func (c *Client) NumericProjectID() (string, error) {
	return c.NumericProjectIDWithContext(context.Background())
}

// NumericProjectIDWithContext returns the current instance's numeric project ID.
func (c *Client) NumericProjectIDWithContext(ctx context.Context) (string, error) {
	return projNum.get(ctx, c)
}

// InstanceID returns the current VM's numeric instance ID.
//
// Deprecated: Please use the context aware variant [Client.InstanceIDWithContext].
func (c *Client) InstanceID() (string, error) {
	return c.InstanceIDWithContext(context.Background())
}

// InstanceIDWithContext returns the current VM's numeric instance ID.
func (c *Client) InstanceIDWithContext(ctx context.Context) (string, error) {
	return instID.get(ctx, c)
}

// InternalIP returns the instance's primary internal IP address.
//
// Deprecated: Please use the context aware variant [Client.InternalIPWithContext].
func (c *Client) InternalIP() (string, error) {
	return c.InternalIPWithContext(context.Background())
}

// InternalIPWithContext returns the instance's primary internal IP address.
func (c *Client) InternalIPWithContext(ctx context.Context) (string, error) {
	return c.getTrimmed(ctx, "instance/network-interfaces/0/ip")
}

// Email returns the email address associated with the service account.
//
// Deprecated: Please use the context aware variant [Client.EmailWithContext].
func (c *Client) Email(serviceAccount string) (string, error) {
	return c.EmailWithContext(context.Background(), serviceAccount)
}

// EmailWithContext returns the email address associated with the service account.
// The serviceAccount parameter default value (empty string or "default" value)
// will use the instance's main account.
func (c *Client) EmailWithContext(ctx context.Context, serviceAccount string) (string, error) {
	if serviceAccount == "" {
		serviceAccount = "default"
	}
	return c.getTrimmed(ctx, "instance/service-accounts/"+serviceAccount+"/email")
}

// ExternalIP returns the instance's primary external (public) IP address.
//
// Deprecated: Please use the context aware variant [Client.ExternalIPWithContext].
func (c *Client) ExternalIP() (string, error) {
	return c.ExternalIPWithContext(context.Background())
}

// ExternalIPWithContext returns the instance's primary external (public) IP address.
func (c *Client) ExternalIPWithContext(ctx context.Context) (string, error) {
	return c.getTrimmed(ctx, "instance/network-interfaces/0/access-configs/0/external-ip")
}

// Hostname returns the instance's hostname. This will be of the form
// "<instanceID>.c.<projID>.internal".
//
// Deprecated: Please use the context aware variant [Client.HostnameWithContext].
func (c *Client) Hostname() (string, error) {
	return c.HostnameWithContext(context.Background())
}

// HostnameWithContext returns the instance's hostname. This will be of the form
// "<instanceID>.c.<projID>.internal".
func (c *Client) HostnameWithContext(ctx context.Context) (string, error) {
	return c.getTrimmed(ctx, "instance/hostname")
}

// InstanceTags returns the list of user-defined instance tags.
//
// Deprecated: Please use the context aware variant [Client.InstanceTagsWithContext].
func (c *Client) InstanceTags() ([]string, error) {
	return c.InstanceTagsWithContext(context.Background())
}

// InstanceTagsWithContext returns the list of user-defined instance tags,
// assigned when initially creating a GCE instance.
func (c *Client) InstanceTagsWithContext(ctx context.Context) ([]string, error) {
	var s []string
	j, err := c.GetWithContext(ctx, "instance/tags")
	if err != nil {
		return nil, err
	}
	if err := json.NewDecoder(strings.NewReader(j)).Decode(&s); err != nil {
		return nil, err
	}
	return s, nil
}

// InstanceName returns the current VM's instance ID string.
//
// Deprecated: Please use the context aware variant [Client.InstanceNameWithContext].
func (c *Client) InstanceName() (string, error) {
	return c.InstanceNameWithContext(context.Background())
}

// InstanceNameWithContext returns the current VM's instance ID string.
func (c *Client) InstanceNameWithContext(ctx context.Context) (string, error) {
	return c.getTrimmed(ctx, "instance/name")
}

// Zone returns the current VM's zone, such as "us-central1-b".
//
// Deprecated: Please use the context aware variant [Client.ZoneWithContext].
func (c *Client) Zone() (string, error) {
	return c.ZoneWithContext(context.Background())
}

// ZoneWithContext returns the current VM's zone, such as "us-central1-b".
func (c *Client) ZoneWithContext(ctx context.Context) (string, error) {
	zone, err := c.getTrimmed(ctx, "instance/zone")
	// zone is of the form "projects/<projNum>/zones/<zoneName>".
	if err != nil {
		return "", err
	}
	return zone[strings.LastIndex(zone, "/")+1:], nil
}

// InstanceAttributes returns the list of user-defined attributes,
// assigned when initially creating a GCE VM instance. The value of an
// attribute can be obtained with InstanceAttributeValue.
//
// Deprecated: Please use the context aware variant [Client.InstanceAttributesWithContext].
func (c *Client) InstanceAttributes() ([]string, error) {
	return c.InstanceAttributesWithContext(context.Background())
}

// InstanceAttributesWithContext returns the list of user-defined attributes,
// assigned when initially creating a GCE VM instance. The value of an
// attribute can be obtained with InstanceAttributeValue.
func (c *Client) InstanceAttributesWithContext(ctx context.Context) ([]string, error) {
	return c.lines(ctx, "instance/attributes/")
}

// ProjectAttributes returns the list of user-defined attributes
// applying to the project as a whole, not just this VM.  The value of
// an attribute can be obtained with ProjectAttributeValue.
//
// Deprecated: Please use the context aware variant [Client.ProjectAttributesWithContext].
func (c *Client) ProjectAttributes() ([]string, error) {
	return c.ProjectAttributesWithContext(context.Background())
}

// ProjectAttributesWithContext returns the list of user-defined attributes
// applying to the project as a whole, not just this VM.  The value of
// an attribute can be obtained with ProjectAttributeValue.
func (c *Client) ProjectAttributesWithContext(ctx context.Context) ([]string, error) {
	return c.lines(ctx, "project/attributes/")
}

// InstanceAttributeValue returns the value of the provided VM
// instance attribute.
//
// If the requested attribute is not defined, the returned error will
// be of type NotDefinedError.
//
// InstanceAttributeValue may return ("", nil) if the attribute was
// defined to be the empty string.
//
// Deprecated: Please use the context aware variant [Client.InstanceAttributeValueWithContext].
func (c *Client) InstanceAttributeValue(attr string) (string, error) {
	return c.InstanceAttributeValueWithContext(context.Background(), attr)
}

// InstanceAttributeValueWithContext returns the value of the provided VM
// instance attribute.
//
// If the requested attribute is not defined, the returned error will
// be of type NotDefinedError.
//
// InstanceAttributeValue may return ("", nil) if the attribute was
// defined to be the empty string.
func (c *Client) InstanceAttributeValueWithContext(ctx context.Context, attr string) (string, error) {
	return c.GetWithContext(ctx, "instance/attributes/"+attr)
}

// ProjectAttributeValue returns the value of the provided
// project attribute.
//
// If the requested attribute is not defined, the returned error will
// be of type NotDefinedError.
//
// ProjectAttributeValue may return ("", nil) if the attribute was
// defined to be the empty string.
//
// Deprecated: Please use the context aware variant [Client.ProjectAttributeValueWithContext].
func (c *Client) ProjectAttributeValue(attr string) (string, error) {
	return c.ProjectAttributeValueWithContext(context.Background(), attr)
}

// ProjectAttributeValueWithContext returns the value of the provided
// project attribute.
//
// If the requested attribute is not defined, the returned error will
// be of type NotDefinedError.
//
// ProjectAttributeValue may return ("", nil) if the attribute was
// defined to be the empty string.
func (c *Client) ProjectAttributeValueWithContext(ctx context.Context, attr string) (string, error) {
	return c.GetWithContext(ctx, "project/attributes/"+attr)
}

// Scopes returns the service account scopes for the given account.
// The account may be empty or the string "default" to use the instance's
// main account.
//
// Deprecated: Please use the context aware variant [Client.ScopesWithContext].
func (c *Client) Scopes(serviceAccount string) ([]string, error) {
	return c.ScopesWithContext(context.Background(), serviceAccount)
}

// ScopesWithContext returns the service account scopes for the given account.
// The account may be empty or the string "default" to use the instance's
// main account.
func (c *Client) ScopesWithContext(ctx context.Context, serviceAccount string) ([]string, error) {
	if serviceAccount == "" {
		serviceAccount = "default"
	}
	return c.lines(ctx, "instance/service-accounts/"+serviceAccount+"/scopes")
}

// Subscribe subscribes to a value from the metadata service.
// The suffix is appended to "http://${GCE_METADATA_HOST}/computeMetadata/v1/".
// The suffix may contain query parameters.
//
// Deprecated: Please use the context aware variant [Client.SubscribeWithContext].
func (c *Client) Subscribe(suffix string, fn func(v string, ok bool) error) error {
	return c.SubscribeWithContext(context.Background(), suffix, func(ctx context.Context, v string, ok bool) error { return fn(v, ok) })
}

// SubscribeWithContext subscribes to a value from the metadata service.
// The suffix is appended to "http://${GCE_METADATA_HOST}/computeMetadata/v1/".
// The suffix may contain query parameters.
//
// SubscribeWithContext calls fn with the latest metadata value indicated by the
// provided suffix. If the metadata value is deleted, fn is called with the
// empty string and ok false. Subscribe blocks until fn returns a non-nil error
// or the value is deleted. Subscribe returns the error value returned from the
// last call to fn, which may be nil when ok == false.
func (c *Client) SubscribeWithContext(ctx context.Context, suffix string, fn func(ctx context.Context, v string, ok bool) error) error {
	const failedSubscribeSleep = time.Second * 5

	// First check to see if the metadata value exists at all.
	val, lastETag, err := c.getETag(ctx, suffix)
	if err != nil {
		return err
	}

	if err := fn(ctx, val, true); err != nil {
		return err
	}

	ok := true
	if strings.ContainsRune(suffix, '?') {
		suffix += "&wait_for_change=true&last_etag="
	} else {
		suffix += "?wait_for_change=true&last_etag="
	}
	for {
		val, etag, err := c.getETag(ctx, suffix+url.QueryEscape(lastETag))
		if err != nil {
			if _, deleted := err.(NotDefinedError); !deleted {
				time.Sleep(failedSubscribeSleep)
				continue // Retry on other errors.
			}
			ok = false
		}
		lastETag = etag

		if err := fn(ctx, val, ok); err != nil || !ok {
			return err
		}
	}
}

// Error contains an error response from the server.
type Error struct {
	// Code is the HTTP response status code.
	Code int
	// Message is the server response message.
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("compute: Received %d `%s`", e.Code, e.Message)
}
