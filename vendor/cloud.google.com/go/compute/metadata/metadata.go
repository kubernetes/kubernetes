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
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"runtime"
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

var defaultClient = &Client{hc: newDefaultHTTPClient()}

func newDefaultHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Dial: (&net.Dialer{
				Timeout:   2 * time.Second,
				KeepAlive: 30 * time.Second,
			}).Dial,
		},
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

func (c *cachedValue) get(cl *Client) (v string, err error) {
	defer c.mu.Unlock()
	c.mu.Lock()
	if c.v != "" {
		return c.v, nil
	}
	if c.trim {
		v, err = cl.getTrimmed(c.k)
	} else {
		v, err = cl.Get(c.k)
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

// OnGCE reports whether this process is running on Google Compute Engine.
func OnGCE() bool {
	onGCEOnce.Do(initOnGCE)
	return onGCE
}

func initOnGCE() {
	onGCE = testOnGCE()
}

func testOnGCE() bool {
	// The user explicitly said they're on GCE, so trust them.
	if os.Getenv(metadataHostEnv) != "" {
		return true
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	resc := make(chan bool, 2)

	// Try two strategies in parallel.
	// See https://github.com/googleapis/google-cloud-go/issues/194
	go func() {
		req, _ := http.NewRequest("GET", "http://"+metadataIP, nil)
		req.Header.Set("User-Agent", userAgent)
		res, err := newDefaultHTTPClient().Do(req.WithContext(ctx))
		if err != nil {
			resc <- false
			return
		}
		defer res.Body.Close()
		resc <- res.Header.Get("Metadata-Flavor") == "Google"
	}()

	go func() {
		resolver := &net.Resolver{}
		addrs, err := resolver.LookupHost(ctx, "metadata.google.internal")
		if err != nil || len(addrs) == 0 {
			resc <- false
			return
		}
		resc <- strsContains(addrs, metadataIP)
	}()

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
		timer := time.NewTimer(5 * time.Second)
		defer timer.Stop()
		select {
		case res = <-resc:
			return res
		case <-timer.C:
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

// systemInfoSuggestsGCE reports whether the local system (without
// doing network requests) suggests that we're running on GCE. If this
// returns true, testOnGCE tries a bit harder to reach its metadata
// server.
func systemInfoSuggestsGCE() bool {
	if runtime.GOOS != "linux" {
		// We don't have any non-Linux clues available, at least yet.
		return false
	}
	slurp, _ := ioutil.ReadFile("/sys/class/dmi/id/product_name")
	name := strings.TrimSpace(string(slurp))
	return name == "Google" || name == "Google Compute Engine"
}

// Subscribe calls Client.Subscribe on the default client.
func Subscribe(suffix string, fn func(v string, ok bool) error) error {
	return defaultClient.Subscribe(suffix, fn)
}

// Get calls Client.Get on the default client.
func Get(suffix string) (string, error) { return defaultClient.Get(suffix) }

// ProjectID returns the current instance's project ID string.
func ProjectID() (string, error) { return defaultClient.ProjectID() }

// NumericProjectID returns the current instance's numeric project ID.
func NumericProjectID() (string, error) { return defaultClient.NumericProjectID() }

// InternalIP returns the instance's primary internal IP address.
func InternalIP() (string, error) { return defaultClient.InternalIP() }

// ExternalIP returns the instance's primary external (public) IP address.
func ExternalIP() (string, error) { return defaultClient.ExternalIP() }

// Email calls Client.Email on the default client.
func Email(serviceAccount string) (string, error) { return defaultClient.Email(serviceAccount) }

// Hostname returns the instance's hostname. This will be of the form
// "<instanceID>.c.<projID>.internal".
func Hostname() (string, error) { return defaultClient.Hostname() }

// InstanceTags returns the list of user-defined instance tags,
// assigned when initially creating a GCE instance.
func InstanceTags() ([]string, error) { return defaultClient.InstanceTags() }

// InstanceID returns the current VM's numeric instance ID.
func InstanceID() (string, error) { return defaultClient.InstanceID() }

// InstanceName returns the current VM's instance ID string.
func InstanceName() (string, error) { return defaultClient.InstanceName() }

// Zone returns the current VM's zone, such as "us-central1-b".
func Zone() (string, error) { return defaultClient.Zone() }

// InstanceAttributes calls Client.InstanceAttributes on the default client.
func InstanceAttributes() ([]string, error) { return defaultClient.InstanceAttributes() }

// ProjectAttributes calls Client.ProjectAttributes on the default client.
func ProjectAttributes() ([]string, error) { return defaultClient.ProjectAttributes() }

// InstanceAttributeValue calls Client.InstanceAttributeValue on the default client.
func InstanceAttributeValue(attr string) (string, error) {
	return defaultClient.InstanceAttributeValue(attr)
}

// ProjectAttributeValue calls Client.ProjectAttributeValue on the default client.
func ProjectAttributeValue(attr string) (string, error) {
	return defaultClient.ProjectAttributeValue(attr)
}

// Scopes calls Client.Scopes on the default client.
func Scopes(serviceAccount string) ([]string, error) { return defaultClient.Scopes(serviceAccount) }

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
	hc *http.Client
}

// NewClient returns a Client that can be used to fetch metadata.
// Returns the client that uses the specified http.Client for HTTP requests.
// If nil is specified, returns the default client.
func NewClient(c *http.Client) *Client {
	if c == nil {
		return defaultClient
	}

	return &Client{hc: c}
}

// getETag returns a value from the metadata service as well as the associated ETag.
// This func is otherwise equivalent to Get.
func (c *Client) getETag(suffix string) (value, etag string, err error) {
	ctx := context.TODO()
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
	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Metadata-Flavor", "Google")
	req.Header.Set("User-Agent", userAgent)
	var res *http.Response
	var reqErr error
	retryer := newRetryer()
	for {
		res, reqErr = c.hc.Do(req)
		var code int
		if res != nil {
			code = res.StatusCode
		}
		if delay, shouldRetry := retryer.Retry(code, reqErr); shouldRetry {
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
	defer res.Body.Close()
	if res.StatusCode == http.StatusNotFound {
		return "", "", NotDefinedError(suffix)
	}
	all, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return "", "", err
	}
	if res.StatusCode != 200 {
		return "", "", &Error{Code: res.StatusCode, Message: string(all)}
	}
	return string(all), res.Header.Get("Etag"), nil
}

// Get returns a value from the metadata service.
// The suffix is appended to "http://${GCE_METADATA_HOST}/computeMetadata/v1/".
//
// If the GCE_METADATA_HOST environment variable is not defined, a default of
// 169.254.169.254 will be used instead.
//
// If the requested metadata is not defined, the returned error will
// be of type NotDefinedError.
func (c *Client) Get(suffix string) (string, error) {
	val, _, err := c.getETag(suffix)
	return val, err
}

func (c *Client) getTrimmed(suffix string) (s string, err error) {
	s, err = c.Get(suffix)
	s = strings.TrimSpace(s)
	return
}

func (c *Client) lines(suffix string) ([]string, error) {
	j, err := c.Get(suffix)
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
func (c *Client) ProjectID() (string, error) { return projID.get(c) }

// NumericProjectID returns the current instance's numeric project ID.
func (c *Client) NumericProjectID() (string, error) { return projNum.get(c) }

// InstanceID returns the current VM's numeric instance ID.
func (c *Client) InstanceID() (string, error) { return instID.get(c) }

// InternalIP returns the instance's primary internal IP address.
func (c *Client) InternalIP() (string, error) {
	return c.getTrimmed("instance/network-interfaces/0/ip")
}

// Email returns the email address associated with the service account.
// The account may be empty or the string "default" to use the instance's
// main account.
func (c *Client) Email(serviceAccount string) (string, error) {
	if serviceAccount == "" {
		serviceAccount = "default"
	}
	return c.getTrimmed("instance/service-accounts/" + serviceAccount + "/email")
}

// ExternalIP returns the instance's primary external (public) IP address.
func (c *Client) ExternalIP() (string, error) {
	return c.getTrimmed("instance/network-interfaces/0/access-configs/0/external-ip")
}

// Hostname returns the instance's hostname. This will be of the form
// "<instanceID>.c.<projID>.internal".
func (c *Client) Hostname() (string, error) {
	return c.getTrimmed("instance/hostname")
}

// InstanceTags returns the list of user-defined instance tags,
// assigned when initially creating a GCE instance.
func (c *Client) InstanceTags() ([]string, error) {
	var s []string
	j, err := c.Get("instance/tags")
	if err != nil {
		return nil, err
	}
	if err := json.NewDecoder(strings.NewReader(j)).Decode(&s); err != nil {
		return nil, err
	}
	return s, nil
}

// InstanceName returns the current VM's instance ID string.
func (c *Client) InstanceName() (string, error) {
	return c.getTrimmed("instance/name")
}

// Zone returns the current VM's zone, such as "us-central1-b".
func (c *Client) Zone() (string, error) {
	zone, err := c.getTrimmed("instance/zone")
	// zone is of the form "projects/<projNum>/zones/<zoneName>".
	if err != nil {
		return "", err
	}
	return zone[strings.LastIndex(zone, "/")+1:], nil
}

// InstanceAttributes returns the list of user-defined attributes,
// assigned when initially creating a GCE VM instance. The value of an
// attribute can be obtained with InstanceAttributeValue.
func (c *Client) InstanceAttributes() ([]string, error) { return c.lines("instance/attributes/") }

// ProjectAttributes returns the list of user-defined attributes
// applying to the project as a whole, not just this VM.  The value of
// an attribute can be obtained with ProjectAttributeValue.
func (c *Client) ProjectAttributes() ([]string, error) { return c.lines("project/attributes/") }

// InstanceAttributeValue returns the value of the provided VM
// instance attribute.
//
// If the requested attribute is not defined, the returned error will
// be of type NotDefinedError.
//
// InstanceAttributeValue may return ("", nil) if the attribute was
// defined to be the empty string.
func (c *Client) InstanceAttributeValue(attr string) (string, error) {
	return c.Get("instance/attributes/" + attr)
}

// ProjectAttributeValue returns the value of the provided
// project attribute.
//
// If the requested attribute is not defined, the returned error will
// be of type NotDefinedError.
//
// ProjectAttributeValue may return ("", nil) if the attribute was
// defined to be the empty string.
func (c *Client) ProjectAttributeValue(attr string) (string, error) {
	return c.Get("project/attributes/" + attr)
}

// Scopes returns the service account scopes for the given account.
// The account may be empty or the string "default" to use the instance's
// main account.
func (c *Client) Scopes(serviceAccount string) ([]string, error) {
	if serviceAccount == "" {
		serviceAccount = "default"
	}
	return c.lines("instance/service-accounts/" + serviceAccount + "/scopes")
}

// Subscribe subscribes to a value from the metadata service.
// The suffix is appended to "http://${GCE_METADATA_HOST}/computeMetadata/v1/".
// The suffix may contain query parameters.
//
// Subscribe calls fn with the latest metadata value indicated by the provided
// suffix. If the metadata value is deleted, fn is called with the empty string
// and ok false. Subscribe blocks until fn returns a non-nil error or the value
// is deleted. Subscribe returns the error value returned from the last call to
// fn, which may be nil when ok == false.
func (c *Client) Subscribe(suffix string, fn func(v string, ok bool) error) error {
	const failedSubscribeSleep = time.Second * 5

	// First check to see if the metadata value exists at all.
	val, lastETag, err := c.getETag(suffix)
	if err != nil {
		return err
	}

	if err := fn(val, true); err != nil {
		return err
	}

	ok := true
	if strings.ContainsRune(suffix, '?') {
		suffix += "&wait_for_change=true&last_etag="
	} else {
		suffix += "?wait_for_change=true&last_etag="
	}
	for {
		val, etag, err := c.getETag(suffix + url.QueryEscape(lastETag))
		if err != nil {
			if _, deleted := err.(NotDefinedError); !deleted {
				time.Sleep(failedSubscribeSleep)
				continue // Retry on other errors.
			}
			ok = false
		}
		lastETag = etag

		if err := fn(val, ok); err != nil || !ok {
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
