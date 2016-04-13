package metrics

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

// TODO Instrumentation? To get statistics?

// More detailed error
type HawkularClientError struct {
	msg  string
	Code int
}

func (self *HawkularClientError) Error() string {
	return fmt.Sprintf("Hawkular returned status code %d, error message: %s", self.Code, self.msg)
}

// Client creation and instance config

const (
	base_url string        = "hawkular/metrics"
	timeout  time.Duration = time.Duration(30 * time.Second)
)

type Parameters struct {
	Tenant    string // Technically optional, but requires setting Tenant() option everytime
	Url       string
	TLSConfig *tls.Config
	Token     string
}

type Client struct {
	Tenant string
	url    *url.URL
	client *http.Client
	Token  string
}

type HawkularClient interface {
	Send(*http.Request) (*http.Response, error)
}

// Modifiers

type Modifier func(*http.Request) error

// Override function to replace the Tenant (defaults to Client default)
func Tenant(tenant string) Modifier {
	return func(r *http.Request) error {
		r.Header.Set("Hawkular-Tenant", tenant)
		return nil
	}
}

// Add payload to the request
func Data(data interface{}) Modifier {
	return func(r *http.Request) error {
		jsonb, err := json.Marshal(data)
		if err != nil {
			return err
		}

		b := bytes.NewBuffer(jsonb)
		rc := ioutil.NopCloser(b)
		r.Body = rc

		// fmt.Printf("Sending: %s\n", string(jsonb))

		if b != nil {
			r.ContentLength = int64(b.Len())
		}
		return nil
	}
}

func (self *Client) Url(method string, e ...Endpoint) Modifier {
	// TODO Create composite URLs? Add().Add().. etc? Easier to modify on the fly..
	return func(r *http.Request) error {
		u := self.createUrl(e...)
		r.URL = u
		r.Method = method
		return nil
	}
}

// Filters for querying

type Filter func(r *http.Request)

func Filters(f ...Filter) Modifier {
	return func(r *http.Request) error {
		for _, filter := range f {
			filter(r)
		}
		return nil // Or should filter return err?
	}
}

// Add query parameters
func Param(k string, v string) Filter {
	return func(r *http.Request) {
		q := r.URL.Query()
		q.Set(k, v)
		r.URL.RawQuery = q.Encode()
	}
}

func TypeFilter(t MetricType) Filter {
	return Param("type", t.shortForm())
}

func TagsFilter(t map[string]string) Filter {
	j := tagsEncoder(t)
	return Param("tags", j)
}

// Requires HWKMETRICS-233
func IdFilter(regexp string) Filter {
	return Param("id", regexp)
}

func StartTimeFilter(startTime time.Time) Filter {
	return Param("start", strconv.Itoa(int(startTime.Unix())))
}

func EndTimeFilter(endTime time.Time) Filter {
	return Param("end", strconv.Itoa(int(endTime.Unix())))
}

func BucketsFilter(buckets int) Filter {
	return Param("buckets", strconv.Itoa(buckets))
}

func PercentilesFilter(percentiles []float64) Filter {
	s := make([]string, 0, len(percentiles))
	for _, v := range percentiles {
		s = append(s, fmt.Sprintf("%v", v))
	}
	j := strings.Join(s, ",")
	return Param("percentiles", j)
}

// The SEND method..

func (self *Client) createRequest() *http.Request {
	req := &http.Request{
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(http.Header),
		Host:       self.url.Host,
	}
	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Hawkular-Tenant", self.Tenant)

	if len(self.Token) > 0 {
		req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", self.Token))
	}

	return req
}

func (self *Client) Send(o ...Modifier) (*http.Response, error) {
	// Initialize
	r := self.createRequest()

	// Run all the modifiers
	for _, f := range o {
		err := f(r)
		if err != nil {
			return nil, err
		}
	}

	return self.client.Do(r)
}

// Commands

func prepend(slice []Modifier, a ...Modifier) []Modifier {
	p := make([]Modifier, 0, len(slice)+len(a))
	p = append(p, a...)
	p = append(p, slice...)
	return p
}

// Create new Definition
func (self *Client) Create(md MetricDefinition, o ...Modifier) (bool, error) {
	// Keep the order, add custom prepend
	o = prepend(o, self.Url("POST", TypeEndpoint(md.Type)), Data(md))

	r, err := self.Send(o...)
	if err != nil {
		return false, err
	}

	defer r.Body.Close()

	if r.StatusCode > 399 {
		err = self.parseErrorResponse(r)
		if err, ok := err.(*HawkularClientError); ok {
			if err.Code != http.StatusConflict {
				return false, err
			} else {
				return false, nil
			}
		}
		return false, err
	}
	return true, nil
}

// Fetch definitions
func (self *Client) Definitions(o ...Modifier) ([]*MetricDefinition, error) {
	o = prepend(o, self.Url("GET", TypeEndpoint(Generic)))

	r, err := self.Send(o...)
	if err != nil {
		return nil, err
	}

	defer r.Body.Close()

	if r.StatusCode == http.StatusOK {
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		md := []*MetricDefinition{}
		if b != nil {
			if err = json.Unmarshal(b, &md); err != nil {
				return nil, err
			}
		}
		return md, err
	} else if r.StatusCode > 399 {
		return nil, self.parseErrorResponse(r)
	}

	return nil, nil
}

// Return a single definition
func (self *Client) Definition(t MetricType, id string, o ...Modifier) (*MetricDefinition, error) {
	o = prepend(o, self.Url("GET", TypeEndpoint(t), SingleMetricEndpoint(id)))

	r, err := self.Send(o...)
	if err != nil {
		return nil, err
	}

	defer r.Body.Close()

	if r.StatusCode == http.StatusOK {
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		md := &MetricDefinition{}
		if b != nil {
			if err = json.Unmarshal(b, md); err != nil {
				return nil, err
			}
		}
		return md, err
	} else if r.StatusCode > 399 {
		return nil, self.parseErrorResponse(r)
	}

	return nil, nil
}

// Update tags
func (self *Client) UpdateTags(t MetricType, id string, tags map[string]string, o ...Modifier) error {
	o = prepend(o, self.Url("PUT", TypeEndpoint(t), SingleMetricEndpoint(id), TagEndpoint()), Data(tags))

	r, err := self.Send(o...)
	if err != nil {
		return err
	}

	defer r.Body.Close()

	if r.StatusCode > 399 {
		return self.parseErrorResponse(r)
	}

	return nil
}

// Delete given tags from the definition
func (self *Client) DeleteTags(t MetricType, id string, tags map[string]string, o ...Modifier) error {
	o = prepend(o, self.Url("DELETE", TypeEndpoint(t), SingleMetricEndpoint(id), TagEndpoint(), TagsEndpoint(tags)))

	r, err := self.Send(o...)
	if err != nil {
		return err
	}

	defer r.Body.Close()

	if r.StatusCode > 399 {
		return self.parseErrorResponse(r)
	}

	return nil
}

// Fetch metric definition tags
func (self *Client) Tags(t MetricType, id string, o ...Modifier) (map[string]string, error) {
	o = prepend(o, self.Url("GET", TypeEndpoint(t), SingleMetricEndpoint(id), TagEndpoint()))

	r, err := self.Send(o...)
	if err != nil {
		return nil, err
	}

	defer r.Body.Close()

	if r.StatusCode == http.StatusOK {
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		tags := make(map[string]string)
		if b != nil {
			if err = json.Unmarshal(b, &tags); err != nil {
				return nil, err
			}
		}
		return tags, nil
	} else if r.StatusCode > 399 {
		return nil, self.parseErrorResponse(r)
	}

	return nil, nil
}

// Write datapoints to the server
func (self *Client) Write(metrics []MetricHeader, o ...Modifier) error {
	if len(metrics) > 0 {
		mHs := make(map[MetricType][]MetricHeader)
		for _, m := range metrics {
			if _, found := mHs[m.Type]; !found {
				mHs[m.Type] = make([]MetricHeader, 0, 1)
			}
			mHs[m.Type] = append(mHs[m.Type], m)
		}

		wg := &sync.WaitGroup{}
		errorsChan := make(chan error, len(mHs))

		for k, v := range mHs {
			wg.Add(1)
			go func(k MetricType, v []MetricHeader) {
				defer wg.Done()

				// Should be sorted and splitted by type & tenant..
				on := o
				on = prepend(on, self.Url("POST", TypeEndpoint(k), DataEndpoint()), Data(v))

				r, err := self.Send(on...)
				if err != nil {
					errorsChan <- err
					return
				}

				defer r.Body.Close()

				if r.StatusCode > 399 {
					errorsChan <- self.parseErrorResponse(r)
				}
			}(k, v)
		}
		wg.Wait()
		select {
		case err, ok := <-errorsChan:
			if ok {
				return err
			}
			// If channel is closed, we're done
		default:
			// Nothing to do
		}

	}
	return nil
}

// Read data from the server
func (self *Client) ReadMetric(t MetricType, id string, o ...Modifier) ([]*Datapoint, error) {
	o = prepend(o, self.Url("GET", TypeEndpoint(t), SingleMetricEndpoint(id), DataEndpoint()))

	r, err := self.Send(o...)
	if err != nil {
		return nil, err
	}

	defer r.Body.Close()

	if r.StatusCode == http.StatusOK {
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}

		// Check for GaugeBucketpoint and so on for the rest.. uh
		dp := []*Datapoint{}
		if b != nil {
			if err = json.Unmarshal(b, &dp); err != nil {
				return nil, err
			}
		}
		return dp, nil
	} else if r.StatusCode > 399 {
		return nil, self.parseErrorResponse(r)
	}

	return nil, nil
}

// TODO ReadMetrics should be equal also, to read new tagsFilter aggregation..
func (self *Client) ReadBuckets(t MetricType, o ...Modifier) ([]*Bucketpoint, error) {
	o = prepend(o, self.Url("GET", TypeEndpoint(t), DataEndpoint()))

	r, err := self.Send(o...)
	if err != nil {
		return nil, err
	}

	defer r.Body.Close()

	if r.StatusCode == http.StatusOK {
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}

		// Check for GaugeBucketpoint and so on for the rest.. uh
		bp := []*Bucketpoint{}
		if b != nil {
			if err = json.Unmarshal(b, &bp); err != nil {
				return nil, err
			}
		}
		return bp, nil
	} else if r.StatusCode > 399 {
		return nil, self.parseErrorResponse(r)
	}

	return nil, nil
}

// Initialization

func NewHawkularClient(p Parameters) (*Client, error) {
	uri, err := url.Parse(p.Url)
	if err != nil {
		return nil, err
	}

	if uri.Path == "" {
		uri.Path = base_url
	}

	u := &url.URL{
		Host:   uri.Host,
		Path:   uri.Path,
		Scheme: uri.Scheme,
		Opaque: fmt.Sprintf("//%s/%s", uri.Host, uri.Path),
	}

	c := &http.Client{
		Timeout: timeout,
	}
	if p.TLSConfig != nil {
		transport := &http.Transport{TLSClientConfig: p.TLSConfig}
		c.Transport = transport
	}

	return &Client{
		url:    u,
		Tenant: p.Tenant,
		Token:  p.Token,
		client: c,
	}, nil
}

// HTTP Helper functions

func cleanId(id string) string {
	return url.QueryEscape(id)
}

func (self *Client) parseErrorResponse(resp *http.Response) error {
	// Parse error messages here correctly..
	reply, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return &HawkularClientError{Code: resp.StatusCode,
			msg: fmt.Sprintf("Reply could not be read: %s", err.Error()),
		}
	}

	details := &HawkularError{}

	err = json.Unmarshal(reply, details)
	if err != nil {
		return &HawkularClientError{Code: resp.StatusCode,
			msg: fmt.Sprintf("Reply could not be parsed: %s", err.Error()),
		}
	}

	return &HawkularClientError{Code: resp.StatusCode,
		msg: details.ErrorMsg,
	}
}

// URL functions (...)

type Endpoint func(u *url.URL)

func (self *Client) createUrl(e ...Endpoint) *url.URL {
	mu := *self.url
	for _, f := range e {
		f(&mu)
	}
	return &mu
}

func TypeEndpoint(t MetricType) Endpoint {
	return func(u *url.URL) {
		addToUrl(u, t.String())
	}
}

func SingleMetricEndpoint(id string) Endpoint {
	return func(u *url.URL) {
		addToUrl(u, url.QueryEscape(id))
	}
}

func TagEndpoint() Endpoint {
	return func(u *url.URL) {
		addToUrl(u, "tags")
	}
}

func TagsEndpoint(tags map[string]string) Endpoint {
	return func(u *url.URL) {
		addToUrl(u, tagsEncoder(tags))
	}
}

func DataEndpoint() Endpoint {
	return func(u *url.URL) {
		addToUrl(u, "data")
	}
}

func addToUrl(u *url.URL, s string) *url.URL {
	u.Opaque = fmt.Sprintf("%s/%s", u.Opaque, s)
	return u
}

func tagsEncoder(t map[string]string) string {
	tags := make([]string, 0, len(t))
	for k, v := range t {
		tags = append(tags, fmt.Sprintf("%s:%s", k, v))
	}
	j := strings.Join(tags, ",")
	return j
}
