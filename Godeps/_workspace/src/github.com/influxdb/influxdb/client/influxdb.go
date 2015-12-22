package client

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/influxdb/influxdb/models"
)

const (
	// DefaultHost is the default host used to connect to an InfluxDB instance
	DefaultHost = "localhost"

	// DefaultPort is the default port used to connect to an InfluxDB instance
	DefaultPort = 8086

	// DefaultTimeout is the default connection timeout used to connect to an InfluxDB instance
	DefaultTimeout = 0
)

// Query is used to send a command to the server. Both Command and Database are required.
type Query struct {
	Command  string
	Database string
}

// ParseConnectionString will parse a string to create a valid connection URL
func ParseConnectionString(path string, ssl bool) (url.URL, error) {
	var host string
	var port int

	h, p, err := net.SplitHostPort(path)
	if err != nil {
		if path == "" {
			host = DefaultHost
		} else {
			host = path
		}
		// If they didn't specify a port, always use the default port
		port = DefaultPort
	} else {
		host = h
		port, err = strconv.Atoi(p)
		if err != nil {
			return url.URL{}, fmt.Errorf("invalid port number %q: %s\n", path, err)
		}
	}

	u := url.URL{
		Scheme: "http",
	}
	if ssl {
		u.Scheme = "https"
	}

	u.Host = net.JoinHostPort(host, strconv.Itoa(port))

	return u, nil
}

// Config is used to specify what server to connect to.
// URL: The URL of the server connecting to.
// Username/Password are optional. They will be passed via basic auth if provided.
// UserAgent: If not provided, will default "InfluxDBClient",
// Timeout: If not provided, will default to 0 (no timeout)
type Config struct {
	URL       url.URL
	Username  string
	Password  string
	UserAgent string
	Timeout   time.Duration
	Precision string
}

// NewConfig will create a config to be used in connecting to the client
func NewConfig() Config {
	return Config{
		Timeout: DefaultTimeout,
	}
}

// Client is used to make calls to the server.
type Client struct {
	url        url.URL
	username   string
	password   string
	httpClient *http.Client
	userAgent  string
	precision  string
}

const (
	// ConsistencyOne requires at least one data node acknowledged a write.
	ConsistencyOne = "one"

	// ConsistencyAll requires all data nodes to acknowledge a write.
	ConsistencyAll = "all"

	// ConsistencyQuorum requires a quorum of data nodes to acknowledge a write.
	ConsistencyQuorum = "quorum"

	// ConsistencyAny allows for hinted hand off, potentially no write happened yet.
	ConsistencyAny = "any"
)

// NewClient will instantiate and return a connected client to issue commands to the server.
func NewClient(c Config) (*Client, error) {
	client := Client{
		url:        c.URL,
		username:   c.Username,
		password:   c.Password,
		httpClient: &http.Client{Timeout: c.Timeout},
		userAgent:  c.UserAgent,
		precision:  c.Precision,
	}
	if client.userAgent == "" {
		client.userAgent = "InfluxDBClient"
	}
	return &client, nil
}

// SetAuth will update the username and passwords
func (c *Client) SetAuth(u, p string) {
	c.username = u
	c.password = p
}

// SetPrecision will update the precision
func (c *Client) SetPrecision(precision string) {
	c.precision = precision
}

// Query sends a command to the server and returns the Response
func (c *Client) Query(q Query) (*Response, error) {
	u := c.url

	u.Path = "query"
	values := u.Query()
	values.Set("q", q.Command)
	values.Set("db", q.Database)
	if c.precision != "" {
		values.Set("epoch", c.precision)
	}
	u.RawQuery = values.Encode()

	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", c.userAgent)
	if c.username != "" {
		req.SetBasicAuth(c.username, c.password)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var response Response
	dec := json.NewDecoder(resp.Body)
	dec.UseNumber()
	decErr := dec.Decode(&response)

	// ignore this error if we got an invalid status code
	if decErr != nil && decErr.Error() == "EOF" && resp.StatusCode != http.StatusOK {
		decErr = nil
	}
	// If we got a valid decode error, send that back
	if decErr != nil {
		return nil, decErr
	}
	// If we don't have an error in our json response, and didn't get StatusOK, then send back an error
	if resp.StatusCode != http.StatusOK && response.Error() == nil {
		return &response, fmt.Errorf("received status code %d from server", resp.StatusCode)
	}
	return &response, nil
}

// Write takes BatchPoints and allows for writing of multiple points with defaults
// If successful, error is nil and Response is nil
// If an error occurs, Response may contain additional information if populated.
func (c *Client) Write(bp BatchPoints) (*Response, error) {
	u := c.url
	u.Path = "write"

	var b bytes.Buffer
	for _, p := range bp.Points {
		if p.Raw != "" {
			if _, err := b.WriteString(p.Raw); err != nil {
				return nil, err
			}
		} else {
			for k, v := range bp.Tags {
				if p.Tags == nil {
					p.Tags = make(map[string]string, len(bp.Tags))
				}
				p.Tags[k] = v
			}

			if _, err := b.WriteString(p.MarshalString()); err != nil {
				return nil, err
			}
		}

		if err := b.WriteByte('\n'); err != nil {
			return nil, err
		}
	}

	req, err := http.NewRequest("POST", u.String(), &b)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "")
	req.Header.Set("User-Agent", c.userAgent)
	if c.username != "" {
		req.SetBasicAuth(c.username, c.password)
	}

	precision := bp.Precision
	if precision == "" {
		precision = c.precision
	}

	params := req.URL.Query()
	params.Set("db", bp.Database)
	params.Set("rp", bp.RetentionPolicy)
	params.Set("precision", precision)
	params.Set("consistency", bp.WriteConsistency)
	req.URL.RawQuery = params.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var response Response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		var err = fmt.Errorf(string(body))
		response.Err = err
		return &response, err
	}

	return nil, nil
}

// WriteLineProtocol takes a string with line returns to delimit each write
// If successful, error is nil and Response is nil
// If an error occurs, Response may contain additional information if populated.
func (c *Client) WriteLineProtocol(data, database, retentionPolicy, precision, writeConsistency string) (*Response, error) {
	u := c.url
	u.Path = "write"

	r := strings.NewReader(data)

	req, err := http.NewRequest("POST", u.String(), r)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "")
	req.Header.Set("User-Agent", c.userAgent)
	if c.username != "" {
		req.SetBasicAuth(c.username, c.password)
	}
	params := req.URL.Query()
	params.Set("db", database)
	params.Set("rp", retentionPolicy)
	params.Set("precision", precision)
	params.Set("consistency", writeConsistency)
	req.URL.RawQuery = params.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var response Response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		err := fmt.Errorf(string(body))
		response.Err = err
		return &response, err
	}

	return nil, nil
}

// Ping will check to see if the server is up
// Ping returns how long the request took, the version of the server it connected to, and an error if one occurred.
func (c *Client) Ping() (time.Duration, string, error) {
	now := time.Now()
	u := c.url
	u.Path = "ping"

	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("User-Agent", c.userAgent)
	if c.username != "" {
		req.SetBasicAuth(c.username, c.password)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()

	version := resp.Header.Get("X-Influxdb-Version")
	return time.Since(now), version, nil
}

// Structs

// Result represents a resultset returned from a single statement.
type Result struct {
	Series []models.Row
	Err    error
}

// MarshalJSON encodes the result into JSON.
func (r *Result) MarshalJSON() ([]byte, error) {
	// Define a struct that outputs "error" as a string.
	var o struct {
		Series []models.Row `json:"series,omitempty"`
		Err    string       `json:"error,omitempty"`
	}

	// Copy fields to output struct.
	o.Series = r.Series
	if r.Err != nil {
		o.Err = r.Err.Error()
	}

	return json.Marshal(&o)
}

// UnmarshalJSON decodes the data into the Result struct
func (r *Result) UnmarshalJSON(b []byte) error {
	var o struct {
		Series []models.Row `json:"series,omitempty"`
		Err    string       `json:"error,omitempty"`
	}

	dec := json.NewDecoder(bytes.NewBuffer(b))
	dec.UseNumber()
	err := dec.Decode(&o)
	if err != nil {
		return err
	}
	r.Series = o.Series
	if o.Err != "" {
		r.Err = errors.New(o.Err)
	}
	return nil
}

// Response represents a list of statement results.
type Response struct {
	Results []Result
	Err     error
}

// MarshalJSON encodes the response into JSON.
func (r *Response) MarshalJSON() ([]byte, error) {
	// Define a struct that outputs "error" as a string.
	var o struct {
		Results []Result `json:"results,omitempty"`
		Err     string   `json:"error,omitempty"`
	}

	// Copy fields to output struct.
	o.Results = r.Results
	if r.Err != nil {
		o.Err = r.Err.Error()
	}

	return json.Marshal(&o)
}

// UnmarshalJSON decodes the data into the Response struct
func (r *Response) UnmarshalJSON(b []byte) error {
	var o struct {
		Results []Result `json:"results,omitempty"`
		Err     string   `json:"error,omitempty"`
	}

	dec := json.NewDecoder(bytes.NewBuffer(b))
	dec.UseNumber()
	err := dec.Decode(&o)
	if err != nil {
		return err
	}
	r.Results = o.Results
	if o.Err != "" {
		r.Err = errors.New(o.Err)
	}
	return nil
}

// Error returns the first error from any statement.
// Returns nil if no errors occurred on any statements.
func (r Response) Error() error {
	if r.Err != nil {
		return r.Err
	}
	for _, result := range r.Results {
		if result.Err != nil {
			return result.Err
		}
	}
	return nil
}

// Point defines the fields that will be written to the database
// Measurement, Time, and Fields are required
// Precision can be specified if the time is in epoch format (integer).
// Valid values for Precision are n, u, ms, s, m, and h
type Point struct {
	Measurement string
	Tags        map[string]string
	Time        time.Time
	Fields      map[string]interface{}
	Precision   string
	Raw         string
}

// MarshalJSON will format the time in RFC3339Nano
// Precision is also ignored as it is only used for writing, not reading
// Or another way to say it is we always send back in nanosecond precision
func (p *Point) MarshalJSON() ([]byte, error) {
	point := struct {
		Measurement string                 `json:"measurement,omitempty"`
		Tags        map[string]string      `json:"tags,omitempty"`
		Time        string                 `json:"time,omitempty"`
		Fields      map[string]interface{} `json:"fields,omitempty"`
		Precision   string                 `json:"precision,omitempty"`
	}{
		Measurement: p.Measurement,
		Tags:        p.Tags,
		Fields:      p.Fields,
		Precision:   p.Precision,
	}
	// Let it omit empty if it's really zero
	if !p.Time.IsZero() {
		point.Time = p.Time.UTC().Format(time.RFC3339Nano)
	}
	return json.Marshal(&point)
}

// MarshalString renders string representation of a Point with specified
// precision. The default precision is nanoseconds.
func (p *Point) MarshalString() string {
	pt, err := models.NewPoint(p.Measurement, p.Tags, p.Fields, p.Time)
	if err != nil {
		return "# ERROR: " + err.Error() + " " + p.Measurement
	}
	if p.Precision == "" || p.Precision == "ns" || p.Precision == "n" {
		return pt.String()
	}
	return pt.PrecisionString(p.Precision)
}

// UnmarshalJSON decodes the data into the Point struct
func (p *Point) UnmarshalJSON(b []byte) error {
	var normal struct {
		Measurement string                 `json:"measurement"`
		Tags        map[string]string      `json:"tags"`
		Time        time.Time              `json:"time"`
		Precision   string                 `json:"precision"`
		Fields      map[string]interface{} `json:"fields"`
	}
	var epoch struct {
		Measurement string                 `json:"measurement"`
		Tags        map[string]string      `json:"tags"`
		Time        *int64                 `json:"time"`
		Precision   string                 `json:"precision"`
		Fields      map[string]interface{} `json:"fields"`
	}

	if err := func() error {
		var err error
		dec := json.NewDecoder(bytes.NewBuffer(b))
		dec.UseNumber()
		if err = dec.Decode(&epoch); err != nil {
			return err
		}
		// Convert from epoch to time.Time, but only if Time
		// was actually set.
		var ts time.Time
		if epoch.Time != nil {
			ts, err = EpochToTime(*epoch.Time, epoch.Precision)
			if err != nil {
				return err
			}
		}
		p.Measurement = epoch.Measurement
		p.Tags = epoch.Tags
		p.Time = ts
		p.Precision = epoch.Precision
		p.Fields = normalizeFields(epoch.Fields)
		return nil
	}(); err == nil {
		return nil
	}

	dec := json.NewDecoder(bytes.NewBuffer(b))
	dec.UseNumber()
	if err := dec.Decode(&normal); err != nil {
		return err
	}
	normal.Time = SetPrecision(normal.Time, normal.Precision)
	p.Measurement = normal.Measurement
	p.Tags = normal.Tags
	p.Time = normal.Time
	p.Precision = normal.Precision
	p.Fields = normalizeFields(normal.Fields)

	return nil
}

// Remove any notion of json.Number
func normalizeFields(fields map[string]interface{}) map[string]interface{} {
	newFields := map[string]interface{}{}

	for k, v := range fields {
		switch v := v.(type) {
		case json.Number:
			jv, e := v.Float64()
			if e != nil {
				panic(fmt.Sprintf("unable to convert json.Number to float64: %s", e))
			}
			newFields[k] = jv
		default:
			newFields[k] = v
		}
	}
	return newFields
}

// BatchPoints is used to send batched data in a single write.
// Database and Points are required
// If no retention policy is specified, it will use the databases default retention policy.
// If tags are specified, they will be "merged" with all points. If a point already has that tag, it will be ignored.
// If time is specified, it will be applied to any point with an empty time.
// Precision can be specified if the time is in epoch format (integer).
// Valid values for Precision are n, u, ms, s, m, and h
type BatchPoints struct {
	Points           []Point           `json:"points,omitempty"`
	Database         string            `json:"database,omitempty"`
	RetentionPolicy  string            `json:"retentionPolicy,omitempty"`
	Tags             map[string]string `json:"tags,omitempty"`
	Time             time.Time         `json:"time,omitempty"`
	Precision        string            `json:"precision,omitempty"`
	WriteConsistency string            `json:"-"`
}

// UnmarshalJSON decodes the data into the BatchPoints struct
func (bp *BatchPoints) UnmarshalJSON(b []byte) error {
	var normal struct {
		Points          []Point           `json:"points"`
		Database        string            `json:"database"`
		RetentionPolicy string            `json:"retentionPolicy"`
		Tags            map[string]string `json:"tags"`
		Time            time.Time         `json:"time"`
		Precision       string            `json:"precision"`
	}
	var epoch struct {
		Points          []Point           `json:"points"`
		Database        string            `json:"database"`
		RetentionPolicy string            `json:"retentionPolicy"`
		Tags            map[string]string `json:"tags"`
		Time            *int64            `json:"time"`
		Precision       string            `json:"precision"`
	}

	if err := func() error {
		var err error
		if err = json.Unmarshal(b, &epoch); err != nil {
			return err
		}
		// Convert from epoch to time.Time
		var ts time.Time
		if epoch.Time != nil {
			ts, err = EpochToTime(*epoch.Time, epoch.Precision)
			if err != nil {
				return err
			}
		}
		bp.Points = epoch.Points
		bp.Database = epoch.Database
		bp.RetentionPolicy = epoch.RetentionPolicy
		bp.Tags = epoch.Tags
		bp.Time = ts
		bp.Precision = epoch.Precision
		return nil
	}(); err == nil {
		return nil
	}

	if err := json.Unmarshal(b, &normal); err != nil {
		return err
	}
	normal.Time = SetPrecision(normal.Time, normal.Precision)
	bp.Points = normal.Points
	bp.Database = normal.Database
	bp.RetentionPolicy = normal.RetentionPolicy
	bp.Tags = normal.Tags
	bp.Time = normal.Time
	bp.Precision = normal.Precision

	return nil
}

// utility functions

// Addr provides the current url as a string of the server the client is connected to.
func (c *Client) Addr() string {
	return c.url.String()
}

// helper functions

// EpochToTime takes a unix epoch time and uses precision to return back a time.Time
func EpochToTime(epoch int64, precision string) (time.Time, error) {
	if precision == "" {
		precision = "s"
	}
	var t time.Time
	switch precision {
	case "h":
		t = time.Unix(0, epoch*int64(time.Hour))
	case "m":
		t = time.Unix(0, epoch*int64(time.Minute))
	case "s":
		t = time.Unix(0, epoch*int64(time.Second))
	case "ms":
		t = time.Unix(0, epoch*int64(time.Millisecond))
	case "u":
		t = time.Unix(0, epoch*int64(time.Microsecond))
	case "n":
		t = time.Unix(0, epoch)
	default:
		return time.Time{}, fmt.Errorf("Unknown precision %q", precision)
	}
	return t, nil
}

// SetPrecision will round a time to the specified precision
func SetPrecision(t time.Time, precision string) time.Time {
	switch precision {
	case "n":
	case "u":
		return t.Round(time.Microsecond)
	case "ms":
		return t.Round(time.Millisecond)
	case "s":
		return t.Round(time.Second)
	case "m":
		return t.Round(time.Minute)
	case "h":
		return t.Round(time.Hour)
	}
	return t
}
