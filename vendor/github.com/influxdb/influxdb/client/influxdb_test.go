package client_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/client"
)

func BenchmarkUnmarshalJSON2Tags(b *testing.B) {
	var bp client.BatchPoints
	data := []byte(`
{
    "database": "foo",
    "retentionPolicy": "bar",
    "points": [
        {
            "name": "cpu",
            "tags": {
                "host": "server01",
                "region": "us-east1"
            },
            "time": 14244733039069373,
            "precision": "n",
            "fields": {
                    "value": 4541770385657154000
            }
        }
    ]
}
`)

	for i := 0; i < b.N; i++ {
		if err := json.Unmarshal(data, &bp); err != nil {
			b.Errorf("unable to unmarshal nanosecond data: %s", err.Error())
		}
		b.SetBytes(int64(len(data)))
	}
}

func BenchmarkUnmarshalJSON10Tags(b *testing.B) {
	var bp client.BatchPoints
	data := []byte(`
{
    "database": "foo",
    "retentionPolicy": "bar",
    "points": [
        {
            "name": "cpu",
            "tags": {
                "host": "server01",
                "region": "us-east1",
                "tag1": "value1",
                "tag2": "value2",
                "tag2": "value3",
                "tag4": "value4",
                "tag5": "value5",
                "tag6": "value6",
                "tag7": "value7",
                "tag8": "value8"
            },
            "time": 14244733039069373,
            "precision": "n",
            "fields": {
                    "value": 4541770385657154000
            }
        }
    ]
}
`)

	for i := 0; i < b.N; i++ {
		if err := json.Unmarshal(data, &bp); err != nil {
			b.Errorf("unable to unmarshal nanosecond data: %s", err.Error())
		}
		b.SetBytes(int64(len(data)))
	}
}

func TestNewClient(t *testing.T) {
	config := client.Config{}
	_, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
}

func TestClient_Ping(t *testing.T) {
	ts := emptyTestServer()
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
	d, version, err := c.Ping()
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
	if d == 0 {
		t.Fatalf("expected a duration greater than zero.  actual %v", d)
	}
	if version != "x.x" {
		t.Fatalf("unexpected version.  expected %s,  actual %v", "x.x", version)
	}
}

func TestClient_Query(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var data client.Response
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(data)
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	query := client.Query{}
	_, err = c.Query(query)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
}

func TestClient_BasicAuth(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u, p, ok := r.BasicAuth()

		if !ok {
			t.Errorf("basic auth error")
		}
		if u != "username" {
			t.Errorf("unexpected username, expected %q, actual %q", "username", u)
		}
		if p != "password" {
			t.Errorf("unexpected password, expected %q, actual %q", "password", p)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	u.User = url.UserPassword("username", "password")
	config := client.Config{URL: *u, Username: "username", Password: "password"}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	_, _, err = c.Ping()
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
}

func TestClient_Write(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var data client.Response
		w.WriteHeader(http.StatusNoContent)
		_ = json.NewEncoder(w).Encode(data)
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	bp := client.BatchPoints{}
	r, err := c.Write(bp)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
	if r != nil {
		t.Fatalf("unexpected response. expected %v, actual %v", nil, r)
	}
}

func TestClient_UserAgent(t *testing.T) {
	receivedUserAgent := ""
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedUserAgent = r.UserAgent()

		var data client.Response
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(data)
	}))
	defer ts.Close()

	_, err := http.Get(ts.URL)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	tests := []struct {
		name      string
		userAgent string
		expected  string
	}{
		{
			name:      "Empty user agent",
			userAgent: "",
			expected:  "InfluxDBClient",
		},
		{
			name:      "Custom user agent",
			userAgent: "Test Influx Client",
			expected:  "Test Influx Client",
		},
	}

	for _, test := range tests {
		u, _ := url.Parse(ts.URL)
		config := client.Config{URL: *u, UserAgent: test.userAgent}
		c, err := client.NewClient(config)
		if err != nil {
			t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
		}

		receivedUserAgent = ""
		query := client.Query{}
		_, err = c.Query(query)
		if err != nil {
			t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
		}
		if !strings.HasPrefix(receivedUserAgent, test.expected) {
			t.Fatalf("Unexpected user agent. expected %v, actual %v", test.expected, receivedUserAgent)
		}

		receivedUserAgent = ""
		bp := client.BatchPoints{}
		_, err = c.Write(bp)
		if err != nil {
			t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
		}
		if !strings.HasPrefix(receivedUserAgent, test.expected) {
			t.Fatalf("Unexpected user agent. expected %v, actual %v", test.expected, receivedUserAgent)
		}

		receivedUserAgent = ""
		_, _, err = c.Ping()
		if err != nil {
			t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
		}
		if receivedUserAgent != test.expected {
			t.Fatalf("Unexpected user agent. expected %v, actual %v", test.expected, receivedUserAgent)
		}
	}
}

func TestPoint_UnmarshalEpoch(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name      string
		epoch     int64
		precision string
		expected  time.Time
	}{
		{
			name:      "nanoseconds",
			epoch:     now.UnixNano(),
			precision: "n",
			expected:  now,
		},
		{
			name:      "microseconds",
			epoch:     now.Round(time.Microsecond).UnixNano() / int64(time.Microsecond),
			precision: "u",
			expected:  now.Round(time.Microsecond),
		},
		{
			name:      "milliseconds",
			epoch:     now.Round(time.Millisecond).UnixNano() / int64(time.Millisecond),
			precision: "ms",
			expected:  now.Round(time.Millisecond),
		},
		{
			name:      "seconds",
			epoch:     now.Round(time.Second).UnixNano() / int64(time.Second),
			precision: "s",
			expected:  now.Round(time.Second),
		},
		{
			name:      "minutes",
			epoch:     now.Round(time.Minute).UnixNano() / int64(time.Minute),
			precision: "m",
			expected:  now.Round(time.Minute),
		},
		{
			name:      "hours",
			epoch:     now.Round(time.Hour).UnixNano() / int64(time.Hour),
			precision: "h",
			expected:  now.Round(time.Hour),
		},
		{
			name:      "max int64",
			epoch:     9223372036854775807,
			precision: "n",
			expected:  time.Unix(0, 9223372036854775807),
		},
		{
			name:      "100 years from now",
			epoch:     now.Add(time.Hour * 24 * 365 * 100).UnixNano(),
			precision: "n",
			expected:  now.Add(time.Hour * 24 * 365 * 100),
		},
	}

	for _, test := range tests {
		t.Logf("testing %q\n", test.name)
		data := []byte(fmt.Sprintf(`{"time": %d, "precision":"%s"}`, test.epoch, test.precision))
		t.Logf("json: %s", string(data))
		var p client.Point
		err := json.Unmarshal(data, &p)
		if err != nil {
			t.Fatalf("unexpected error.  exptected: %v, actual: %v", nil, err)
		}
		if !p.Time.Equal(test.expected) {
			t.Fatalf("Unexpected time.  expected: %v, actual: %v", test.expected, p.Time)
		}
	}
}

func TestPoint_UnmarshalRFC(t *testing.T) {
	now := time.Now().UTC()
	tests := []struct {
		name     string
		rfc      string
		now      time.Time
		expected time.Time
	}{
		{
			name:     "RFC3339Nano",
			rfc:      time.RFC3339Nano,
			now:      now,
			expected: now,
		},
		{
			name:     "RFC3339",
			rfc:      time.RFC3339,
			now:      now.Round(time.Second),
			expected: now.Round(time.Second),
		},
	}

	for _, test := range tests {
		t.Logf("testing %q\n", test.name)
		ts := test.now.Format(test.rfc)
		data := []byte(fmt.Sprintf(`{"time": %q}`, ts))
		t.Logf("json: %s", string(data))
		var p client.Point
		err := json.Unmarshal(data, &p)
		if err != nil {
			t.Fatalf("unexpected error.  exptected: %v, actual: %v", nil, err)
		}
		if !p.Time.Equal(test.expected) {
			t.Fatalf("Unexpected time.  expected: %v, actual: %v", test.expected, p.Time)
		}
	}
}

func TestPoint_MarshalOmitempty(t *testing.T) {
	now := time.Now().UTC()
	tests := []struct {
		name     string
		point    client.Point
		now      time.Time
		expected string
	}{
		{
			name:     "all empty",
			point:    client.Point{Measurement: "cpu", Fields: map[string]interface{}{"value": 1.1}},
			now:      now,
			expected: `{"measurement":"cpu","fields":{"value":1.1}}`,
		},
		{
			name:     "with time",
			point:    client.Point{Measurement: "cpu", Fields: map[string]interface{}{"value": 1.1}, Time: now},
			now:      now,
			expected: fmt.Sprintf(`{"measurement":"cpu","time":"%s","fields":{"value":1.1}}`, now.Format(time.RFC3339Nano)),
		},
		{
			name:     "with tags",
			point:    client.Point{Measurement: "cpu", Fields: map[string]interface{}{"value": 1.1}, Tags: map[string]string{"foo": "bar"}},
			now:      now,
			expected: `{"measurement":"cpu","tags":{"foo":"bar"},"fields":{"value":1.1}}`,
		},
		{
			name:     "with precision",
			point:    client.Point{Measurement: "cpu", Fields: map[string]interface{}{"value": 1.1}, Precision: "ms"},
			now:      now,
			expected: `{"measurement":"cpu","fields":{"value":1.1},"precision":"ms"}`,
		},
	}

	for _, test := range tests {
		t.Logf("testing %q\n", test.name)
		b, err := json.Marshal(&test.point)
		if err != nil {
			t.Fatalf("unexpected error.  exptected: %v, actual: %v", nil, err)
		}
		if test.expected != string(b) {
			t.Fatalf("Unexpected result.  expected: %v, actual: %v", test.expected, string(b))
		}
	}
}

func TestEpochToTime(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name      string
		epoch     int64
		precision string
		expected  time.Time
	}{
		{name: "nanoseconds", epoch: now.UnixNano(), precision: "n", expected: now},
		{name: "microseconds", epoch: now.Round(time.Microsecond).UnixNano() / int64(time.Microsecond), precision: "u", expected: now.Round(time.Microsecond)},
		{name: "milliseconds", epoch: now.Round(time.Millisecond).UnixNano() / int64(time.Millisecond), precision: "ms", expected: now.Round(time.Millisecond)},
		{name: "seconds", epoch: now.Round(time.Second).UnixNano() / int64(time.Second), precision: "s", expected: now.Round(time.Second)},
		{name: "minutes", epoch: now.Round(time.Minute).UnixNano() / int64(time.Minute), precision: "m", expected: now.Round(time.Minute)},
		{name: "hours", epoch: now.Round(time.Hour).UnixNano() / int64(time.Hour), precision: "h", expected: now.Round(time.Hour)},
	}

	for _, test := range tests {
		t.Logf("testing %q\n", test.name)
		tm, e := client.EpochToTime(test.epoch, test.precision)
		if e != nil {
			t.Fatalf("unexpected error: expected %v, actual: %v", nil, e)
		}
		if tm != test.expected {
			t.Fatalf("unexpected time: expected %v, actual %v", test.expected, tm)
		}
	}
}

// helper functions

func emptyTestServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Influxdb-Version", "x.x")
		return
	}))
}

// Ensure that data with epoch times can be decoded.
func TestBatchPoints_Normal(t *testing.T) {
	var bp client.BatchPoints
	data := []byte(`
{
    "database": "foo",
    "retentionPolicy": "bar",
    "points": [
        {
            "name": "cpu",
            "tags": {
                "host": "server01"
            },
            "time": 14244733039069373,
            "precision": "n",
            "values": {
                    "value": 4541770385657154000
            }
        },
        {
            "name": "cpu",
             "tags": {
                "host": "server01"
            },
            "time": 14244733039069380,
            "precision": "n",
            "values": {
                    "value": 7199311900554737000
            }
        }
    ]
}
`)

	if err := json.Unmarshal(data, &bp); err != nil {
		t.Errorf("unable to unmarshal nanosecond data: %s", err.Error())
	}
}

func TestClient_Timeout(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(1 * time.Second)
		var data client.Response
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(data)
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u, Timeout: 500 * time.Millisecond}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	query := client.Query{}
	_, err = c.Query(query)
	if err == nil {
		t.Fatalf("unexpected success.  expected timeout error")
	} else if !strings.Contains(err.Error(), "use of closed network connection") {
		t.Fatalf("unexpected error.  expected 'use of closed network connection' error, got %v", err)
	}

	confignotimeout := client.Config{URL: *u}
	cnotimeout, err := client.NewClient(confignotimeout)
	_, err = cnotimeout.Query(query)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}
}
