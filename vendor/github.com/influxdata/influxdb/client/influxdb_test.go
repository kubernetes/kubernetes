package client_test

import (
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/influxdata/influxdb/client"
)

func BenchmarkWrite(b *testing.B) {
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
		b.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	bp := client.BatchPoints{
		Points: []client.Point{
			{Fields: map[string]interface{}{"value": 101}}},
	}
	for i := 0; i < b.N; i++ {
		r, err := c.Write(bp)
		if err != nil {
			b.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
		}
		if r != nil {
			b.Fatalf("unexpected response. expected %v, actual %v", nil, r)
		}
	}
}

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
	if d.Nanoseconds() == 0 {
		t.Fatalf("expected a duration greater than zero.  actual %v", d.Nanoseconds())
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

func TestClient_ChunkedQuery(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var data client.Response
		w.WriteHeader(http.StatusOK)
		enc := json.NewEncoder(w)
		_ = enc.Encode(data)
		_ = enc.Encode(data)
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	query := client.Query{Chunked: true}
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

func TestClient_Messages(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"results":[{"messages":[{"level":"warning","text":"deprecation test"}]}]}`))
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	query := client.Query{}
	resp, err := c.Query(query)
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	}

	if got, exp := len(resp.Results), 1; got != exp {
		t.Fatalf("unexpected number of results.  expected %v, actual %v", exp, got)
	}

	r := resp.Results[0]
	if got, exp := len(r.Messages), 1; got != exp {
		t.Fatalf("unexpected number of messages.  expected %v, actual %v", exp, got)
	}

	m := r.Messages[0]
	if got, exp := m.Level, "warning"; got != exp {
		t.Errorf("unexpected message level.  expected %v, actual %v", exp, got)
	}
	if got, exp := m.Text, "deprecation test"; got != exp {
		t.Errorf("unexpected message text.  expected %v, actual %v", exp, got)
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
		time.Sleep(50 * time.Millisecond)
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
	done := make(chan bool)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-done
	}))
	defer ts.Close()
	defer func() { done <- true }()
	u, _ := url.Parse(ts.URL)
	config := client.Config{URL: *u, Timeout: 500 * time.Millisecond}
	c, err := client.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error. expected %v, actual %v", nil, err)
	}
	query := client.Query{}
	_, err = c.Query(query)
	if err == nil {
		t.Fatalf("unexpected success. expected timeout error")
	} else if !strings.Contains(err.Error(), "request canceled") &&
		!strings.Contains(err.Error(), "use of closed network connection") {
		t.Fatalf("unexpected error. expected 'request canceled' error, got %v", err)
	}
}

func TestClient_NoTimeout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(1 * time.Second)
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

func TestClient_WriteUint64(t *testing.T) {
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
	bp := client.BatchPoints{
		Points: []client.Point{
			{
				Fields: map[string]interface{}{"value": uint64(10)},
			},
		},
	}
	r, err := c.Write(bp)
	if err == nil {
		t.Fatalf("unexpected error. expected err, actual %v", err)
	}
	if r != nil {
		t.Fatalf("unexpected response. expected %v, actual %v", nil, r)
	}
}

func TestClient_ParseConnectionString_IPv6(t *testing.T) {
	path := "[fdf5:9ede:1875:0:a9ee:a600:8fe3:d495]:8086"
	u, err := client.ParseConnectionString(path, false)
	if err != nil {
		t.Fatalf("unexpected error, expected %v, actual %v", nil, err)
	}
	if u.Host != path {
		t.Fatalf("ipv6 parse failed, expected %s, actual %s", path, u.Host)
	}
}

func TestClient_CustomCertificates(t *testing.T) {
	// generated with:
	// openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -config influx.cnf
	// influx.cnf:
	// [req]
	// distinguished_name = req_distinguished_name
	// x509_extensions = v3_req
	// prompt = no
	// [req_distinguished_name]
	// C = US
	// ST = CA
	// L = San Francisco
	// O = InfluxDB
	// CN = github.com/influxdata
	// [v3_req]
	// keyUsage = keyEncipherment, dataEncipherment
	// extendedKeyUsage = serverAuth
	// subjectAltName = @alt_names
	// [alt_names]
	// IP.1 = 127.0.0.1
	//
	key := `
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDLswqKJLxfhBRi
4qdj7+jpBxTAi4MewrcMPp+9YlbLke3F7w2DPrZVkYVeWmg8LyTPAigrXeadK6hv
qjRr05a7sMc5+ynivGbWUySWT+u17V85x6VR5TMIkJEOqpiIU8aYk0l+3UcrzVjS
1QZCUBoxVwAVaSR6AXTA8YrVXdk/AI3f22dYiBjFmV4LJJkGjTaCnlDKu54hMU1t
WTyFcoY9TBzZ1XA+ng5RQ/QADeL2PYrTW4s/mLI3jfKKD53EI4uM2FjW37ZfuxTa
mhCR7/lxM4COg9K70y5uebfqJvuoXAwXLOzVbdfF5b9fJFbL67kaK2tiMT3Wt39m
hXzclLTDAgMBAAECggEAK8mpElkjRUUXPMqMQSdpYe5rv5g973bb8n3jyMpC7i/I
dSwWM4hfmbVWfhnhHk7kErvb9raQxGiGJLrp2eP6Gw69RPGA54SodpoY21cCzHDi
b4FDQH+MoOKyy/xQHb4kitfejK70ha320huI5OhjOQgCtJeNh8yYVIGX3pX2BVyu
36UB9tfX1S5pbiHeih3vZGd322Muj/joNzIelnYRBnoO0xqvQ0S1Dk+dLCTHO0/m
u9AZN8c2TsRWZpJPMWwBv8LuABbE0e66/TSsrfklAn86ELCo44lZURDE7uPZ4pIH
FWtmf+nW5Hy6aPhy60E40MqotlejhWwB3ktY/m3JAQKBgQDuB4nhxzJA9lH9EaCt
byvJ9wGVvI3k79hOwc/Z2R3dNe+Ma+TJy+aBppvsLF4qz83aWC+canyasbHcPNR/
vXQGlsgKfucrmd1PfMV7uvOIkfOjK0E6mRC+jMuKtNTQrdtM1BU/Z7LY0iy0fNJ6
aNqhFdlJmmk0g+4bR4SAWB6FkwKBgQDbE/7r1u+GdJk/mhdjTi1aegr9lXb0l7L6
BCvOYhs/Z/pXfsaYPSXhgk2w+LiGk6BaEA2/4Sr0YS2MAAaIhBVeFBIXVpNrXB3K
Yg1jOEeLQ3qoVBeJFhJNrN9ZQx33HANC1W/Y1apMwaYqCRUGVQkrdcsN2KNea1z0
3qeYeCCSEQKBgCKZKeuNfrp+k1BLnaVYAW9r3ekb7SwXyMM53LJ3oqWiz10D2c+T
OcAirYtYr59dcTiJlPIRcGcz6PxwQxsGOLU0eYM9CvEFfmutYS8o73ksbdOL2AFi
elKYOIXC3yQuATBbq3L56b8mXaUmd5mfYBgGCv1t2ljtzFBext248UbNAoGBAIv1
2V24YiwnH6THf/ucfVMZNx5Mt8OJivk5YvcmLDw05HWzc5LdNe89PP871z963u3K
5c3ZP4UC9INFnOboY3JIJkqsr9/d6NZcECt8UBDDmoAhwSt+Y1EmiUZQn7s4NUkk
bKE919/Ts6GVTc5O013lkkUVS0HOG4QBH1dEH6LRAoGAStl11WA9tuKXiBl5XG/C
cq9mFPNJK3pEgd6YH874vEnYEEqENR4MFK3uWXus9Nm+VYxbUbPEzFF4kpsfukDg
/JAVqY4lUam7g6fyyaoIIPQEp7jGjbsUf46IjnUjFcaojOugA3EAfn9awREUDuJZ
cvh4WzEegcExTppINW1NB5E=
-----END PRIVATE KEY-----
`
	cert := `
-----BEGIN CERTIFICATE-----
MIIDdjCCAl6gAwIBAgIJAMYGAwkxUV51MA0GCSqGSIb3DQEBCwUAMFgxCzAJBgNV
BAYTAlVTMQswCQYDVQQIDAJDQTEWMBQGA1UEBwwNU2FuIEZyYW5jaXNjbzERMA8G
A1UECgwISW5mbHV4REIxETAPBgNVBAMMCGluZmx1eGRiMB4XDTE1MTIyOTAxNTg1
NloXDTI1MTIyNjAxNTg1NlowWDELMAkGA1UEBhMCVVMxCzAJBgNVBAgMAkNBMRYw
FAYDVQQHDA1TYW4gRnJhbmNpc2NvMREwDwYDVQQKDAhJbmZsdXhEQjERMA8GA1UE
AwwIaW5mbHV4ZGIwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDLswqK
JLxfhBRi4qdj7+jpBxTAi4MewrcMPp+9YlbLke3F7w2DPrZVkYVeWmg8LyTPAigr
XeadK6hvqjRr05a7sMc5+ynivGbWUySWT+u17V85x6VR5TMIkJEOqpiIU8aYk0l+
3UcrzVjS1QZCUBoxVwAVaSR6AXTA8YrVXdk/AI3f22dYiBjFmV4LJJkGjTaCnlDK
u54hMU1tWTyFcoY9TBzZ1XA+ng5RQ/QADeL2PYrTW4s/mLI3jfKKD53EI4uM2FjW
37ZfuxTamhCR7/lxM4COg9K70y5uebfqJvuoXAwXLOzVbdfF5b9fJFbL67kaK2ti
MT3Wt39mhXzclLTDAgMBAAGjQzBBMAwGA1UdEwQFMAMBAf8wCwYDVR0PBAQDAgQw
MBMGA1UdJQQMMAoGCCsGAQUFBwMBMA8GA1UdEQQIMAaHBH8AAAEwDQYJKoZIhvcN
AQELBQADggEBAJxgHeduV9q2BuKnrt+sjXLGn/HwbMbgGbgFK6kUKJBWtv6Pa7JJ
m4teDmTMWiaeB2g4N2bmaWTuEZzzShNKG5roFeWm1ilFMAyzkb+VifN4YuDKH62F
3e259qsytiGbbJF3F//4sjfMw8qZVEPvspG1zKsASo0PpSOOUFmxcj0oMAXhnMrk
rRcbk6fufhyq0iZGl8ZLKTCrkjk0b3qlNs6UaRD9/XBB59VlQ8I338sfjV06edwY
jn5Amab0uyoFNEp70Y4WGxrxUTS1GAC1LCA13S7EnidD440UrnWALTarjmHAK6aW
war3JNM1mGB3o2iAtuOJlFIKLpI1x+1e8pI=
-----END CERTIFICATE-----
`
	cer, err := tls.X509KeyPair([]byte(cert), []byte(key))

	if err != nil {
		t.Fatalf("Received error: %v", err)
	}

	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var data client.Response
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(data)
	}))
	server.TLS = &tls.Config{Certificates: []tls.Certificate{cer}}
	server.TLS.BuildNameToCertificate()
	server.StartTLS()
	defer server.Close()

	certFile, _ := ioutil.TempFile("", "influx-cert-")
	certFile.WriteString(cert)
	certFile.Close()
	defer os.Remove(certFile.Name())

	u, _ := url.Parse(server.URL)

	tests := []struct {
		name      string
		unsafeSsl bool
		expected  error
	}{
		{name: "validate certificates", unsafeSsl: false, expected: errors.New("error")},
		{name: "not validate certificates", unsafeSsl: true, expected: nil},
	}

	for _, test := range tests {
		config := client.Config{URL: *u, UnsafeSsl: test.unsafeSsl}
		c, err := client.NewClient(config)
		if err != nil {
			t.Fatalf("unexpected error. expected %v, actual %v", nil, err)
		}
		query := client.Query{}
		_, err = c.Query(query)

		if (test.expected == nil) != (err == nil) {
			t.Fatalf("%s: expected %v. got %v. unsafeSsl: %v", test.name, test.expected, err, test.unsafeSsl)
		}
	}
}

func TestChunkedResponse(t *testing.T) {
	s := `{"results":[{},{}]}{"results":[{}]}`
	r := client.NewChunkedResponse(strings.NewReader(s))
	resp, err := r.NextResponse()
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	} else if actual := len(resp.Results); actual != 2 {
		t.Fatalf("unexpected number of results.  expected %v, actual %v", 2, actual)
	}

	resp, err = r.NextResponse()
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	} else if actual := len(resp.Results); actual != 1 {
		t.Fatalf("unexpected number of results.  expected %v, actual %v", 1, actual)
	}

	resp, err = r.NextResponse()
	if err != nil {
		t.Fatalf("unexpected error.  expected %v, actual %v", nil, err)
	} else if resp != nil {
		t.Fatalf("unexpected response.  expected %v, actual %v", nil, resp)
	}
}
