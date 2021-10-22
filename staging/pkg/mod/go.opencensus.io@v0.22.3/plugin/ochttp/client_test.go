// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ochttp_test

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"go.opencensus.io/plugin/ochttp"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/trace"
)

const reqCount = 5

func TestClientNew(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(resp http.ResponseWriter, req *http.Request) {
		resp.Write([]byte("Hello, world!"))
	}))
	defer server.Close()

	if err := view.Register(
		ochttp.ClientSentBytesDistribution,
		ochttp.ClientReceivedBytesDistribution,
		ochttp.ClientRoundtripLatencyDistribution,
		ochttp.ClientCompletedCount,
	); err != nil {
		t.Fatalf("Failed to register ochttp.DefaultClientViews error: %v", err)
	}

	views := []string{
		"opencensus.io/http/client/sent_bytes",
		"opencensus.io/http/client/received_bytes",
		"opencensus.io/http/client/roundtrip_latency",
		"opencensus.io/http/client/completed_count",
	}
	for _, name := range views {
		v := view.Find(name)
		if v == nil {
			t.Errorf("view not found %q", name)
			continue
		}
	}

	var wg sync.WaitGroup
	var tr ochttp.Transport
	errs := make(chan error, reqCount)
	wg.Add(reqCount)

	for i := 0; i < reqCount; i++ {
		go func() {
			defer wg.Done()
			req, err := http.NewRequest("POST", server.URL, strings.NewReader("req-body"))
			if err != nil {
				errs <- fmt.Errorf("error creating request: %v", err)
			}
			resp, err := tr.RoundTrip(req)
			if err != nil {
				errs <- fmt.Errorf("response error: %v", err)
			}
			if err := resp.Body.Close(); err != nil {
				errs <- fmt.Errorf("error closing response body: %v", err)
			}
			if got, want := resp.StatusCode, 200; got != want {
				errs <- fmt.Errorf("resp.StatusCode=%d; wantCount %d", got, want)
			}
		}()
	}

	go func() {
		wg.Wait()
		close(errs)
	}()

	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}

	for _, viewName := range views {
		v := view.Find(viewName)
		if v == nil {
			t.Errorf("view not found %q", viewName)
			continue
		}
		rows, err := view.RetrieveData(v.Name)
		if err != nil {
			t.Error(err)
			continue
		}
		if got, want := len(rows), 1; got != want {
			t.Errorf("len(%q) = %d; want %d", viewName, got, want)
			continue
		}
		data := rows[0].Data
		var count int64
		switch data := data.(type) {
		case *view.CountData:
			count = data.Value
		case *view.DistributionData:
			count = data.Count
		default:
			t.Errorf("Unknown data type: %v", data)
			continue
		}
		if got := count; got != reqCount {
			t.Fatalf("%s = %d; want %d", viewName, got, reqCount)
		}
	}
}

func TestClientOld(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(resp http.ResponseWriter, req *http.Request) {
		resp.Write([]byte("Hello, world!"))
	}))
	defer server.Close()

	if err := view.Register(ochttp.DefaultClientViews...); err != nil {
		t.Fatalf("Failed to register ochttp.DefaultClientViews error: %v", err)
	}

	views := []string{
		"opencensus.io/http/client/request_count",
		"opencensus.io/http/client/latency",
		"opencensus.io/http/client/request_bytes",
		"opencensus.io/http/client/response_bytes",
	}
	for _, name := range views {
		v := view.Find(name)
		if v == nil {
			t.Errorf("view not found %q", name)
			continue
		}
	}

	var wg sync.WaitGroup
	var tr ochttp.Transport
	errs := make(chan error, reqCount)
	wg.Add(reqCount)

	for i := 0; i < reqCount; i++ {
		go func() {
			defer wg.Done()
			req, err := http.NewRequest("POST", server.URL, strings.NewReader("req-body"))
			if err != nil {
				errs <- fmt.Errorf("error creating request: %v", err)
			}
			resp, err := tr.RoundTrip(req)
			if err != nil {
				errs <- fmt.Errorf("response error: %v", err)
			}
			if err := resp.Body.Close(); err != nil {
				errs <- fmt.Errorf("error closing response body: %v", err)
			}
			if got, want := resp.StatusCode, 200; got != want {
				errs <- fmt.Errorf("resp.StatusCode=%d; wantCount %d", got, want)
			}
		}()
	}

	go func() {
		wg.Wait()
		close(errs)
	}()

	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}

	for _, viewName := range views {
		v := view.Find(viewName)
		if v == nil {
			t.Errorf("view not found %q", viewName)
			continue
		}
		rows, err := view.RetrieveData(v.Name)
		if err != nil {
			t.Error(err)
			continue
		}
		if got, want := len(rows), 1; got != want {
			t.Errorf("len(%q) = %d; want %d", viewName, got, want)
			continue
		}
		data := rows[0].Data
		var count int64
		switch data := data.(type) {
		case *view.CountData:
			count = data.Value
		case *view.DistributionData:
			count = data.Count
		default:
			t.Errorf("Unknown data type: %v", data)
			continue
		}
		if got := count; got != reqCount {
			t.Fatalf("%s = %d; want %d", viewName, got, reqCount)
		}
	}
}

var noTrace = trace.StartOptions{Sampler: trace.NeverSample()}

func BenchmarkTransportNoTrace(b *testing.B) {
	benchmarkClientServer(b, &ochttp.Transport{StartOptions: noTrace})
}

func BenchmarkTransport(b *testing.B) {
	benchmarkClientServer(b, &ochttp.Transport{})
}

func benchmarkClientServer(b *testing.B, transport *ochttp.Transport) {
	b.ReportAllocs()
	ts := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(rw, "Hello world.\n")
	}))
	defer ts.Close()
	transport.StartOptions.Sampler = trace.AlwaysSample()
	var client http.Client
	client.Transport = transport
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		res, err := client.Get(ts.URL)
		if err != nil {
			b.Fatalf("Get: %v", err)
		}
		all, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			b.Fatal("ReadAll:", err)
		}
		body := string(all)
		if body != "Hello world.\n" {
			b.Fatal("Got body:", body)
		}
	}
}

func BenchmarkTransportParallel64NoTrace(b *testing.B) {
	benchmarkClientServerParallel(b, 64, &ochttp.Transport{StartOptions: noTrace})
}

func BenchmarkTransportParallel64(b *testing.B) {
	benchmarkClientServerParallel(b, 64, &ochttp.Transport{})
}

func benchmarkClientServerParallel(b *testing.B, parallelism int, transport *ochttp.Transport) {
	b.ReportAllocs()
	ts := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(rw, "Hello world.\n")
	}))
	defer ts.Close()

	var c http.Client
	transport.Base = &http.Transport{
		MaxIdleConns:        parallelism,
		MaxIdleConnsPerHost: parallelism,
	}
	transport.StartOptions.Sampler = trace.AlwaysSample()
	c.Transport = transport

	b.ResetTimer()

	// TODO(ramonza): replace with b.RunParallel (it didn't work when I tried)

	var wg sync.WaitGroup
	wg.Add(parallelism)
	for i := 0; i < parallelism; i++ {
		iterations := b.N / parallelism
		if i == 0 {
			iterations += b.N % parallelism
		}
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				res, err := c.Get(ts.URL)
				if err != nil {
					b.Logf("Get: %v", err)
					return
				}
				all, err := ioutil.ReadAll(res.Body)
				res.Body.Close()
				if err != nil {
					b.Logf("ReadAll: %v", err)
					return
				}
				body := string(all)
				if body != "Hello world.\n" {
					panic("Got body: " + body)
				}
			}
		}()
	}
	wg.Wait()
}
