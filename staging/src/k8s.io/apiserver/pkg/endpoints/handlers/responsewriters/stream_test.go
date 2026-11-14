/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package responsewriters

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flushwriter"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

type fakeStreamer struct {
	out   io.ReadCloser
	flush bool
}

func (s *fakeStreamer) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (s *fakeStreamer) DeepCopyObject() runtime.Object   { panic("not implemented") }
func (s *fakeStreamer) InputStream(ctx context.Context, apiVersion, acceptHeader string) (io.ReadCloser, bool, string, error) {
	return s.out, s.flush, "text/plain", nil
}

func streamObjectServer(streamer *fakeStreamer) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		StreamObject(http.StatusOK, schema.GroupVersion{Version: "v1"}, nil, streamer, w, req)
	}))
}

// rawClient sends Accept-Encoding only when a test sets it and does not
// decode responses.
var rawClient = &http.Client{Transport: &http.Transport{DisableCompression: true}}

// get requests url with the given Accept-Encoding and returns the raw,
// undecoded response.
func get(t *testing.T, url, acceptEncoding string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	if acceptEncoding != "" {
		req.Header.Set("Accept-Encoding", acceptEncoding)
	}
	resp, err := rawClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Uncompressed {
		t.Fatal("the client decoded the response")
	}
	return resp
}

func TestStreamObjectCompression(t *testing.T) {
	payload := strings.Repeat("2026-10-03T12:00:00Z level=info msg=\"request served\" path=/healthz\n", 1000)

	tests := []struct {
		name           string
		compression    bool
		acceptEncoding string
		flush          bool
		wantGzip       bool
	}{
		{name: "gzip requested", compression: true, acceptEncoding: "gzip", wantGzip: true},
		{name: "gzip requested, followed", compression: true, acceptEncoding: "gzip", flush: true, wantGzip: true},
		{name: "gzip among others", compression: true, acceptEncoding: "deflate, gzip", wantGzip: true},
		{name: "not requested", compression: true},
		{name: "not requested, followed", compression: true, flush: true},
		{name: "feature disabled", compression: false, acceptEncoding: "gzip"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIResponseCompression, tt.compression)

			server := streamObjectServer(&fakeStreamer{out: io.NopCloser(strings.NewReader(payload)), flush: tt.flush})
			defer server.Close()

			resp := get(t, server.URL, tt.acceptEncoding)
			defer resp.Body.Close()

			if got := resp.Header.Get("Content-Type"); got != "text/plain" {
				t.Errorf("Content-Type = %q, want text/plain", got)
			}
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if !tt.wantGzip {
				if got := resp.Header.Get("Content-Encoding"); got != "" {
					t.Errorf("Content-Encoding = %q, want none", got)
				}
				if string(body) != payload {
					t.Errorf("body differs from the stream")
				}
				return
			}
			if got := resp.Header.Get("Content-Encoding"); got != "gzip" {
				t.Fatalf("Content-Encoding = %q, want gzip", got)
			}
			if got := resp.Header.Get("Vary"); got != "Accept-Encoding" {
				t.Errorf("Vary = %q, want Accept-Encoding", got)
			}
			if len(body) >= len(payload) {
				t.Errorf("compressed body is %d bytes, stream is %d", len(body), len(payload))
			}
			gr, err := gzip.NewReader(bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := io.ReadAll(gr)
			if err != nil {
				t.Fatalf("gzip body did not decode to the end: %v", err)
			}
			if string(decoded) != payload {
				t.Errorf("decoded body differs from the stream")
			}
		})
	}
}

// TestStreamObjectCompressionFollow checks that a followed stream is not held
// back by the compressor: every line must reach the client before the next
// one is written. Buffering in gzip is what kept compression off streaming
// responses when APIResponseCompression was introduced.
func TestStreamObjectCompressionFollow(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIResponseCompression, true)

	pr, pw := io.Pipe()
	server := streamObjectServer(&fakeStreamer{out: pr, flush: true})
	defer server.Close()
	// Ends the handler before the server waits for it, also when the test fails.
	defer pw.Close()

	resp := get(t, server.URL, "gzip")
	defer resp.Body.Close()
	if got := resp.Header.Get("Content-Encoding"); got != "gzip" {
		t.Fatalf("Content-Encoding = %q, want gzip", got)
	}
	lines := readLines(resp.Body)

	for i, line := range []string{"first", "second", "third"} {
		if _, err := io.WriteString(pw, line+"\n"); err != nil {
			t.Fatal(err)
		}
		select {
		case got := <-lines:
			if got != line {
				t.Fatalf("line %d = %q, want %q", i, got, line)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatalf("line %d did not reach the client while the stream was open", i)
		}
	}

	pw.Close()
	if got, ok := <-lines; ok {
		t.Fatalf("unexpected line %q after the stream ended", got)
	}
}

// TestGzipStreamWriterBatch checks that a stream that is not followed is sent
// once gzipBatchBytes are pending, not only at the end.
func TestGzipStreamWriterBatch(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := newGzipStreamWriter(rec, false)
	line := strings.Repeat("x", 1023) + "\n"
	for written := 0; written < gzipBatchBytes-len(line); written += len(line) {
		if _, err := io.WriteString(sw, line); err != nil {
			t.Fatal(err)
		}
	}
	if rec.Body.Len() != 0 {
		t.Fatalf("%d bytes sent before gzipBatchBytes were pending", rec.Body.Len())
	}
	if _, err := io.WriteString(sw, line); err != nil {
		t.Fatal(err)
	}
	if rec.Body.Len() == 0 {
		t.Fatal("nothing sent once gzipBatchBytes were pending")
	}
	if rec.Flushed {
		t.Error("a stream that is not followed was flushed")
	}
	if err := sw.Close(); err != nil {
		t.Fatal(err)
	}
	gr, err := gzip.NewReader(rec.Body)
	if err != nil {
		t.Fatal(err)
	}
	if decoded, err := io.ReadAll(gr); err != nil || len(decoded) != gzipBatchBytes {
		t.Fatalf("decoded %d bytes, %v; want %d", len(decoded), err, gzipBatchBytes)
	}
}

// TestGzipStreamWriterEachWrite checks, for writes of many sizes, that the
// bytes sent for each write decode to exactly that write as soon as they are
// sent, with the stream's history primed in between.
func TestGzipStreamWriterEachWrite(t *testing.T) {
	var writes [][]byte
	for i := 0; i < 200; i++ {
		switch {
		case i%50 == 0:
			writes = append(writes, []byte("ok\n"))
		case i%97 == 0:
			writes = append(writes, bytes.Repeat([]byte("a long line that repeats\n"), 3000)) // > 64 KiB
		case i%31 == 0:
			writes = append(writes, bytes.Repeat([]byte{byte(i)}, gzipHistoryBytes))
		case i%13 == 0:
			writes = append(writes, nil)
		default:
			writes = append(writes, []byte(fmt.Sprintf("2026-10-03T12:00:%02dZ level=info msg=\"request served\" path=/api/v1/items/%d\n", i%60, i*7919)))
		}
	}

	rec := httptest.NewRecorder()
	sw := newGzipStreamWriter(rec, true)
	pr, pw := io.Pipe()
	defer pw.Close()
	decoded := make(chan []byte, 1)
	go func() {
		gr, err := gzip.NewReader(pr)
		if err != nil {
			t.Error(err)
			return
		}
		for _, w := range writes {
			buf := make([]byte, len(w))
			if _, err := io.ReadFull(gr, buf); err != nil {
				t.Error(err)
				return
			}
			decoded <- buf
		}
		if n, err := gr.Read(make([]byte, 1)); n != 0 || err != io.EOF {
			t.Errorf("after the last write: n=%d err=%v", n, err)
		}
		decoded <- nil
	}()
	sent := 0
	for i, w := range writes {
		if _, err := sw.Write(w); err != nil {
			t.Fatal(err)
		}
		if len(w) > 0 && rec.Body.Len() == sent {
			t.Fatalf("write %d (%d bytes) sent nothing", i, len(w))
		}
		if _, err := pw.Write(rec.Body.Bytes()[sent:]); err != nil {
			t.Fatal(err)
		}
		sent = rec.Body.Len()
		select {
		case got := <-decoded:
			if !bytes.Equal(got, w) {
				t.Fatalf("write %d decoded to %d bytes, want %d", i, len(got), len(w))
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatalf("write %d (%d bytes) was not decodable once sent", i, len(w))
		}
	}
	if err := sw.Close(); err != nil {
		t.Fatal(err)
	}
	// The reader needs the body's EOF to know no gzip member follows.
	pw.Write(rec.Body.Bytes()[sent:])
	pw.Close()
	select {
	case <-decoded:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("the end of the stream was not decodable")
	}
}

// TestGzipStreamWriterHistory checks that priming lets a followed stream of
// similar lines, each compressed on its own, still compress against the
// lines before it.
func TestGzipStreamWriterHistory(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := newGzipStreamWriter(rec, true)
	raw := 0
	for i := 0; i < 500; i++ {
		line := fmt.Sprintf("10.244.3.17 - - [03/Oct/2026:14:37:%02d +0000] \"GET /api/v1/items/%d HTTP/1.1\" 200 %d \"-\" \"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36\" req-%d\n", i%60, i*7919%100000, 1000+i%5000, i)
		raw += len(line)
		if _, err := io.WriteString(sw, line); err != nil {
			t.Fatal(err)
		}
	}
	if err := sw.Close(); err != nil {
		t.Fatal(err)
	}
	if ratio := float64(raw) / float64(rec.Body.Len()); ratio < 2 {
		t.Errorf("compressed %d bytes to %d (%.2fx), want at least 2x", raw, rec.Body.Len(), ratio)
	}
	gr, err := gzip.NewReader(rec.Body)
	if err != nil {
		t.Fatal(err)
	}
	if decoded, err := io.ReadAll(gr); err != nil || len(decoded) != raw {
		t.Fatalf("decoded %d bytes, %v; want %d", len(decoded), err, raw)
	}
}

func TestGzipStreamWriterWriteAfterClose(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := newGzipStreamWriter(rec, true)
	if _, err := io.WriteString(sw, "line\n"); err != nil {
		t.Fatal(err)
	}
	if err := sw.Close(); err != nil {
		t.Fatal(err)
	}
	size := rec.Body.Len()
	if n, err := io.WriteString(sw, "late\n"); n != 0 || !errors.Is(err, errGzipStreamClosed) {
		t.Fatalf("Write after Close = %d, %v; want 0, %v", n, err, errGzipStreamClosed)
	}
	if rec.Body.Len() != size {
		t.Fatalf("Write after Close wrote %d bytes", rec.Body.Len()-size)
	}
	gr, err := gzip.NewReader(rec.Body)
	if err != nil {
		t.Fatal(err)
	}
	if body, err := io.ReadAll(gr); err != nil || string(body) != "line\n" {
		t.Fatalf("decoded %q, %v; want %q", body, err, "line\n")
	}
}

func TestAppendTail(t *testing.T) {
	var hist []byte
	var all []byte
	for i := 0; i < 1000; i++ {
		p := bytes.Repeat([]byte{byte(i)}, i%300)
		all = append(all, p...)
		hist = appendTail(hist, p, 256)
		want := all
		if len(want) > 256 {
			want = want[len(want)-256:]
		}
		if !bytes.Equal(hist, want) {
			t.Fatalf("step %d: history differs", i)
		}
		if cap(hist) > 256 {
			t.Fatalf("step %d: cap %d > 256", i, cap(hist))
		}
	}
}

// TestStreamObjectCompressionEmpty checks that a stream with no output is
// still a valid gzip body.
func TestStreamObjectCompressionEmpty(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIResponseCompression, true)

	for _, flush := range []bool{false, true} {
		server := streamObjectServer(&fakeStreamer{out: io.NopCloser(strings.NewReader("")), flush: flush})
		resp := get(t, server.URL, "gzip")
		gr, err := gzip.NewReader(resp.Body)
		if err != nil {
			t.Fatalf("flush=%v: body is not gzip: %v", flush, err)
		}
		if body, err := io.ReadAll(gr); err != nil || len(body) != 0 {
			t.Errorf("flush=%v: got %q, %v; want an empty body", flush, body, err)
		}
		resp.Body.Close()
		server.Close()
	}
}

// readLines decodes a gzip body and delivers its lines until it ends.
func readLines(body io.Reader) <-chan string {
	lines := make(chan string)
	go func() {
		defer close(lines)
		gr, err := gzip.NewReader(body)
		if err != nil {
			return
		}
		scanner := bufio.NewScanner(gr)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
	}()
	return lines
}

// benchLines returns a deterministic mix of log lines: nginx access logs,
// JSON application logs and short lines.
func benchLines() [][]byte {
	rng := rand.New(rand.NewSource(1))
	paths := []string{"/api/v1/items", "/healthz", "/api/v1/orders", "/static/app.js", "/api/v2/users"}
	levels := []string{"info", "info", "info", "warn", "error"}
	lines := make([][]byte, 2000)
	for i := range lines {
		switch r := rng.Intn(10); {
		case r < 6:
			lines[i] = fmt.Appendf(nil, "10.244.%d.%d - - [03/Oct/2026:14:%02d:%02d +0000] \"GET %s/%d HTTP/1.1\" %d %d \"-\" \"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36\" %d 0.%03d req-%x\n",
				rng.Intn(8), rng.Intn(250), rng.Intn(60), rng.Intn(60), paths[rng.Intn(len(paths))], rng.Intn(100000), []int{200, 200, 200, 304, 404}[rng.Intn(5)], rng.Intn(9000)+200, rng.Intn(900)+100, rng.Intn(999), rng.Uint64())
		case r < 9:
			lines[i] = fmt.Appendf(nil, `{"ts":"2026-10-03T14:%02d:%02d.%03dZ","level":"%s","msg":"request completed","method":"GET","path":"%s","status":%d,"duration_ms":%d,"trace_id":"%x"}`+"\n",
				rng.Intn(60), rng.Intn(60), rng.Intn(1000), levels[rng.Intn(len(levels))], paths[rng.Intn(len(paths))], []int{200, 200, 201, 404, 500}[rng.Intn(5)], rng.Intn(500), rng.Uint64())
		default:
			lines[i] = fmt.Appendf(nil, "worker %d: processed batch %d in %dms\n", rng.Intn(16), rng.Intn(1000000), rng.Intn(900))
		}
	}
	return lines
}

// discardResponseWriter counts what is written to it.
type discardResponseWriter struct {
	header http.Header
	n      int64
}

func (w *discardResponseWriter) Header() http.Header {
	if w.header == nil {
		w.header = http.Header{}
	}
	return w.header
}
func (w *discardResponseWriter) WriteHeader(int) {}
func (w *discardResponseWriter) Write(p []byte) (int, error) {
	w.n += int64(len(p))
	return len(p), nil
}
func (w *discardResponseWriter) Flush() {}

// BenchmarkLogStreamFollow writes one log line per write, as a followed pods/log
// stream does, without and with compression.
func BenchmarkLogStreamFollow(b *testing.B) {
	lines := benchLines()
	for _, gz := range []bool{false, true} {
		b.Run(map[bool]string{false: "plain", true: "gzip"}[gz], func(b *testing.B) {
			rw := &discardResponseWriter{}
			var w io.Writer = flushwriter.Wrap(rw)
			var sw *gzipStreamWriter
			if gz {
				sw = newGzipStreamWriter(rw, true)
				w = sw
			}
			var raw int64
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				line := lines[i%len(lines)]
				if _, err := w.Write(line); err != nil {
					b.Fatal(err)
				}
				raw += int64(len(line))
			}
			b.StopTimer()
			if sw != nil {
				sw.Close()
			}
			b.SetBytes(raw / int64(b.N))
			b.ReportMetric(float64(raw)/float64(rw.n), "ratio")
		})
	}
}

// BenchmarkLogStreamBulk writes 32 KiB at a time, as a pods/log request that is
// not followed does, without and with compression.
func BenchmarkLogStreamBulk(b *testing.B) {
	var chunk []byte
	for _, l := range benchLines() {
		chunk = append(chunk, l...)
	}
	chunk = chunk[:32*1024]
	for _, gz := range []bool{false, true} {
		b.Run(map[bool]string{false: "plain", true: "gzip"}[gz], func(b *testing.B) {
			rw := &discardResponseWriter{}
			var w io.Writer = rw
			var sw *gzipStreamWriter
			if gz {
				sw = newGzipStreamWriter(rw, false)
				w = sw
			}
			b.ReportAllocs()
			b.SetBytes(int64(len(chunk)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := w.Write(chunk); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			if sw != nil {
				sw.Close()
			}
			b.ReportMetric(float64(int64(b.N)*int64(len(chunk)))/float64(rw.n), "ratio")
		})
	}
}
