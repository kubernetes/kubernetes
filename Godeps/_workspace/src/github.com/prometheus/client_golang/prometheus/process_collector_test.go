package prometheus

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"testing"

	"github.com/prometheus/procfs"
)

func TestProcessCollector(t *testing.T) {
	if _, err := procfs.Self(); err != nil {
		t.Skipf("skipping TestProcessCollector, procfs not available: %s", err)
	}

	registry := newRegistry()
	registry.Register(NewProcessCollector(os.Getpid(), ""))
	registry.Register(NewProcessCollectorPIDFn(
		func() (int, error) { return os.Getpid(), nil }, "foobar"))

	s := httptest.NewServer(InstrumentHandler("prometheus", registry))
	defer s.Close()
	r, err := http.Get(s.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}

	for _, re := range []*regexp.Regexp{
		regexp.MustCompile("process_cpu_seconds_total [0-9]"),
		regexp.MustCompile("process_max_fds [0-9]{2,}"),
		regexp.MustCompile("process_open_fds [1-9]"),
		regexp.MustCompile("process_virtual_memory_bytes [1-9]"),
		regexp.MustCompile("process_resident_memory_bytes [1-9]"),
		regexp.MustCompile("process_start_time_seconds [0-9.]{10,}"),
		regexp.MustCompile("foobar_process_cpu_seconds_total [0-9]"),
		regexp.MustCompile("foobar_process_max_fds [0-9]{2,}"),
		regexp.MustCompile("foobar_process_open_fds [1-9]"),
		regexp.MustCompile("foobar_process_virtual_memory_bytes [1-9]"),
		regexp.MustCompile("foobar_process_resident_memory_bytes [1-9]"),
		regexp.MustCompile("foobar_process_start_time_seconds [0-9.]{10,}"),
	} {
		if !re.Match(body) {
			t.Errorf("want body to match %s\n%s", re, body)
		}
	}
}
