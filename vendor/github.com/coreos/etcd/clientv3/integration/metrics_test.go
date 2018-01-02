// Copyright 2016 The etcd Authors
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

package integration

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/pkg/transport"

	"github.com/prometheus/client_golang/prometheus"
	"golang.org/x/net/context"
)

func TestV3ClientMetrics(t *testing.T) {
	defer testutil.AfterTest(t)

	var (
		addr string = "localhost:27989"
		ln   net.Listener
		err  error
	)

	// listen for all prometheus metrics
	donec := make(chan struct{})
	go func() {
		defer close(donec)

		srv := &http.Server{Handler: prometheus.Handler()}
		srv.SetKeepAlivesEnabled(false)

		ln, err = transport.NewUnixListener(addr)
		if err != nil {
			t.Fatalf("Error: %v occurred while listening on addr: %v", err, addr)
		}

		err = srv.Serve(ln)
		if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			t.Fatalf("Err serving http requests: %v", err)
		}
	}()

	url := "unix://" + addr + "/metrics"

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	client := clus.Client(0)

	w := clientv3.NewWatcher(client)
	defer w.Close()

	kv := clientv3.NewKV(client)

	wc := w.Watch(context.Background(), "foo")

	wBefore := sumCountersForMetricAndLabels(t, url, "grpc_client_msg_received_total", "Watch", "bidi_stream")

	pBefore := sumCountersForMetricAndLabels(t, url, "grpc_client_started_total", "Put", "unary")

	_, err = kv.Put(context.Background(), "foo", "bar")
	if err != nil {
		t.Errorf("Error putting value in key store")
	}

	pAfter := sumCountersForMetricAndLabels(t, url, "grpc_client_started_total", "Put", "unary")
	if pBefore+1 != pAfter {
		t.Errorf("grpc_client_started_total expected %d, got %d", 1, pAfter-pBefore)
	}

	// consume watch response
	select {
	case <-wc:
	case <-time.After(10 * time.Second):
		t.Error("Timeout occurred for getting watch response")
	}

	wAfter := sumCountersForMetricAndLabels(t, url, "grpc_client_msg_received_total", "Watch", "bidi_stream")
	if wBefore+1 != wAfter {
		t.Errorf("grpc_client_msg_received_total expected %d, got %d", 1, wAfter-wBefore)
	}

	ln.Close()
	<-donec
}

func sumCountersForMetricAndLabels(t *testing.T, url string, metricName string, matchingLabelValues ...string) int {
	count := 0
	for _, line := range getHTTPBodyAsLines(t, url) {
		ok := true
		if !strings.HasPrefix(line, metricName) {
			continue
		}

		for _, labelValue := range matchingLabelValues {
			if !strings.Contains(line, `"`+labelValue+`"`) {
				ok = false
				break
			}
		}

		if !ok {
			continue
		}

		valueString := line[strings.LastIndex(line, " ")+1 : len(line)-1]
		valueFloat, err := strconv.ParseFloat(valueString, 32)
		if err != nil {
			t.Fatalf("failed parsing value for line: %v and matchingLabelValues: %v", line, matchingLabelValues)
		}
		count += int(valueFloat)
	}
	return count
}

func getHTTPBodyAsLines(t *testing.T, url string) []string {
	cfgtls := transport.TLSInfo{}
	tr, err := transport.NewTransport(cfgtls, time.Second)
	if err != nil {
		t.Fatalf("Error getting transport: %v", err)
	}

	tr.MaxIdleConns = -1
	tr.DisableKeepAlives = true

	cli := &http.Client{Transport: tr}

	resp, err := cli.Get(url)
	if err != nil {
		t.Fatalf("Error fetching: %v", err)
	}

	reader := bufio.NewReader(resp.Body)
	lines := []string{}
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			} else {
				t.Fatalf("error reading: %v", err)
			}
		}
		lines = append(lines, line)
	}
	resp.Body.Close()
	return lines
}
