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
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"golang.org/x/net/context"

	"github.com/coreos/etcd/client"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
)

// TestV2NoRetryEOF tests destructive api calls won't retry on a disconnection.
func TestV2NoRetryEOF(t *testing.T) {
	defer testutil.AfterTest(t)
	// generate an EOF response; specify address so appears first in sorted ep list
	lEOF := integration.NewListenerWithAddr(t, fmt.Sprintf("eof:123.%d.sock", os.Getpid()))
	defer lEOF.Close()
	tries := uint32(0)
	go func() {
		for {
			conn, err := lEOF.Accept()
			if err != nil {
				return
			}
			atomic.AddUint32(&tries, 1)
			conn.Close()
		}
	}()
	eofURL := integration.UrlScheme + "://" + lEOF.Addr().String()
	cli := integration.MustNewHTTPClient(t, []string{eofURL, eofURL}, nil)
	kapi := client.NewKeysAPI(cli)
	for i, f := range noRetryList(kapi) {
		startTries := atomic.LoadUint32(&tries)
		if err := f(); err == nil {
			t.Errorf("#%d: expected EOF error, got nil", i)
		}
		endTries := atomic.LoadUint32(&tries)
		if startTries+1 != endTries {
			t.Errorf("#%d: expected 1 try, got %d", i, endTries-startTries)
		}
	}
}

// TestV2NoRetryNoLeader tests destructive api calls won't retry if given an error code.
func TestV2NoRetryNoLeader(t *testing.T) {
	defer testutil.AfterTest(t)

	lHttp := integration.NewListenerWithAddr(t, fmt.Sprintf("errHttp:123.%d.sock", os.Getpid()))
	eh := &errHandler{errCode: http.StatusServiceUnavailable}
	srv := httptest.NewUnstartedServer(eh)
	defer lHttp.Close()
	defer srv.Close()
	srv.Listener = lHttp
	go srv.Start()
	lHttpURL := integration.UrlScheme + "://" + lHttp.Addr().String()

	cli := integration.MustNewHTTPClient(t, []string{lHttpURL, lHttpURL}, nil)
	kapi := client.NewKeysAPI(cli)
	// test error code
	for i, f := range noRetryList(kapi) {
		reqs := eh.reqs
		if err := f(); err == nil || !strings.Contains(err.Error(), "no leader") {
			t.Errorf("#%d: expected \"no leader\", got %v", i, err)
		}
		if eh.reqs != reqs+1 {
			t.Errorf("#%d: expected 1 request, got %d", i, eh.reqs-reqs)
		}
	}
}

// TestV2RetryRefuse tests destructive api calls will retry if a connection is refused.
func TestV2RetryRefuse(t *testing.T) {
	defer testutil.AfterTest(t)
	cl := integration.NewCluster(t, 1)
	cl.Launch(t)
	defer cl.Terminate(t)
	// test connection refused; expect no error failover
	cli := integration.MustNewHTTPClient(t, []string{integration.UrlScheme + "://refuseconn:123", cl.URL(0)}, nil)
	kapi := client.NewKeysAPI(cli)
	if _, err := kapi.Set(context.Background(), "/delkey", "def", nil); err != nil {
		t.Fatal(err)
	}
	for i, f := range noRetryList(kapi) {
		if err := f(); err != nil {
			t.Errorf("#%d: unexpected retry failure (%v)", i, err)
		}
	}
}

type errHandler struct {
	errCode int
	reqs    int
}

func (eh *errHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	req.Body.Close()
	eh.reqs++
	w.WriteHeader(eh.errCode)
}

func noRetryList(kapi client.KeysAPI) []func() error {
	return []func() error{
		func() error {
			opts := &client.SetOptions{PrevExist: client.PrevNoExist}
			_, err := kapi.Set(context.Background(), "/setkey", "bar", opts)
			return err
		},
		func() error {
			_, err := kapi.Delete(context.Background(), "/delkey", nil)
			return err
		},
	}
}
