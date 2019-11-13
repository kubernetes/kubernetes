/*
Copyright 2018 The Kubernetes Authors.

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

package master

import (
	"io"
	"io/ioutil"
	"net/http"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestGracefulShutdown(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())

	tearDownOnce := sync.Once{}
	defer tearDownOnce.Do(server.TearDownFn)

	transport, err := rest.TransportFor(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	client := http.Client{Transport: transport}

	req, body, err := newBlockingRequest("POST", server.ClientConfig.Host+"/api/v1/namespaces")
	if err != nil {
		t.Fatal(err)
	}
	respErrCh := backgroundRoundtrip(transport, req)

	t.Logf("server should be blocking request for data in body")
	time.Sleep(time.Millisecond * 500)
	select {
	case respErr := <-respErrCh:
		if respErr.err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		bs, err := ioutil.ReadAll(respErr.resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		t.Fatalf("unexpected server answer: %d, body: %s", respErr.resp.StatusCode, string(bs))
	default:
	}

	t.Logf("server should answer")
	resp, err := client.Get(server.ClientConfig.Host + "/")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	t.Logf("shutting down server")
	tearDownOnce.Do(server.TearDownFn)

	t.Logf("server should fail new requests")
	if err := wait.Poll(time.Millisecond*100, wait.ForeverTestTimeout, func() (done bool, err error) {
		resp, err := client.Get(server.ClientConfig.Host + "/")
		if err != nil {
			return true, nil
		}
		resp.Body.Close()
		return false, nil
	}); err != nil {
		t.Fatalf("server did not shutdown")
	}

	t.Logf("server should answer pending request")
	time.Sleep(time.Millisecond * 500)
	if _, err := body.Write([]byte("garbage")); err != nil {
		t.Fatal(err)
	}
	body.Close()
	respErr := <-respErrCh
	if respErr.err != nil {
		t.Fatal(respErr.err)
	}
	defer respErr.resp.Body.Close()
	bs, err := ioutil.ReadAll(respErr.resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("response: code %d, body: %s", respErr.resp.StatusCode, string(bs))
}

type responseErrorPair struct {
	resp *http.Response
	err  error
}

func backgroundRoundtrip(transport http.RoundTripper, req *http.Request) <-chan responseErrorPair {
	ch := make(chan responseErrorPair)
	go func() {
		resp, err := transport.RoundTrip(req)
		ch <- responseErrorPair{resp, err}
	}()
	return ch
}

func newBlockingRequest(method, url string) (*http.Request, io.WriteCloser, error) {
	bodyReader, bodyWriter := io.Pipe()
	req, err := http.NewRequest(method, url, bodyReader)
	return req, bodyWriter, err
}
