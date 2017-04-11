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

package tcpproxy

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestUserspaceProxy(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	want := "hello proxy"
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, want)
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	p := TCPProxy{
		Listener:  l,
		Endpoints: []string{u.Host},
	}
	go p.Run()
	defer p.Stop()

	u.Host = l.Addr().String()

	res, err := http.Get(u.String())
	if err != nil {
		t.Fatal(err)
	}
	got, gerr := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if gerr != nil {
		t.Fatal(gerr)
	}

	if string(got) != want {
		t.Errorf("got = %s, want %s", got, want)
	}
}
