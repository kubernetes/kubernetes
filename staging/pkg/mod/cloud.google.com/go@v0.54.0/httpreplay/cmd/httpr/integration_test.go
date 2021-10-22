// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main_test

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"cloud.google.com/go/internal/testutil"
	"cloud.google.com/go/storage"
	"golang.org/x/oauth2"
	"google.golang.org/api/option"
)

const initial = "initial state"

func TestIntegration_HTTPR(t *testing.T) {
	if testing.Short() {
		t.Skip("Integration tests skipped in short mode")
	}
	if testutil.ProjID() == "" {
		t.Fatal("set GCLOUD_TESTS_GOLANG_PROJECT_ID and GCLOUD_TESTS_GOLANG_KEY")
	}
	// Get a unique temporary filename.
	f, err := ioutil.TempFile("", "httpreplay")
	if err != nil {
		t.Fatal(err)
	}
	replayFilename := f.Name()
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(replayFilename)

	if err := exec.Command("go", "build").Run(); err != nil {
		t.Fatalf("running 'go build': %v", err)
	}
	defer os.Remove("./httpr")
	want := runRecord(t, replayFilename)
	got := runReplay(t, replayFilename)
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func runRecord(t *testing.T, filename string) string {
	cmd, tr, cport, err := start("-record", filename)
	if err != nil {
		t.Fatal(err)
	}
	defer stop(t, cmd)

	ctx := context.Background()
	hc := &http.Client{
		Transport: &oauth2.Transport{
			Base:   tr,
			Source: testutil.TokenSource(ctx, storage.ScopeFullControl),
		},
	}
	res, err := http.Post(
		fmt.Sprintf("http://localhost:%s/initial", cport),
		"text/plain",
		strings.NewReader(initial))
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 200 {
		t.Fatalf("from POST: %s", res.Status)
	}
	info, err := getBucketInfo(ctx, hc)
	if err != nil {
		t.Fatal(err)
	}
	return info
}

func runReplay(t *testing.T, filename string) string {
	cmd, tr, cport, err := start("-replay", filename)
	if err != nil {
		t.Fatal(err)
	}
	defer stop(t, cmd)

	hc := &http.Client{Transport: tr}
	res, err := http.Get(fmt.Sprintf("http://localhost:%s/initial", cport))
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 200 {
		t.Fatalf("from GET: %s", res.Status)
	}
	bytes, err := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(bytes), initial; got != want {
		t.Errorf("initial: got %q, want %q", got, want)
	}
	info, err := getBucketInfo(context.Background(), hc)
	if err != nil {
		t.Fatal(err)
	}
	return info
}

// Start the proxy binary and wait for it to come up.
// Return a transport that talks to the proxy, as well as the control port.
// modeFlag must be either "-record" or "-replay".
func start(modeFlag, filename string) (*exec.Cmd, *http.Transport, string, error) {
	pport, err := pickPort()
	if err != nil {
		return nil, nil, "", err
	}
	cport, err := pickPort()
	if err != nil {
		return nil, nil, "", err
	}
	cmd := exec.Command("./httpr", "-port", pport, "-control-port", cport, modeFlag, filename, "-debug-headers")
	if err := cmd.Start(); err != nil {
		return nil, nil, "", err
	}
	// Wait for the server to come up.
	serverUp := false
	for i := 0; i < 10; i++ {
		if conn, err := net.Dial("tcp", "localhost:"+cport); err == nil {
			conn.Close()
			serverUp = true
			break
		}
		time.Sleep(time.Second)
	}
	if !serverUp {
		return nil, nil, "", errors.New("server never came up")
	}
	tr, err := proxyTransport(pport, cport)
	if err != nil {
		return nil, nil, "", err
	}
	return cmd, tr, cport, nil
}

func stop(t *testing.T, cmd *exec.Cmd) {
	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
}

// pickPort picks an unused port.
func pickPort() (string, error) {
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		return "", err
	}
	addr := l.Addr().String()
	_, port, err := net.SplitHostPort(addr)
	if err != nil {
		return "", err
	}
	l.Close()
	return port, nil
}

func proxyTransport(pport, cport string) (*http.Transport, error) {
	caCert, err := getBody(fmt.Sprintf("http://localhost:%s/authority.cer", cport))
	if err != nil {
		return nil, err
	}
	caCertPool := x509.NewCertPool()
	if !caCertPool.AppendCertsFromPEM([]byte(caCert)) {
		return nil, errors.New("bad CA Cert")
	}
	return &http.Transport{
		Proxy:           http.ProxyURL(&url.URL{Host: "localhost:" + pport}),
		TLSClientConfig: &tls.Config{RootCAs: caCertPool},
	}, nil
}

func getBucketInfo(ctx context.Context, hc *http.Client) (string, error) {
	client, err := storage.NewClient(ctx, option.WithHTTPClient(hc))
	if err != nil {
		return "", err
	}
	defer client.Close()
	b := client.Bucket(testutil.ProjID())
	attrs, err := b.Attrs(ctx)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("name:%s reqpays:%v location:%s sclass:%s",
		attrs.Name, attrs.RequesterPays, attrs.Location, attrs.StorageClass), nil
}

func getBody(url string) ([]byte, error) {
	res, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	if res.StatusCode != 200 {
		return nil, fmt.Errorf("response: %s", res.Status)
	}
	defer res.Body.Close()
	return ioutil.ReadAll(res.Body)
}
