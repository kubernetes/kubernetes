// Copyright 2015 flannel authors
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

package remote

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"path"
	"time"

	"github.com/coreos/etcd/pkg/transport"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

// implements subnet.Manager by sending requests to the server
type RemoteManager struct {
	base      string // includes scheme, host, and port, and version
	transport *Transport
}

func NewTransport(info transport.TLSInfo) (*Transport, error) {
	cfg, err := info.ClientConfig()
	if err != nil {
		return nil, err
	}

	t := &Transport{
		// timeouts taken from http.DefaultTransport
		Dial: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).Dial,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     cfg,
	}

	return t, nil
}

func NewRemoteManager(listenAddr, cafile, certfile, keyfile string) (subnet.Manager, error) {
	tls := transport.TLSInfo{
		CAFile:   cafile,
		CertFile: certfile,
		KeyFile:  keyfile,
	}

	t, err := NewTransport(tls)
	if err != nil {
		return nil, err
	}

	var scheme string
	if tls.Empty() && tls.CAFile == "" {
		scheme = "http://"
	} else {
		scheme = "https://"
	}

	return &RemoteManager{
		base:      scheme + listenAddr + "/v1",
		transport: t,
	}, nil
}

func (m *RemoteManager) mkurl(network string, parts ...string) string {
	if network == "" {
		network = "/_"
	}
	if network[0] != '/' {
		network = "/" + network
	}
	return m.base + path.Join(append([]string{network}, parts...)...)
}

func (m *RemoteManager) GetNetworkConfig(ctx context.Context, network string) (*subnet.Config, error) {
	url := m.mkurl(network, "config")

	resp, err := m.httpVerb(ctx, "GET", url, "", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, httpError(resp)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	config, err := subnet.ParseConfig(string(body))
	if err != nil {
		return nil, err
	}

	return config, nil
}

func (m *RemoteManager) AcquireLease(ctx context.Context, network string, attrs *subnet.LeaseAttrs) (*subnet.Lease, error) {
	url := m.mkurl(network, "leases/")

	body, err := json.Marshal(attrs)
	if err != nil {
		return nil, err
	}

	resp, err := m.httpVerb(ctx, "POST", url, "application/json", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, httpError(resp)
	}

	newLease := &subnet.Lease{}
	if err := json.NewDecoder(resp.Body).Decode(newLease); err != nil {
		return nil, err
	}

	return newLease, nil
}

func (m *RemoteManager) RenewLease(ctx context.Context, network string, lease *subnet.Lease) error {
	url := m.mkurl(network, "leases", lease.Key())

	body, err := json.Marshal(lease)
	if err != nil {
		return err
	}

	resp, err := m.httpVerb(ctx, "PUT", url, "application/json", body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return httpError(resp)
	}

	newLease := &subnet.Lease{}
	if err := json.NewDecoder(resp.Body).Decode(newLease); err != nil {
		return err
	}

	*lease = *newLease
	return nil
}

func (m *RemoteManager) RevokeLease(ctx context.Context, network string, sn ip.IP4Net) error {
	url := m.mkurl(network, "leases", subnet.MakeSubnetKey(sn))

	resp, err := m.httpVerb(ctx, "DELETE", url, "", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return httpError(resp)
	}

	return nil
}

func (m *RemoteManager) watch(ctx context.Context, url string, cursor interface{}, wr interface{}) error {
	if cursor != nil {
		c, ok := cursor.(string)
		if !ok {
			return fmt.Errorf("internal error: RemoteManager.watch received non-string cursor")
		}

		url = fmt.Sprintf("%v?next=%v", url, c)
	}

	resp, err := m.httpVerb(ctx, "GET", url, "", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return httpError(resp)
	}

	if err := json.NewDecoder(resp.Body).Decode(wr); err != nil {
		return err
	}

	return nil
}

func (m *RemoteManager) WatchLease(ctx context.Context, network string, sn ip.IP4Net, cursor interface{}) (subnet.LeaseWatchResult, error) {
	url := m.mkurl(network, "leases", subnet.MakeSubnetKey(sn))

	wr := subnet.LeaseWatchResult{}
	err := m.watch(ctx, url, cursor, &wr)
	if err != nil {
		return subnet.LeaseWatchResult{}, err
	}
	if _, ok := wr.Cursor.(string); !ok {
		return subnet.LeaseWatchResult{}, fmt.Errorf("watch returned non-string cursor")
	}

	return wr, nil
}

func (m *RemoteManager) WatchLeases(ctx context.Context, network string, cursor interface{}) (subnet.LeaseWatchResult, error) {
	url := m.mkurl(network, "leases")

	wr := subnet.LeaseWatchResult{}
	err := m.watch(ctx, url, cursor, &wr)
	if err != nil {
		return subnet.LeaseWatchResult{}, err
	}
	if _, ok := wr.Cursor.(string); !ok {
		return subnet.LeaseWatchResult{}, fmt.Errorf("watch returned non-string cursor")
	}

	return wr, nil
}

func (m *RemoteManager) WatchNetworks(ctx context.Context, cursor interface{}) (subnet.NetworkWatchResult, error) {
	wr := subnet.NetworkWatchResult{}
	err := m.watch(ctx, m.base+"/", cursor, &wr)
	if err != nil {
		return subnet.NetworkWatchResult{}, err
	}

	if _, ok := wr.Cursor.(string); !ok {
		return subnet.NetworkWatchResult{}, fmt.Errorf("watch returned non-string cursor")
	}

	return wr, nil
}

func (m *RemoteManager) AddReservation(ctx context.Context, network string, r *subnet.Reservation) error {
	url := m.mkurl(network, "reservations")

	body, err := json.Marshal(r)
	if err != nil {
		return err
	}

	resp, err := m.httpVerb(ctx, "POST", url, "application/json", body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return httpError(resp)
	}
	return nil
}

func (m *RemoteManager) RemoveReservation(ctx context.Context, network string, sn ip.IP4Net) error {
	url := m.mkurl(network, "reservations", subnet.MakeSubnetKey(sn))

	resp, err := m.httpVerb(ctx, "DELETE", url, "", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return httpError(resp)
	}

	return nil
}

func (m *RemoteManager) ListReservations(ctx context.Context, network string) ([]subnet.Reservation, error) {
	url := m.mkurl(network, "reservations")

	resp, err := m.httpVerb(ctx, "GET", url, "", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, httpError(resp)
	}

	rs := []subnet.Reservation{}
	if err := json.NewDecoder(resp.Body).Decode(&rs); err != nil {
		return nil, err
	}

	return rs, nil
}

func httpError(resp *http.Response) error {
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	return fmt.Errorf("%v: %v", resp.Status, string(b))
}

type httpRespErr struct {
	resp *http.Response
	err  error
}

func (m *RemoteManager) httpDo(ctx context.Context, req *http.Request) (*http.Response, error) {
	// Run the HTTP request in a goroutine (so it can be canceled) and pass
	// the result via the channel c
	client := &http.Client{Transport: m.transport}
	c := make(chan httpRespErr, 1)
	go func() {
		resp, err := client.Do(req)
		c <- httpRespErr{resp, err}
	}()

	select {
	case <-ctx.Done():
		m.transport.CancelRequest(req)
		<-c // Wait for f to return.
		return nil, ctx.Err()
	case r := <-c:
		return r.resp, r.err
	}
}

func (m *RemoteManager) httpVerb(ctx context.Context, method, url, contentType string, body []byte) (*http.Response, error) {
	var r io.Reader
	if body != nil {
		r = bytes.NewBuffer(body)
	}

	req, err := http.NewRequest(method, url, r)
	if err != nil {
		return nil, err
	}

	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	return m.httpDo(ctx, req)
}
