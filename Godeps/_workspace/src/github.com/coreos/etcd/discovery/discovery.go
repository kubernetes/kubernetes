// Copyright 2015 CoreOS, Inc.
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

// Package discovery provides an implementation of the cluster discovery that
// is used by etcd.
package discovery

import (
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/coreos/etcd/client"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/pkg/capnslog"
	"github.com/jonboulle/clockwork"
	"golang.org/x/net/context"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "discovery")

	ErrInvalidURL           = errors.New("discovery: invalid URL")
	ErrBadSizeKey           = errors.New("discovery: size key is bad")
	ErrSizeNotFound         = errors.New("discovery: size key not found")
	ErrTokenNotFound        = errors.New("discovery: token not found")
	ErrDuplicateID          = errors.New("discovery: found duplicate id")
	ErrDuplicateName        = errors.New("discovery: found duplicate name")
	ErrFullCluster          = errors.New("discovery: cluster is full")
	ErrTooManyRetries       = errors.New("discovery: too many retries")
	ErrBadDiscoveryEndpoint = errors.New("discovery: bad discovery endpoint")
)

var (
	// Number of retries discovery will attempt before giving up and erroring out.
	nRetries = uint(math.MaxUint32)
)

// JoinCluster will connect to the discovery service at the given url, and
// register the server represented by the given id and config to the cluster
func JoinCluster(durl, dproxyurl string, id types.ID, config string) (string, error) {
	d, err := newDiscovery(durl, dproxyurl, id)
	if err != nil {
		return "", err
	}
	return d.joinCluster(config)
}

// GetCluster will connect to the discovery service at the given url and
// retrieve a string describing the cluster
func GetCluster(durl, dproxyurl string) (string, error) {
	d, err := newDiscovery(durl, dproxyurl, 0)
	if err != nil {
		return "", err
	}
	return d.getCluster()
}

type discovery struct {
	cluster string
	id      types.ID
	c       client.KeysAPI
	retries uint
	url     *url.URL

	clock clockwork.Clock
}

// newProxyFunc builds a proxy function from the given string, which should
// represent a URL that can be used as a proxy. It performs basic
// sanitization of the URL and returns any error encountered.
func newProxyFunc(proxy string) (func(*http.Request) (*url.URL, error), error) {
	if proxy == "" {
		return nil, nil
	}
	// Do a small amount of URL sanitization to help the user
	// Derived from net/http.ProxyFromEnvironment
	proxyURL, err := url.Parse(proxy)
	if err != nil || !strings.HasPrefix(proxyURL.Scheme, "http") {
		// proxy was bogus. Try prepending "http://" to it and
		// see if that parses correctly. If not, we ignore the
		// error and complain about the original one
		var err2 error
		proxyURL, err2 = url.Parse("http://" + proxy)
		if err2 == nil {
			err = nil
		}
	}
	if err != nil {
		return nil, fmt.Errorf("invalid proxy address %q: %v", proxy, err)
	}

	plog.Infof("using proxy %q", proxyURL.String())
	return http.ProxyURL(proxyURL), nil
}

func newDiscovery(durl, dproxyurl string, id types.ID) (*discovery, error) {
	u, err := url.Parse(durl)
	if err != nil {
		return nil, err
	}
	token := u.Path
	u.Path = ""
	pf, err := newProxyFunc(dproxyurl)
	if err != nil {
		return nil, err
	}
	cfg := client.Config{
		Transport: &http.Transport{
			Proxy: pf,
			Dial: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).Dial,
			TLSHandshakeTimeout: 10 * time.Second,
			// TODO: add ResponseHeaderTimeout back when watch on discovery service writes header early
		},
		Endpoints: []string{u.String()},
	}
	c, err := client.New(cfg)
	if err != nil {
		return nil, err
	}
	dc := client.NewKeysAPIWithPrefix(c, "")
	return &discovery{
		cluster: token,
		c:       dc,
		id:      id,
		url:     u,
		clock:   clockwork.NewRealClock(),
	}, nil
}

func (d *discovery) joinCluster(config string) (string, error) {
	// fast path: if the cluster is full, return the error
	// do not need to register to the cluster in this case.
	if _, _, _, err := d.checkCluster(); err != nil {
		return "", err
	}

	if err := d.createSelf(config); err != nil {
		// Fails, even on a timeout, if createSelf times out.
		// TODO(barakmich): Retrying the same node might want to succeed here
		// (ie, createSelf should be idempotent for discovery).
		return "", err
	}

	nodes, size, index, err := d.checkCluster()
	if err != nil {
		return "", err
	}

	all, err := d.waitNodes(nodes, size, index)
	if err != nil {
		return "", err
	}

	return nodesToCluster(all, size)
}

func (d *discovery) getCluster() (string, error) {
	nodes, size, index, err := d.checkCluster()
	if err != nil {
		if err == ErrFullCluster {
			return nodesToCluster(nodes, size)
		}
		return "", err
	}

	all, err := d.waitNodes(nodes, size, index)
	if err != nil {
		return "", err
	}
	return nodesToCluster(all, size)
}

func (d *discovery) createSelf(contents string) error {
	ctx, cancel := context.WithTimeout(context.Background(), client.DefaultRequestTimeout)
	resp, err := d.c.Create(ctx, d.selfKey(), contents)
	cancel()
	if err != nil {
		if eerr, ok := err.(client.Error); ok && eerr.Code == client.ErrorCodeNodeExist {
			return ErrDuplicateID
		}
		return err
	}

	// ensure self appears on the server we connected to
	w := d.c.Watcher(d.selfKey(), &client.WatcherOptions{AfterIndex: resp.Node.CreatedIndex - 1})
	_, err = w.Next(context.Background())
	return err
}

func (d *discovery) checkCluster() ([]*client.Node, int, uint64, error) {
	configKey := path.Join("/", d.cluster, "_config")
	ctx, cancel := context.WithTimeout(context.Background(), client.DefaultRequestTimeout)
	// find cluster size
	resp, err := d.c.Get(ctx, path.Join(configKey, "size"), nil)
	cancel()
	if err != nil {
		if eerr, ok := err.(*client.Error); ok && eerr.Code == client.ErrorCodeKeyNotFound {
			return nil, 0, 0, ErrSizeNotFound
		}
		if err == client.ErrInvalidJSON {
			return nil, 0, 0, ErrBadDiscoveryEndpoint
		}
		if ce, ok := err.(*client.ClusterError); ok {
			plog.Error(ce.Detail())
			return d.checkClusterRetry()
		}
		return nil, 0, 0, err
	}
	size, err := strconv.Atoi(resp.Node.Value)
	if err != nil {
		return nil, 0, 0, ErrBadSizeKey
	}

	ctx, cancel = context.WithTimeout(context.Background(), client.DefaultRequestTimeout)
	resp, err = d.c.Get(ctx, d.cluster, nil)
	cancel()
	if err != nil {
		if ce, ok := err.(*client.ClusterError); ok {
			plog.Error(ce.Detail())
			return d.checkClusterRetry()
		}
		return nil, 0, 0, err
	}
	nodes := make([]*client.Node, 0)
	// append non-config keys to nodes
	for _, n := range resp.Node.Nodes {
		if !(path.Base(n.Key) == path.Base(configKey)) {
			nodes = append(nodes, n)
		}
	}

	snodes := sortableNodes{nodes}
	sort.Sort(snodes)

	// find self position
	for i := range nodes {
		if path.Base(nodes[i].Key) == path.Base(d.selfKey()) {
			break
		}
		if i >= size-1 {
			return nodes[:size], size, resp.Index, ErrFullCluster
		}
	}
	return nodes, size, resp.Index, nil
}

func (d *discovery) logAndBackoffForRetry(step string) {
	d.retries++
	retryTime := time.Second * (0x1 << d.retries)
	plog.Infof("%s: error connecting to %s, retrying in %s", step, d.url, retryTime)
	d.clock.Sleep(retryTime)
}

func (d *discovery) checkClusterRetry() ([]*client.Node, int, uint64, error) {
	if d.retries < nRetries {
		d.logAndBackoffForRetry("cluster status check")
		return d.checkCluster()
	}
	return nil, 0, 0, ErrTooManyRetries
}

func (d *discovery) waitNodesRetry() ([]*client.Node, error) {
	if d.retries < nRetries {
		d.logAndBackoffForRetry("waiting for other nodes")
		nodes, n, index, err := d.checkCluster()
		if err != nil {
			return nil, err
		}
		return d.waitNodes(nodes, n, index)
	}
	return nil, ErrTooManyRetries
}

func (d *discovery) waitNodes(nodes []*client.Node, size int, index uint64) ([]*client.Node, error) {
	if len(nodes) > size {
		nodes = nodes[:size]
	}
	// watch from the next index
	w := d.c.Watcher(d.cluster, &client.WatcherOptions{AfterIndex: index, Recursive: true})
	all := make([]*client.Node, len(nodes))
	copy(all, nodes)
	for _, n := range all {
		if path.Base(n.Key) == path.Base(d.selfKey()) {
			plog.Noticef("found self %s in the cluster", path.Base(d.selfKey()))
		} else {
			plog.Noticef("found peer %s in the cluster", path.Base(n.Key))
		}
	}

	// wait for others
	for len(all) < size {
		plog.Noticef("found %d peer(s), waiting for %d more", len(all), size-len(all))
		resp, err := w.Next(context.Background())
		if err != nil {
			if ce, ok := err.(*client.ClusterError); ok {
				plog.Error(ce.Detail())
				return d.waitNodesRetry()
			}
			return nil, err
		}
		plog.Noticef("found peer %s in the cluster", path.Base(resp.Node.Key))
		all = append(all, resp.Node)
	}
	plog.Noticef("found %d needed peer(s)", len(all))
	return all, nil
}

func (d *discovery) selfKey() string {
	return path.Join("/", d.cluster, d.id.String())
}

func nodesToCluster(ns []*client.Node, size int) (string, error) {
	s := make([]string, len(ns))
	for i, n := range ns {
		s[i] = n.Value
	}
	us := strings.Join(s, ",")
	m, err := types.NewURLsMap(us)
	if err != nil {
		return us, ErrInvalidURL
	}
	if m.Len() != size {
		return us, ErrDuplicateName
	}
	return us, nil
}

type sortableNodes struct{ Nodes []*client.Node }

func (ns sortableNodes) Len() int { return len(ns.Nodes) }
func (ns sortableNodes) Less(i, j int) bool {
	return ns.Nodes[i].CreatedIndex < ns.Nodes[j].CreatedIndex
}
func (ns sortableNodes) Swap(i, j int) { ns.Nodes[i], ns.Nodes[j] = ns.Nodes[j], ns.Nodes[i] }
