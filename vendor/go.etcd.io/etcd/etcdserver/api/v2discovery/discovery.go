// Copyright 2015 The etcd Authors
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

// Package v2discovery provides an implementation of the cluster discovery that
// is used by etcd with v2 client.
package v2discovery

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"

	"go.etcd.io/etcd/client"
	"go.etcd.io/etcd/pkg/transport"
	"go.etcd.io/etcd/pkg/types"

	"github.com/coreos/pkg/capnslog"
	"github.com/jonboulle/clockwork"
	"go.uber.org/zap"
)

var (
	plog = capnslog.NewPackageLogger("go.etcd.io/etcd", "discovery")

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
	nRetries             = uint(math.MaxUint32)
	maxExpoentialRetries = uint(8)
)

// JoinCluster will connect to the discovery service at the given url, and
// register the server represented by the given id and config to the cluster
func JoinCluster(lg *zap.Logger, durl, dproxyurl string, id types.ID, config string) (string, error) {
	d, err := newDiscovery(lg, durl, dproxyurl, id)
	if err != nil {
		return "", err
	}
	return d.joinCluster(config)
}

// GetCluster will connect to the discovery service at the given url and
// retrieve a string describing the cluster
func GetCluster(lg *zap.Logger, durl, dproxyurl string) (string, error) {
	d, err := newDiscovery(lg, durl, dproxyurl, 0)
	if err != nil {
		return "", err
	}
	return d.getCluster()
}

type discovery struct {
	lg      *zap.Logger
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
func newProxyFunc(lg *zap.Logger, proxy string) (func(*http.Request) (*url.URL, error), error) {
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

	if lg != nil {
		lg.Info("running proxy with discovery", zap.String("proxy-url", proxyURL.String()))
	} else {
		plog.Infof("using proxy %q", proxyURL.String())
	}
	return http.ProxyURL(proxyURL), nil
}

func newDiscovery(lg *zap.Logger, durl, dproxyurl string, id types.ID) (*discovery, error) {
	u, err := url.Parse(durl)
	if err != nil {
		return nil, err
	}
	token := u.Path
	u.Path = ""
	pf, err := newProxyFunc(lg, dproxyurl)
	if err != nil {
		return nil, err
	}

	// TODO: add ResponseHeaderTimeout back when watch on discovery service writes header early
	tr, err := transport.NewTransport(transport.TLSInfo{}, 30*time.Second)
	if err != nil {
		return nil, err
	}
	tr.Proxy = pf
	cfg := client.Config{
		Transport: tr,
		Endpoints: []string{u.String()},
	}
	c, err := client.New(cfg)
	if err != nil {
		return nil, err
	}
	dc := client.NewKeysAPIWithPrefix(c, "")
	return &discovery{
		lg:      lg,
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
			if d.lg != nil {
				d.lg.Warn(
					"failed to get from discovery server",
					zap.String("discovery-url", d.url.String()),
					zap.String("path", path.Join(configKey, "size")),
					zap.Error(err),
					zap.String("err-detail", ce.Detail()),
				)
			} else {
				plog.Error(ce.Detail())
			}
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
			if d.lg != nil {
				d.lg.Warn(
					"failed to get from discovery server",
					zap.String("discovery-url", d.url.String()),
					zap.String("path", d.cluster),
					zap.Error(err),
					zap.String("err-detail", ce.Detail()),
				)
			} else {
				plog.Error(ce.Detail())
			}
			return d.checkClusterRetry()
		}
		return nil, 0, 0, err
	}
	var nodes []*client.Node
	// append non-config keys to nodes
	for _, n := range resp.Node.Nodes {
		if path.Base(n.Key) != path.Base(configKey) {
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
	// logAndBackoffForRetry stops exponential backoff when the retries are more than maxExpoentialRetries and is set to a constant backoff afterward.
	retries := d.retries
	if retries > maxExpoentialRetries {
		retries = maxExpoentialRetries
	}
	retryTimeInSecond := time.Duration(0x1<<retries) * time.Second
	if d.lg != nil {
		d.lg.Info(
			"retry connecting to discovery service",
			zap.String("url", d.url.String()),
			zap.String("reason", step),
			zap.Duration("backoff", retryTimeInSecond),
		)
	} else {
		plog.Infof("%s: error connecting to %s, retrying in %s", step, d.url, retryTimeInSecond)
	}
	d.clock.Sleep(retryTimeInSecond)
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
			if d.lg != nil {
				d.lg.Info(
					"found self from discovery server",
					zap.String("discovery-url", d.url.String()),
					zap.String("self", path.Base(d.selfKey())),
				)
			} else {
				plog.Noticef("found self %s in the cluster", path.Base(d.selfKey()))
			}
		} else {
			if d.lg != nil {
				d.lg.Info(
					"found peer from discovery server",
					zap.String("discovery-url", d.url.String()),
					zap.String("peer", path.Base(n.Key)),
				)
			} else {
				plog.Noticef("found peer %s in the cluster", path.Base(n.Key))
			}
		}
	}

	// wait for others
	for len(all) < size {
		if d.lg != nil {
			d.lg.Info(
				"found peers from discovery server; waiting for more",
				zap.String("discovery-url", d.url.String()),
				zap.Int("found-peers", len(all)),
				zap.Int("needed-peers", size-len(all)),
			)
		} else {
			plog.Noticef("found %d peer(s), waiting for %d more", len(all), size-len(all))
		}
		resp, err := w.Next(context.Background())
		if err != nil {
			if ce, ok := err.(*client.ClusterError); ok {
				plog.Error(ce.Detail())
				return d.waitNodesRetry()
			}
			return nil, err
		}
		if d.lg != nil {
			d.lg.Info(
				"found peer from discovery server",
				zap.String("discovery-url", d.url.String()),
				zap.String("peer", path.Base(resp.Node.Key)),
			)
		} else {
			plog.Noticef("found peer %s in the cluster", path.Base(resp.Node.Key))
		}
		all = append(all, resp.Node)
	}
	if d.lg != nil {
		d.lg.Info(
			"found all needed peers from discovery server",
			zap.String("discovery-url", d.url.String()),
			zap.Int("found-peers", len(all)),
		)
	} else {
		plog.Noticef("found %d needed peer(s)", len(all))
	}
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
