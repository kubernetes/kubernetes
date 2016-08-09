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

package clientv3

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
)

var (
	ErrNoAvailableEndpoints = errors.New("etcdclient: no available endpoints")
)

// Client provides and manages an etcd v3 client session.
type Client struct {
	Cluster
	KV
	Lease
	Watcher
	Auth
	Maintenance

	conn  *grpc.ClientConn
	cfg   Config
	creds *credentials.TransportCredentials

	ctx    context.Context
	cancel context.CancelFunc

	// Username is a username for authentication
	Username string
	// Password is a password for authentication
	Password string
}

// New creates a new etcdv3 client from a given configuration.
func New(cfg Config) (*Client, error) {
	if len(cfg.Endpoints) == 0 {
		return nil, ErrNoAvailableEndpoints
	}

	return newClient(&cfg)
}

// NewFromURL creates a new etcdv3 client from a URL.
func NewFromURL(url string) (*Client, error) {
	return New(Config{Endpoints: []string{url}})
}

// NewFromConfigFile creates a new etcdv3 client from a configuration file.
func NewFromConfigFile(path string) (*Client, error) {
	cfg, err := configFromFile(path)
	if err != nil {
		return nil, err
	}
	return New(*cfg)
}

// Close shuts down the client's etcd connections.
func (c *Client) Close() error {
	c.cancel()
	return toErr(c.ctx, c.conn.Close())
}

// Ctx is a context for "out of band" messages (e.g., for sending
// "clean up" message when another context is canceled). It is
// canceled on client Close().
func (c *Client) Ctx() context.Context { return c.ctx }

// Endpoints lists the registered endpoints for the client.
func (c *Client) Endpoints() []string { return c.cfg.Endpoints }

type authTokenCredential struct {
	token string
}

func (cred authTokenCredential) RequireTransportSecurity() bool {
	return false
}

func (cred authTokenCredential) GetRequestMetadata(ctx context.Context, s ...string) (map[string]string, error) {
	return map[string]string{
		"token": cred.token,
	}, nil
}

func (c *Client) dialTarget(endpoint string) (proto string, host string, creds *credentials.TransportCredentials) {
	proto = "tcp"
	host = endpoint
	creds = c.creds
	url, uerr := url.Parse(endpoint)
	if uerr != nil || !strings.Contains(endpoint, "://") {
		return
	}
	// strip scheme:// prefix since grpc dials by host
	host = url.Host
	switch url.Scheme {
	case "unix":
		proto = "unix"
	case "http":
		creds = nil
	case "https":
		if creds != nil {
			break
		}
		tlsconfig := &tls.Config{}
		emptyCreds := credentials.NewTLS(tlsconfig)
		creds = &emptyCreds
	default:
		return "", "", nil
	}
	return
}

// dialSetupOpts gives the dial opts prioer to any authentication
func (c *Client) dialSetupOpts(endpoint string, dopts ...grpc.DialOption) []grpc.DialOption {
	opts := []grpc.DialOption{
		grpc.WithBlock(),
		grpc.WithTimeout(c.cfg.DialTimeout),
	}
	opts = append(opts, dopts...)

	// grpc issues TLS cert checks using the string passed into dial so
	// that string must be the host. To recover the full scheme://host URL,
	// have a map from hosts to the original endpoint.
	host2ep := make(map[string]string)
	for i := range c.cfg.Endpoints {
		_, host, _ := c.dialTarget(c.cfg.Endpoints[i])
		host2ep[host] = c.cfg.Endpoints[i]
	}

	f := func(host string, t time.Duration) (net.Conn, error) {
		proto, host, _ := c.dialTarget(host2ep[host])
		if proto == "" {
			return nil, fmt.Errorf("unknown scheme for %q", host)
		}
		select {
		case <-c.ctx.Done():
			return nil, c.ctx.Err()
		default:
		}
		return net.DialTimeout(proto, host, t)
	}
	opts = append(opts, grpc.WithDialer(f))

	_, _, creds := c.dialTarget(endpoint)
	if creds != nil {
		opts = append(opts, grpc.WithTransportCredentials(*creds))
	} else {
		opts = append(opts, grpc.WithInsecure())
	}

	return opts
}

// Dial connects to a single endpoint using the client's config.
func (c *Client) Dial(endpoint string) (*grpc.ClientConn, error) {
	return c.dial(endpoint)
}

func (c *Client) dial(endpoint string, dopts ...grpc.DialOption) (*grpc.ClientConn, error) {
	opts := c.dialSetupOpts(endpoint, dopts...)
	host := getHost(endpoint)
	if c.Username != "" && c.Password != "" {
		// use dial options without dopts to avoid reusing the client balancer
		auth, err := newAuthenticator(host, c.dialSetupOpts(endpoint))
		if err != nil {
			return nil, err
		}
		defer auth.close()

		resp, err := auth.authenticate(c.ctx, c.Username, c.Password)
		if err != nil {
			return nil, err
		}
		opts = append(opts, grpc.WithPerRPCCredentials(authTokenCredential{token: resp.Token}))
	}

	conn, err := grpc.Dial(host, opts...)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

// WithRequireLeader requires client requests to only succeed
// when the cluster has a leader.
func WithRequireLeader(ctx context.Context) context.Context {
	md := metadata.Pairs(rpctypes.MetadataRequireLeaderKey, rpctypes.MetadataHasLeader)
	return metadata.NewContext(ctx, md)
}

func newClient(cfg *Config) (*Client, error) {
	if cfg == nil {
		cfg = &Config{}
	}
	var creds *credentials.TransportCredentials
	if cfg.TLS != nil {
		c := credentials.NewTLS(cfg.TLS)
		creds = &c
	}

	// use a temporary skeleton client to bootstrap first connection
	ctx, cancel := context.WithCancel(context.TODO())
	client := &Client{
		conn:   nil,
		cfg:    *cfg,
		creds:  creds,
		ctx:    ctx,
		cancel: cancel,
	}
	if cfg.Username != "" && cfg.Password != "" {
		client.Username = cfg.Username
		client.Password = cfg.Password
	}

	b := newSimpleBalancer(cfg.Endpoints)
	conn, err := client.dial(cfg.Endpoints[0], grpc.WithBalancer(b))
	if err != nil {
		return nil, err
	}
	client.conn = conn

	client.Cluster = NewCluster(client)
	client.KV = NewKV(client)
	client.Lease = NewLease(client)
	client.Watcher = NewWatcher(client)
	client.Auth = NewAuth(client)
	client.Maintenance = NewMaintenance(client)
	if cfg.Logger != nil {
		logger.Set(cfg.Logger)
	} else {
		// disable client side grpc by default
		logger.Set(log.New(ioutil.Discard, "", 0))
	}

	return client, nil
}

// ActiveConnection returns the current in-use connection
func (c *Client) ActiveConnection() *grpc.ClientConn { return c.conn }

// isHaltErr returns true if the given error and context indicate no forward
// progress can be made, even after reconnecting.
func isHaltErr(ctx context.Context, err error) bool {
	if ctx != nil && ctx.Err() != nil {
		return true
	}
	if err == nil {
		return false
	}
	eErr := rpctypes.Error(err)
	if _, ok := eErr.(rpctypes.EtcdError); ok {
		return eErr != rpctypes.ErrStopped && eErr != rpctypes.ErrNoLeader
	}
	// treat etcdserver errors not recognized by the client as halting
	return strings.Contains(err.Error(), grpc.ErrClientConnClosing.Error()) ||
		strings.Contains(err.Error(), "etcdserver:")
}

func toErr(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}
	err = rpctypes.Error(err)
	if ctx.Err() != nil && strings.Contains(err.Error(), "context") {
		err = ctx.Err()
	} else if strings.Contains(err.Error(), grpc.ErrClientConnClosing.Error()) {
		err = grpc.ErrClientConnClosing
	}
	return err
}
