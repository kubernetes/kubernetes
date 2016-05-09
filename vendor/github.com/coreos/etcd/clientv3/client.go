// Copyright 2016 CoreOS, Inc.
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
	"errors"
	"io/ioutil"
	"log"
	"net"
	"net/url"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
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

	conn   *grpc.ClientConn
	cfg    Config
	creds  *credentials.TransportAuthenticator
	mu     sync.RWMutex // protects connection selection and error list
	errors []error      // errors passed to retryConnection

	ctx    context.Context
	cancel context.CancelFunc
}

// New creates a new etcdv3 client from a given configuration.
func New(cfg Config) (*Client, error) {
	if cfg.RetryDialer == nil {
		cfg.RetryDialer = dialEndpointList
	}
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
	c.mu.Lock()
	if c.cancel == nil {
		c.mu.Unlock()
		return nil
	}
	c.cancel()
	c.cancel = nil
	c.mu.Unlock()
	c.Watcher.Close()
	c.Lease.Close()
	return c.conn.Close()
}

// Ctx is a context for "out of band" messages (e.g., for sending
// "clean up" message when another context is canceled). It is
// canceled on client Close().
func (c *Client) Ctx() context.Context { return c.ctx }

// Endpoints lists the registered endpoints for the client.
func (c *Client) Endpoints() []string { return c.cfg.Endpoints }

// Errors returns all errors that have been observed since called last.
func (c *Client) Errors() (errs []error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	errs = c.errors
	c.errors = nil
	return errs
}

// Dial establishes a connection for a given endpoint using the client's config
func (c *Client) Dial(endpoint string) (*grpc.ClientConn, error) {
	opts := []grpc.DialOption{
		grpc.WithBlock(),
		grpc.WithTimeout(c.cfg.DialTimeout),
	}
	if c.creds != nil {
		opts = append(opts, grpc.WithTransportCredentials(*c.creds))
	} else {
		opts = append(opts, grpc.WithInsecure())
	}

	proto := "tcp"
	if url, uerr := url.Parse(endpoint); uerr == nil && url.Scheme == "unix" {
		proto = "unix"
		// strip unix:// prefix so certs work
		endpoint = url.Host
	}
	f := func(a string, t time.Duration) (net.Conn, error) {
		select {
		case <-c.ctx.Done():
			return nil, c.ctx.Err()
		default:
		}
		return net.DialTimeout(proto, a, t)
	}
	opts = append(opts, grpc.WithDialer(f))

	conn, err := grpc.Dial(endpoint, opts...)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

func newClient(cfg *Config) (*Client, error) {
	if cfg == nil {
		cfg = &Config{RetryDialer: dialEndpointList}
	}
	var creds *credentials.TransportAuthenticator
	if cfg.TLS != nil {
		c := credentials.NewTLS(cfg.TLS)
		creds = &c
	}
	// use a temporary skeleton client to bootstrap first connection
	ctx, cancel := context.WithCancel(context.TODO())
	conn, err := cfg.RetryDialer(&Client{cfg: *cfg, creds: creds, ctx: ctx})
	if err != nil {
		return nil, err
	}
	client := &Client{
		conn:   conn,
		cfg:    *cfg,
		creds:  creds,
		ctx:    ctx,
		cancel: cancel,
	}
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
func (c *Client) ActiveConnection() *grpc.ClientConn {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.conn
}

// retryConnection establishes a new connection
func (c *Client) retryConnection(oldConn *grpc.ClientConn, err error) (*grpc.ClientConn, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		c.errors = append(c.errors, err)
	}
	if c.cancel == nil {
		return nil, c.ctx.Err()
	}
	if oldConn != c.conn {
		// conn has already been updated
		return c.conn, nil
	}

	oldConn.Close()
	if st, _ := oldConn.State(); st != grpc.Shutdown {
		// wait for shutdown so grpc doesn't leak sleeping goroutines
		oldConn.WaitForStateChange(c.ctx, st)
	}

	conn, dialErr := c.cfg.RetryDialer(c)
	if dialErr != nil {
		c.errors = append(c.errors, dialErr)
		return nil, dialErr
	}
	c.conn = conn
	return c.conn, nil
}

// dialEndpointList attempts to connect to each endpoint in order until a
// connection is established.
func dialEndpointList(c *Client) (*grpc.ClientConn, error) {
	var err error
	for _, ep := range c.Endpoints() {
		conn, curErr := c.Dial(ep)
		if curErr != nil {
			err = curErr
		} else {
			return conn, nil
		}
	}
	return nil, err
}

// isHalted returns true if the given error and context indicate no forward
// progress can be made, even after reconnecting.
func isHalted(ctx context.Context, err error) bool {
	isRPCError := strings.HasPrefix(grpc.ErrorDesc(err), "etcdserver: ")
	return isRPCError || ctx.Err() != nil
}
