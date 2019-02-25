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
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

var (
	ErrNoAvailableEndpoints = errors.New("etcdclient: no available endpoints")
	ErrOldCluster           = errors.New("etcdclient: old cluster version")
)

// Client provides and manages an etcd v3 client session.
type Client struct {
	Cluster
	KV
	Lease
	Watcher
	Auth
	Maintenance

	conn     *grpc.ClientConn
	dialerrc chan error

	cfg      Config
	creds    *credentials.TransportCredentials
	balancer *healthBalancer
	mu       *sync.Mutex

	ctx    context.Context
	cancel context.CancelFunc

	// Username is a user name for authentication.
	Username string
	// Password is a password for authentication.
	Password string
	// tokenCred is an instance of WithPerRPCCredentials()'s argument
	tokenCred *authTokenCredential

	callOpts []grpc.CallOption
}

// New creates a new etcdv3 client from a given configuration.
func New(cfg Config) (*Client, error) {
	if len(cfg.Endpoints) == 0 {
		return nil, ErrNoAvailableEndpoints
	}

	return newClient(&cfg)
}

// NewCtxClient creates a client with a context but no underlying grpc
// connection. This is useful for embedded cases that override the
// service interface implementations and do not need connection management.
func NewCtxClient(ctx context.Context) *Client {
	cctx, cancel := context.WithCancel(ctx)
	return &Client{ctx: cctx, cancel: cancel}
}

// NewFromURL creates a new etcdv3 client from a URL.
func NewFromURL(url string) (*Client, error) {
	return New(Config{Endpoints: []string{url}})
}

// Close shuts down the client's etcd connections.
func (c *Client) Close() error {
	c.cancel()
	c.Watcher.Close()
	c.Lease.Close()
	if c.conn != nil {
		return toErr(c.ctx, c.conn.Close())
	}
	return c.ctx.Err()
}

// Ctx is a context for "out of band" messages (e.g., for sending
// "clean up" message when another context is canceled). It is
// canceled on client Close().
func (c *Client) Ctx() context.Context { return c.ctx }

// Endpoints lists the registered endpoints for the client.
func (c *Client) Endpoints() (eps []string) {
	// copy the slice; protect original endpoints from being changed
	eps = make([]string, len(c.cfg.Endpoints))
	copy(eps, c.cfg.Endpoints)
	return
}

// SetEndpoints updates client's endpoints.
func (c *Client) SetEndpoints(eps ...string) {
	c.mu.Lock()
	c.cfg.Endpoints = eps
	c.mu.Unlock()
	c.balancer.updateAddrs(eps...)

	// updating notifyCh can trigger new connections,
	// need update addrs if all connections are down
	// or addrs does not include pinAddr.
	c.balancer.mu.RLock()
	update := !hasAddr(c.balancer.addrs, c.balancer.pinAddr)
	c.balancer.mu.RUnlock()
	if update {
		select {
		case c.balancer.updateAddrsC <- notifyNext:
		case <-c.balancer.stopc:
		}
	}
}

// Sync synchronizes client's endpoints with the known endpoints from the etcd membership.
func (c *Client) Sync(ctx context.Context) error {
	mresp, err := c.MemberList(ctx)
	if err != nil {
		return err
	}
	var eps []string
	for _, m := range mresp.Members {
		eps = append(eps, m.ClientURLs...)
	}
	c.SetEndpoints(eps...)
	return nil
}

func (c *Client) autoSync() {
	if c.cfg.AutoSyncInterval == time.Duration(0) {
		return
	}

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-time.After(c.cfg.AutoSyncInterval):
			ctx, cancel := context.WithTimeout(c.ctx, 5*time.Second)
			err := c.Sync(ctx)
			cancel()
			if err != nil && err != c.ctx.Err() {
				logger.Println("Auto sync endpoints failed:", err)
			}
		}
	}
}

type authTokenCredential struct {
	token   string
	tokenMu *sync.RWMutex
}

func (cred authTokenCredential) RequireTransportSecurity() bool {
	return false
}

func (cred authTokenCredential) GetRequestMetadata(ctx context.Context, s ...string) (map[string]string, error) {
	cred.tokenMu.RLock()
	defer cred.tokenMu.RUnlock()
	return map[string]string{
		"token": cred.token,
	}, nil
}

func parseEndpoint(endpoint string) (proto string, host string, scheme string) {
	proto = "tcp"
	host = endpoint
	url, uerr := url.Parse(endpoint)
	if uerr != nil || !strings.Contains(endpoint, "://") {
		return proto, host, scheme
	}
	scheme = url.Scheme

	// strip scheme:// prefix since grpc dials by host
	host = url.Host
	switch url.Scheme {
	case "http", "https":
	case "unix", "unixs":
		proto = "unix"
		host = url.Host + url.Path
	default:
		proto, host = "", ""
	}
	return proto, host, scheme
}

func (c *Client) processCreds(scheme string) (creds *credentials.TransportCredentials) {
	creds = c.creds
	switch scheme {
	case "unix":
	case "http":
		creds = nil
	case "https", "unixs":
		if creds != nil {
			break
		}
		tlsconfig := &tls.Config{}
		emptyCreds := credentials.NewTLS(tlsconfig)
		creds = &emptyCreds
	default:
		creds = nil
	}
	return creds
}

// dialSetupOpts gives the dial opts prior to any authentication
func (c *Client) dialSetupOpts(endpoint string, dopts ...grpc.DialOption) (opts []grpc.DialOption) {
	if c.cfg.DialTimeout > 0 {
		opts = []grpc.DialOption{grpc.WithTimeout(c.cfg.DialTimeout)}
	}
	if c.cfg.DialKeepAliveTime > 0 {
		params := keepalive.ClientParameters{
			Time:    c.cfg.DialKeepAliveTime,
			Timeout: c.cfg.DialKeepAliveTimeout,
		}
		opts = append(opts, grpc.WithKeepaliveParams(params))
	}
	opts = append(opts, dopts...)

	f := func(host string, t time.Duration) (net.Conn, error) {
		proto, host, _ := parseEndpoint(c.balancer.endpoint(host))
		if host == "" && endpoint != "" {
			// dialing an endpoint not in the balancer; use
			// endpoint passed into dial
			proto, host, _ = parseEndpoint(endpoint)
		}
		if proto == "" {
			return nil, fmt.Errorf("unknown scheme for %q", host)
		}
		select {
		case <-c.ctx.Done():
			return nil, c.ctx.Err()
		default:
		}
		dialer := &net.Dialer{Timeout: t}
		conn, err := dialer.DialContext(c.ctx, proto, host)
		if err != nil {
			select {
			case c.dialerrc <- err:
			default:
			}
		}
		return conn, err
	}
	opts = append(opts, grpc.WithDialer(f))

	creds := c.creds
	if _, _, scheme := parseEndpoint(endpoint); len(scheme) != 0 {
		creds = c.processCreds(scheme)
	}
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

func (c *Client) getToken(ctx context.Context) error {
	var err error // return last error in a case of fail
	var auth *authenticator

	for i := 0; i < len(c.cfg.Endpoints); i++ {
		endpoint := c.cfg.Endpoints[i]
		host := getHost(endpoint)
		// use dial options without dopts to avoid reusing the client balancer
		auth, err = newAuthenticator(host, c.dialSetupOpts(endpoint), c)
		if err != nil {
			continue
		}
		defer auth.close()

		var resp *AuthenticateResponse
		resp, err = auth.authenticate(ctx, c.Username, c.Password)
		if err != nil {
			continue
		}

		c.tokenCred.tokenMu.Lock()
		c.tokenCred.token = resp.Token
		c.tokenCred.tokenMu.Unlock()

		return nil
	}

	return err
}

func (c *Client) dial(endpoint string, dopts ...grpc.DialOption) (*grpc.ClientConn, error) {
	opts := c.dialSetupOpts(endpoint, dopts...)
	host := getHost(endpoint)
	if c.Username != "" && c.Password != "" {
		c.tokenCred = &authTokenCredential{
			tokenMu: &sync.RWMutex{},
		}

		ctx := c.ctx
		if c.cfg.DialTimeout > 0 {
			cctx, cancel := context.WithTimeout(ctx, c.cfg.DialTimeout)
			defer cancel()
			ctx = cctx
		}

		err := c.getToken(ctx)
		if err != nil {
			if toErr(ctx, err) != rpctypes.ErrAuthNotEnabled {
				if err == ctx.Err() && ctx.Err() != c.ctx.Err() {
					err = context.DeadlineExceeded
				}
				return nil, err
			}
		} else {
			opts = append(opts, grpc.WithPerRPCCredentials(c.tokenCred))
		}
	}

	opts = append(opts, c.cfg.DialOptions...)

	conn, err := grpc.DialContext(c.ctx, host, opts...)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

// WithRequireLeader requires client requests to only succeed
// when the cluster has a leader.
func WithRequireLeader(ctx context.Context) context.Context {
	md := metadata.Pairs(rpctypes.MetadataRequireLeaderKey, rpctypes.MetadataHasLeader)
	return metadata.NewOutgoingContext(ctx, md)
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
	baseCtx := context.TODO()
	if cfg.Context != nil {
		baseCtx = cfg.Context
	}

	ctx, cancel := context.WithCancel(baseCtx)
	client := &Client{
		conn:     nil,
		dialerrc: make(chan error, 1),
		cfg:      *cfg,
		creds:    creds,
		ctx:      ctx,
		cancel:   cancel,
		mu:       new(sync.Mutex),
		callOpts: defaultCallOpts,
	}
	if cfg.Username != "" && cfg.Password != "" {
		client.Username = cfg.Username
		client.Password = cfg.Password
	}
	if cfg.MaxCallSendMsgSize > 0 || cfg.MaxCallRecvMsgSize > 0 {
		if cfg.MaxCallRecvMsgSize > 0 && cfg.MaxCallSendMsgSize > cfg.MaxCallRecvMsgSize {
			return nil, fmt.Errorf("gRPC message recv limit (%d bytes) must be greater than send limit (%d bytes)", cfg.MaxCallRecvMsgSize, cfg.MaxCallSendMsgSize)
		}
		callOpts := []grpc.CallOption{
			defaultFailFast,
			defaultMaxCallSendMsgSize,
			defaultMaxCallRecvMsgSize,
		}
		if cfg.MaxCallSendMsgSize > 0 {
			callOpts[1] = grpc.MaxCallSendMsgSize(cfg.MaxCallSendMsgSize)
		}
		if cfg.MaxCallRecvMsgSize > 0 {
			callOpts[2] = grpc.MaxCallRecvMsgSize(cfg.MaxCallRecvMsgSize)
		}
		client.callOpts = callOpts
	}

	client.balancer = newHealthBalancer(cfg.Endpoints, cfg.DialTimeout, func(ep string) (bool, error) {
		return grpcHealthCheck(client, ep)
	})

	// use Endpoints[0] so that for https:// without any tls config given, then
	// grpc will assume the certificate server name is the endpoint host.
	conn, err := client.dial(cfg.Endpoints[0], grpc.WithBalancer(client.balancer))
	if err != nil {
		client.cancel()
		client.balancer.Close()
		return nil, err
	}
	client.conn = conn

	// wait for a connection
	if cfg.DialTimeout > 0 {
		hasConn := false
		waitc := time.After(cfg.DialTimeout)
		select {
		case <-client.balancer.ready():
			hasConn = true
		case <-ctx.Done():
		case <-waitc:
		}
		if !hasConn {
			err := context.DeadlineExceeded
			select {
			case err = <-client.dialerrc:
			default:
			}
			client.cancel()
			client.balancer.Close()
			conn.Close()
			return nil, err
		}
	}

	client.Cluster = NewCluster(client)
	client.KV = NewKV(client)
	client.Lease = NewLease(client)
	client.Watcher = NewWatcher(client)
	client.Auth = NewAuth(client)
	client.Maintenance = NewMaintenance(client)

	if cfg.RejectOldCluster {
		if err := client.checkVersion(); err != nil {
			client.Close()
			return nil, err
		}
	}

	go client.autoSync()
	return client, nil
}

func (c *Client) checkVersion() (err error) {
	var wg sync.WaitGroup
	errc := make(chan error, len(c.cfg.Endpoints))
	ctx, cancel := context.WithCancel(c.ctx)
	if c.cfg.DialTimeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, c.cfg.DialTimeout)
	}
	wg.Add(len(c.cfg.Endpoints))
	for _, ep := range c.cfg.Endpoints {
		// if cluster is current, any endpoint gives a recent version
		go func(e string) {
			defer wg.Done()
			resp, rerr := c.Status(ctx, e)
			if rerr != nil {
				errc <- rerr
				return
			}
			vs := strings.Split(resp.Version, ".")
			maj, min := 0, 0
			if len(vs) >= 2 {
				maj, _ = strconv.Atoi(vs[0])
				min, rerr = strconv.Atoi(vs[1])
			}
			if maj < 3 || (maj == 3 && min < 2) {
				rerr = ErrOldCluster
			}
			errc <- rerr
		}(ep)
	}
	// wait for success
	for i := 0; i < len(c.cfg.Endpoints); i++ {
		if err = <-errc; err == nil {
			break
		}
	}
	cancel()
	wg.Wait()
	return err
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
	ev, _ := status.FromError(err)
	// Unavailable codes mean the system will be right back.
	// (e.g., can't connect, lost leader)
	// Treat Internal codes as if something failed, leaving the
	// system in an inconsistent state, but retrying could make progress.
	// (e.g., failed in middle of send, corrupted frame)
	// TODO: are permanent Internal errors possible from grpc?
	return ev.Code() != codes.Unavailable && ev.Code() != codes.Internal
}

// isUnavailableErr returns true if the given error is an unavailable error
func isUnavailableErr(ctx context.Context, err error) bool {
	if ctx != nil && ctx.Err() != nil {
		return false
	}
	if err == nil {
		return false
	}
	ev, _ := status.FromError(err)
	// Unavailable codes mean the system will be right back.
	// (e.g., can't connect, lost leader)
	return ev.Code() == codes.Unavailable
}

func toErr(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}
	err = rpctypes.Error(err)
	if _, ok := err.(rpctypes.EtcdError); ok {
		return err
	}
	ev, _ := status.FromError(err)
	code := ev.Code()
	switch code {
	case codes.DeadlineExceeded:
		fallthrough
	case codes.Canceled:
		if ctx.Err() != nil {
			err = ctx.Err()
		}
	case codes.Unavailable:
	case codes.FailedPrecondition:
		err = grpc.ErrClientConnClosing
	}
	return err
}

func canceledByCaller(stopCtx context.Context, err error) bool {
	if stopCtx.Err() == nil || err == nil {
		return false
	}

	return err == context.Canceled || err == context.DeadlineExceeded
}
