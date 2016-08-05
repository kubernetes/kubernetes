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

package integration

import (
	"crypto/tls"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"

	"github.com/coreos/etcd/client"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http"
	"github.com/coreos/etcd/etcdserver/api/v3rpc"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/rafthttp"
)

const (
	tickDuration   = 10 * time.Millisecond
	clusterName    = "etcd"
	requestTimeout = 20 * time.Second
)

var (
	electionTicks = 10

	// integration test uses well-known ports to listen for each running member,
	// which ensures restarted member could listen on specific port again.
	nextListenPort int64 = 20000

	testTLSInfo = transport.TLSInfo{
		KeyFile:        "./fixtures/server.key.insecure",
		CertFile:       "./fixtures/server.crt",
		TrustedCAFile:  "./fixtures/ca.crt",
		ClientCertAuth: true,
	}
)

type ClusterConfig struct {
	Size         int
	PeerTLS      *transport.TLSInfo
	ClientTLS    *transport.TLSInfo
	DiscoveryURL string
	UseGRPC      bool
}

type cluster struct {
	cfg     *ClusterConfig
	Members []*member
}

func (c *cluster) fillClusterForMembers() error {
	if c.cfg.DiscoveryURL != "" {
		// cluster will be discovered
		return nil
	}

	addrs := make([]string, 0)
	for _, m := range c.Members {
		scheme := "http"
		if m.PeerTLSInfo != nil {
			scheme = "https"
		}
		for _, l := range m.PeerListeners {
			addrs = append(addrs, fmt.Sprintf("%s=%s://%s", m.Name, scheme, l.Addr().String()))
		}
	}
	clusterStr := strings.Join(addrs, ",")
	var err error
	for _, m := range c.Members {
		m.InitialPeerURLsMap, err = types.NewURLsMap(clusterStr)
		if err != nil {
			return err
		}
	}
	return nil
}

func newCluster(t *testing.T, cfg *ClusterConfig) *cluster {
	c := &cluster{cfg: cfg}
	ms := make([]*member, cfg.Size)
	for i := 0; i < cfg.Size; i++ {
		ms[i] = c.mustNewMember(t)
	}
	c.Members = ms
	if err := c.fillClusterForMembers(); err != nil {
		t.Fatal(err)
	}

	return c
}

// NewCluster returns an unlaunched cluster of the given size which has been
// set to use static bootstrap.
func NewCluster(t *testing.T, size int) *cluster {
	return newCluster(t, &ClusterConfig{Size: size})
}

// NewClusterByConfig returns an unlaunched cluster defined by a cluster configuration
func NewClusterByConfig(t *testing.T, cfg *ClusterConfig) *cluster {
	return newCluster(t, cfg)
}

func (c *cluster) Launch(t *testing.T) {
	errc := make(chan error)
	for _, m := range c.Members {
		// Members are launched in separate goroutines because if they boot
		// using discovery url, they have to wait for others to register to continue.
		go func(m *member) {
			errc <- m.Launch()
		}(m)
	}
	for range c.Members {
		if err := <-errc; err != nil {
			t.Fatalf("error setting up member: %v", err)
		}
	}
	// wait cluster to be stable to receive future client requests
	c.waitMembersMatch(t, c.HTTPMembers())
	c.waitVersion()
}

func (c *cluster) URL(i int) string {
	return c.Members[i].ClientURLs[0].String()
}

// URLs returns a list of all active client URLs in the cluster
func (c *cluster) URLs() []string {
	urls := make([]string, 0)
	for _, m := range c.Members {
		select {
		case <-m.s.StopNotify():
			continue
		default:
		}
		for _, u := range m.ClientURLs {
			urls = append(urls, u.String())
		}
	}
	return urls
}

// HTTPMembers returns a list of all active members as client.Members
func (c *cluster) HTTPMembers() []client.Member {
	ms := []client.Member{}
	for _, m := range c.Members {
		pScheme, cScheme := "http", "http"
		if m.PeerTLSInfo != nil {
			pScheme = "https"
		}
		if m.ClientTLSInfo != nil {
			cScheme = "https"
		}
		cm := client.Member{Name: m.Name}
		for _, ln := range m.PeerListeners {
			cm.PeerURLs = append(cm.PeerURLs, pScheme+"://"+ln.Addr().String())
		}
		for _, ln := range m.ClientListeners {
			cm.ClientURLs = append(cm.ClientURLs, cScheme+"://"+ln.Addr().String())
		}
		ms = append(ms, cm)
	}
	return ms
}

func (c *cluster) mustNewMember(t *testing.T) *member {
	name := c.name(rand.Int())
	m := mustNewMember(t, name, c.cfg.PeerTLS, c.cfg.ClientTLS)
	m.DiscoveryURL = c.cfg.DiscoveryURL
	if c.cfg.UseGRPC {
		if err := m.listenGRPC(); err != nil {
			t.Fatal(err)
		}
	}
	return m
}

func (c *cluster) addMember(t *testing.T) {
	m := c.mustNewMember(t)

	scheme := "http"
	if c.cfg.PeerTLS != nil {
		scheme = "https"
	}

	// send add request to the cluster
	var err error
	for i := 0; i < len(c.Members); i++ {
		clientURL := c.URL(i)
		peerURL := scheme + "://" + m.PeerListeners[0].Addr().String()
		if err = c.addMemberByURL(t, clientURL, peerURL); err == nil {
			break
		}
	}
	if err != nil {
		t.Fatalf("add member failed on all members error: %v", err)
	}

	m.InitialPeerURLsMap = types.URLsMap{}
	for _, mm := range c.Members {
		m.InitialPeerURLsMap[mm.Name] = mm.PeerURLs
	}
	m.InitialPeerURLsMap[m.Name] = m.PeerURLs
	m.NewCluster = false
	if err := m.Launch(); err != nil {
		t.Fatal(err)
	}
	c.Members = append(c.Members, m)
	// wait cluster to be stable to receive future client requests
	c.waitMembersMatch(t, c.HTTPMembers())
}

func (c *cluster) addMemberByURL(t *testing.T, clientURL, peerURL string) error {
	cc := mustNewHTTPClient(t, []string{clientURL}, c.cfg.ClientTLS)
	ma := client.NewMembersAPI(cc)
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	if _, err := ma.Add(ctx, peerURL); err != nil {
		return err
	}
	cancel()

	// wait for the add node entry applied in the cluster
	members := append(c.HTTPMembers(), client.Member{PeerURLs: []string{peerURL}, ClientURLs: []string{}})
	c.waitMembersMatch(t, members)
	return nil
}

func (c *cluster) AddMember(t *testing.T) {
	c.addMember(t)
}

func (c *cluster) RemoveMember(t *testing.T, id uint64) {
	// send remove request to the cluster
	cc := mustNewHTTPClient(t, c.URLs(), c.cfg.ClientTLS)
	ma := client.NewMembersAPI(cc)
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	if err := ma.Remove(ctx, types.ID(id).String()); err != nil {
		t.Fatalf("unexpected remove error %v", err)
	}
	cancel()
	newMembers := make([]*member, 0)
	for _, m := range c.Members {
		if uint64(m.s.ID()) != id {
			newMembers = append(newMembers, m)
		} else {
			select {
			case <-m.s.StopNotify():
				m.Terminate(t)
			// 1s stop delay + election timeout + 1s disk and network delay + connection write timeout
			// TODO: remove connection write timeout by selecting on http response closeNotifier
			// blocking on https://github.com/golang/go/issues/9524
			case <-time.After(time.Second + time.Duration(electionTicks)*tickDuration + time.Second + rafthttp.ConnWriteTimeout):
				t.Fatalf("failed to remove member %s in time", m.s.ID())
			}
		}
	}
	c.Members = newMembers
	c.waitMembersMatch(t, c.HTTPMembers())
}

func (c *cluster) Terminate(t *testing.T) {
	for _, m := range c.Members {
		m.Terminate(t)
	}
}

func (c *cluster) waitMembersMatch(t *testing.T, membs []client.Member) {
	for _, u := range c.URLs() {
		cc := mustNewHTTPClient(t, []string{u}, c.cfg.ClientTLS)
		ma := client.NewMembersAPI(cc)
		for {
			ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
			ms, err := ma.List(ctx)
			cancel()
			if err == nil && isMembersEqual(ms, membs) {
				break
			}
			time.Sleep(tickDuration)
		}
	}
	return
}

func (c *cluster) waitLeader(t *testing.T, membs []*member) int {
	possibleLead := make(map[uint64]bool)
	var lead uint64
	for _, m := range membs {
		possibleLead[uint64(m.s.ID())] = true
	}

	for lead == 0 || !possibleLead[lead] {
		lead = 0
		for _, m := range membs {
			select {
			case <-m.s.StopNotify():
				continue
			default:
			}
			if lead != 0 && lead != m.s.Lead() {
				lead = 0
				break
			}
			lead = m.s.Lead()
		}
		time.Sleep(10 * tickDuration)
	}

	for i, m := range membs {
		if uint64(m.s.ID()) == lead {
			return i
		}
	}

	return -1
}

func (c *cluster) waitVersion() {
	for _, m := range c.Members {
		for {
			if m.s.ClusterVersion() != nil {
				break
			}
			time.Sleep(tickDuration)
		}
	}
}

func (c *cluster) name(i int) string {
	return fmt.Sprint("node", i)
}

// isMembersEqual checks whether two members equal except ID field.
// The given wmembs should always set ID field to empty string.
func isMembersEqual(membs []client.Member, wmembs []client.Member) bool {
	sort.Sort(SortableMemberSliceByPeerURLs(membs))
	sort.Sort(SortableMemberSliceByPeerURLs(wmembs))
	for i := range membs {
		membs[i].ID = ""
	}
	return reflect.DeepEqual(membs, wmembs)
}

func newLocalListener(t *testing.T) net.Listener {
	port := atomic.AddInt64(&nextListenPort, 1)
	l, err := net.Listen("tcp", "127.0.0.1:"+strconv.FormatInt(port, 10))
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func newListenerWithAddr(t *testing.T, addr string) net.Listener {
	var err error
	var l net.Listener
	// TODO: we want to reuse a previous closed port immediately.
	// a better way is to set SO_REUSExx instead of doing retry.
	for i := 0; i < 5; i++ {
		l, err = net.Listen("tcp", addr)
		if err == nil {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	if err != nil {
		t.Fatal(err)
	}
	return l
}

type member struct {
	etcdserver.ServerConfig
	PeerListeners, ClientListeners []net.Listener
	grpcListener                   net.Listener
	// PeerTLSInfo enables peer TLS when set
	PeerTLSInfo *transport.TLSInfo
	// ClientTLSInfo enables client TLS when set
	ClientTLSInfo *transport.TLSInfo

	raftHandler *testutil.PauseableHandler
	s           *etcdserver.EtcdServer
	hss         []*httptest.Server

	grpcServer *grpc.Server
	grpcAddr   string
}

// mustNewMember return an inited member with the given name. If peerTLS is
// set, it will use https scheme to communicate between peers.
func mustNewMember(t *testing.T, name string, peerTLS *transport.TLSInfo, clientTLS *transport.TLSInfo) *member {
	var err error
	m := &member{}

	peerScheme, clientScheme := "http", "http"
	if peerTLS != nil {
		peerScheme = "https"
	}
	if clientTLS != nil {
		clientScheme = "https"
	}

	pln := newLocalListener(t)
	m.PeerListeners = []net.Listener{pln}
	m.PeerURLs, err = types.NewURLs([]string{peerScheme + "://" + pln.Addr().String()})
	if err != nil {
		t.Fatal(err)
	}
	m.PeerTLSInfo = peerTLS

	cln := newLocalListener(t)
	m.ClientListeners = []net.Listener{cln}
	m.ClientURLs, err = types.NewURLs([]string{clientScheme + "://" + cln.Addr().String()})
	if err != nil {
		t.Fatal(err)
	}
	m.ClientTLSInfo = clientTLS

	m.Name = name

	m.DataDir, err = ioutil.TempDir(os.TempDir(), "etcd")
	if err != nil {
		t.Fatal(err)
	}
	clusterStr := fmt.Sprintf("%s=%s://%s", name, peerScheme, pln.Addr().String())
	m.InitialPeerURLsMap, err = types.NewURLsMap(clusterStr)
	if err != nil {
		t.Fatal(err)
	}
	m.InitialClusterToken = clusterName
	m.NewCluster = true
	m.BootstrapTimeout = 10 * time.Millisecond
	if m.PeerTLSInfo != nil {
		m.ServerConfig.PeerTLSInfo = *m.PeerTLSInfo
	}
	m.ElectionTicks = electionTicks
	m.TickMs = uint(tickDuration / time.Millisecond)
	return m
}

// listenGRPC starts a grpc server over a unix domain socket on the member
func (m *member) listenGRPC() error {
	// prefix with localhost so cert has right domain
	m.grpcAddr = "localhost:" + m.Name + ".sock"
	if err := os.RemoveAll(m.grpcAddr); err != nil {
		return err
	}
	l, err := net.Listen("unix", m.grpcAddr)
	if err != nil {
		return fmt.Errorf("listen failed on grpc socket %s (%v)", m.grpcAddr, err)
	}
	m.grpcAddr = "unix://" + m.grpcAddr
	m.grpcListener = l
	return nil
}

// NewClientV3 creates a new grpc client connection to the member
func NewClientV3(m *member) (*clientv3.Client, error) {
	if m.grpcAddr == "" {
		return nil, fmt.Errorf("member not configured for grpc")
	}

	cfg := clientv3.Config{
		Endpoints:   []string{m.grpcAddr},
		DialTimeout: 5 * time.Second,
	}

	if m.ClientTLSInfo != nil {
		tls, err := m.ClientTLSInfo.ClientConfig()
		if err != nil {
			return nil, err
		}
		cfg.TLS = tls
	}
	return clientv3.New(cfg)
}

// Clone returns a member with the same server configuration. The returned
// member will not set PeerListeners and ClientListeners.
func (m *member) Clone(t *testing.T) *member {
	mm := &member{}
	mm.ServerConfig = m.ServerConfig

	var err error
	clientURLStrs := m.ClientURLs.StringSlice()
	mm.ClientURLs, err = types.NewURLs(clientURLStrs)
	if err != nil {
		// this should never fail
		panic(err)
	}
	peerURLStrs := m.PeerURLs.StringSlice()
	mm.PeerURLs, err = types.NewURLs(peerURLStrs)
	if err != nil {
		// this should never fail
		panic(err)
	}
	clusterStr := m.InitialPeerURLsMap.String()
	mm.InitialPeerURLsMap, err = types.NewURLsMap(clusterStr)
	if err != nil {
		// this should never fail
		panic(err)
	}
	mm.InitialClusterToken = m.InitialClusterToken
	mm.ElectionTicks = m.ElectionTicks
	mm.PeerTLSInfo = m.PeerTLSInfo
	mm.ClientTLSInfo = m.ClientTLSInfo
	return mm
}

// Launch starts a member based on ServerConfig, PeerListeners
// and ClientListeners.
func (m *member) Launch() error {
	var err error
	if m.s, err = etcdserver.NewServer(&m.ServerConfig); err != nil {
		return fmt.Errorf("failed to initialize the etcd server: %v", err)
	}
	m.s.SyncTicker = time.Tick(500 * time.Millisecond)
	m.s.Start()

	m.raftHandler = &testutil.PauseableHandler{Next: v2http.NewPeerHandler(m.s)}

	for _, ln := range m.PeerListeners {
		hs := &httptest.Server{
			Listener: ln,
			Config:   &http.Server{Handler: m.raftHandler},
		}
		if m.PeerTLSInfo == nil {
			hs.Start()
		} else {
			hs.TLS, err = m.PeerTLSInfo.ServerConfig()
			if err != nil {
				return err
			}
			hs.StartTLS()
		}
		m.hss = append(m.hss, hs)
	}
	for _, ln := range m.ClientListeners {
		hs := &httptest.Server{
			Listener: ln,
			Config:   &http.Server{Handler: v2http.NewClientHandler(m.s, m.ServerConfig.ReqTimeout())},
		}
		if m.ClientTLSInfo == nil {
			hs.Start()
		} else {
			hs.TLS, err = m.ClientTLSInfo.ServerConfig()
			if err != nil {
				return err
			}
			hs.StartTLS()
		}
		m.hss = append(m.hss, hs)
	}
	if m.grpcListener != nil {
		var (
			tlscfg *tls.Config
		)
		if m.ClientTLSInfo != nil && !m.ClientTLSInfo.Empty() {
			tlscfg, err = m.ClientTLSInfo.ServerConfig()
			if err != nil {
				return err
			}
		}
		m.grpcServer = v3rpc.Server(m.s, tlscfg)
		go m.grpcServer.Serve(m.grpcListener)
	}
	return nil
}

func (m *member) WaitOK(t *testing.T) {
	cc := mustNewHTTPClient(t, []string{m.URL()}, m.ClientTLSInfo)
	kapi := client.NewKeysAPI(cc)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
		_, err := kapi.Get(ctx, "/", nil)
		if err != nil {
			time.Sleep(tickDuration)
			continue
		}
		cancel()
		break
	}
	for m.s.Leader() == 0 {
		time.Sleep(tickDuration)
	}
}

func (m *member) URL() string { return m.ClientURLs[0].String() }

func (m *member) Pause() {
	m.raftHandler.Pause()
	m.s.PauseSending()
}

func (m *member) Resume() {
	m.raftHandler.Resume()
	m.s.ResumeSending()
}

// Close stops the member's etcdserver and closes its connections
func (m *member) Close() {
	if m.grpcServer != nil {
		m.grpcServer.Stop()
		m.grpcServer = nil
	}
	m.s.Stop()
	for _, hs := range m.hss {
		hs.CloseClientConnections()
		hs.Close()
	}
}

// Stop stops the member, but the data dir of the member is preserved.
func (m *member) Stop(t *testing.T) {
	m.Close()
	m.hss = nil
}

// StopNotify unblocks when a member stop completes
func (m *member) StopNotify() <-chan struct{} {
	return m.s.StopNotify()
}

// Restart starts the member using the preserved data dir.
func (m *member) Restart(t *testing.T) error {
	newPeerListeners := make([]net.Listener, 0)
	for _, ln := range m.PeerListeners {
		newPeerListeners = append(newPeerListeners, newListenerWithAddr(t, ln.Addr().String()))
	}
	m.PeerListeners = newPeerListeners
	newClientListeners := make([]net.Listener, 0)
	for _, ln := range m.ClientListeners {
		newClientListeners = append(newClientListeners, newListenerWithAddr(t, ln.Addr().String()))
	}
	m.ClientListeners = newClientListeners

	if m.grpcListener != nil {
		if err := m.listenGRPC(); err != nil {
			t.Fatal(err)
		}
	}

	return m.Launch()
}

// Terminate stops the member and removes the data dir.
func (m *member) Terminate(t *testing.T) {
	m.Close()
	if err := os.RemoveAll(m.ServerConfig.DataDir); err != nil {
		t.Fatal(err)
	}
}

func mustNewHTTPClient(t *testing.T, eps []string, tls *transport.TLSInfo) client.Client {
	cfgtls := transport.TLSInfo{}
	if tls != nil {
		cfgtls = *tls
	}
	cfg := client.Config{Transport: mustNewTransport(t, cfgtls), Endpoints: eps}
	c, err := client.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func mustNewTransport(t *testing.T, tlsInfo transport.TLSInfo) *http.Transport {
	// tick in integration test is short, so 1s dial timeout could play well.
	tr, err := transport.NewTimeoutTransport(tlsInfo, time.Second, rafthttp.ConnReadTimeout, rafthttp.ConnWriteTimeout)
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

type SortableMemberSliceByPeerURLs []client.Member

func (p SortableMemberSliceByPeerURLs) Len() int { return len(p) }
func (p SortableMemberSliceByPeerURLs) Less(i, j int) bool {
	return p[i].PeerURLs[0] < p[j].PeerURLs[0]
}
func (p SortableMemberSliceByPeerURLs) Swap(i, j int) { p[i], p[j] = p[j], p[i] }

type ClusterV3 struct {
	*cluster
	clients []*clientv3.Client
}

// NewClusterV3 returns a launched cluster with a grpc client connection
// for each cluster member.
func NewClusterV3(t *testing.T, cfg *ClusterConfig) *ClusterV3 {
	cfg.UseGRPC = true
	clus := &ClusterV3{cluster: NewClusterByConfig(t, cfg)}
	for _, m := range clus.Members {
		client, err := NewClientV3(m)
		if err != nil {
			t.Fatal(err)
		}
		clus.clients = append(clus.clients, client)
	}
	clus.Launch(t)
	return clus
}

func (c *ClusterV3) Terminate(t *testing.T) {
	for _, client := range c.clients {
		if err := client.Close(); err != nil {
			t.Error(err)
		}
	}
	c.cluster.Terminate(t)
}

func (c *ClusterV3) RandClient() *clientv3.Client {
	return c.clients[rand.Intn(len(c.clients))]
}

func (c *ClusterV3) Client(i int) *clientv3.Client {
	return c.clients[i]
}

type grpcAPI struct {
	// Cluster is the cluster API for the client's connection.
	Cluster pb.ClusterClient
	// KV is the keyvalue API for the client's connection.
	KV pb.KVClient
	// Lease is the lease API for the client's connection.
	Lease pb.LeaseClient
	// Watch is the watch API for the client's connection.
	Watch pb.WatchClient
	// Maintenance is the maintenance API for the client's connection.
	Maintenance pb.MaintenanceClient
}

func toGRPC(c *clientv3.Client) grpcAPI {
	return grpcAPI{
		pb.NewClusterClient(c.ActiveConnection()),
		pb.NewKVClient(c.ActiveConnection()),
		pb.NewLeaseClient(c.ActiveConnection()),
		pb.NewWatchClient(c.ActiveConnection()),
		pb.NewMaintenanceClient(c.ActiveConnection()),
	}
}
