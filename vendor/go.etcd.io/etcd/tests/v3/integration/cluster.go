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

package integration

import (
	"context"
	"crypto/tls"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/client/pkg/v3/testutil"
	"go.etcd.io/etcd/client/pkg/v3/tlsutil"
	"go.etcd.io/etcd/client/pkg/v3/transport"
	"go.etcd.io/etcd/client/pkg/v3/types"
	"go.etcd.io/etcd/client/v2"
	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/raft/v3"
	"go.etcd.io/etcd/server/v3/config"
	"go.etcd.io/etcd/server/v3/embed"
	"go.etcd.io/etcd/server/v3/etcdserver"
	"go.etcd.io/etcd/server/v3/etcdserver/api/etcdhttp"
	"go.etcd.io/etcd/server/v3/etcdserver/api/rafthttp"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2http"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3client"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3election"
	epb "go.etcd.io/etcd/server/v3/etcdserver/api/v3election/v3electionpb"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3lock"
	lockpb "go.etcd.io/etcd/server/v3/etcdserver/api/v3lock/v3lockpb"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3rpc"
	"go.etcd.io/etcd/server/v3/verify"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest"

	"github.com/soheilhy/cmux"
	"go.uber.org/zap"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
)

const (
	// RequestWaitTimeout is the time duration to wait for a request to go through or detect leader loss.
	RequestWaitTimeout = 5 * time.Second
	tickDuration       = 10 * time.Millisecond
	requestTimeout     = 20 * time.Second

	clusterName  = "etcd"
	basePort     = 21000
	URLScheme    = "unix"
	URLSchemeTLS = "unixs"
)

var (
	electionTicks = 10

	// integration test uses unique ports, counting up, to listen for each
	// member, ensuring restarted members can listen on the same port again.
	localListenCount = int64(0)

	testTLSInfo = transport.TLSInfo{
		KeyFile:        MustAbsPath("../fixtures/server.key.insecure"),
		CertFile:       MustAbsPath("../fixtures/server.crt"),
		TrustedCAFile:  MustAbsPath("../fixtures/ca.crt"),
		ClientCertAuth: true,
	}

	testTLSInfoWithSpecificUsage = transport.TLSInfo{
		KeyFile:        MustAbsPath("../fixtures/server-serverusage.key.insecure"),
		CertFile:       MustAbsPath("../fixtures/server-serverusage.crt"),
		ClientKeyFile:  MustAbsPath("../fixtures/client-clientusage.key.insecure"),
		ClientCertFile: MustAbsPath("../fixtures/client-clientusage.crt"),
		TrustedCAFile:  MustAbsPath("../fixtures/ca.crt"),
		ClientCertAuth: true,
	}

	testTLSInfoIP = transport.TLSInfo{
		KeyFile:        MustAbsPath("../fixtures/server-ip.key.insecure"),
		CertFile:       MustAbsPath("../fixtures/server-ip.crt"),
		TrustedCAFile:  MustAbsPath("../fixtures/ca.crt"),
		ClientCertAuth: true,
	}

	testTLSInfoExpired = transport.TLSInfo{
		KeyFile:        MustAbsPath("./fixtures-expired/server.key.insecure"),
		CertFile:       MustAbsPath("./fixtures-expired/server.crt"),
		TrustedCAFile:  MustAbsPath("./fixtures-expired/ca.crt"),
		ClientCertAuth: true,
	}

	testTLSInfoExpiredIP = transport.TLSInfo{
		KeyFile:        MustAbsPath("./fixtures-expired/server-ip.key.insecure"),
		CertFile:       MustAbsPath("./fixtures-expired/server-ip.crt"),
		TrustedCAFile:  MustAbsPath("./fixtures-expired/ca.crt"),
		ClientCertAuth: true,
	}

	defaultTokenJWT = fmt.Sprintf("jwt,pub-key=%s,priv-key=%s,sign-method=RS256,ttl=1s",
		MustAbsPath("../fixtures/server.crt"), MustAbsPath("../fixtures/server.key.insecure"))
)

type ClusterConfig struct {
	Size      int
	PeerTLS   *transport.TLSInfo
	ClientTLS *transport.TLSInfo

	DiscoveryURL string

	AuthToken string

	UseGRPC bool

	QuotaBackendBytes int64

	MaxTxnOps              uint
	MaxRequestBytes        uint
	SnapshotCount          uint64
	SnapshotCatchUpEntries uint64

	GRPCKeepAliveMinTime  time.Duration
	GRPCKeepAliveInterval time.Duration
	GRPCKeepAliveTimeout  time.Duration

	// SkipCreatingClient to skip creating clients for each member.
	SkipCreatingClient bool

	ClientMaxCallSendMsgSize int
	ClientMaxCallRecvMsgSize int

	// UseIP is true to use only IP for gRPC requests.
	UseIP bool

	EnableLeaseCheckpoint   bool
	LeaseCheckpointInterval time.Duration

	WatchProgressNotifyInterval time.Duration
}

type cluster struct {
	cfg           *ClusterConfig
	Members       []*member
	lastMemberNum int
}

func (c *cluster) generateMemberName() string {
	c.lastMemberNum++
	return fmt.Sprintf("m%v", c.lastMemberNum-1)
}

func schemeFromTLSInfo(tls *transport.TLSInfo) string {
	if tls == nil {
		return URLScheme
	}
	return URLSchemeTLS
}

func (c *cluster) fillClusterForMembers() error {
	if c.cfg.DiscoveryURL != "" {
		// cluster will be discovered
		return nil
	}

	addrs := make([]string, 0)
	for _, m := range c.Members {
		scheme := schemeFromTLSInfo(m.PeerTLSInfo)
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

func newCluster(t testutil.TB, cfg *ClusterConfig) *cluster {
	testutil.SkipTestIfShortMode(t, "Cannot start etcd cluster in --short tests")

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
func NewCluster(t testutil.TB, size int) *cluster {
	t.Helper()
	return newCluster(t, &ClusterConfig{Size: size})
}

// NewClusterByConfig returns an unlaunched cluster defined by a cluster configuration
func NewClusterByConfig(t testutil.TB, cfg *ClusterConfig) *cluster {
	return newCluster(t, cfg)
}

func (c *cluster) Launch(t testutil.TB) {
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
			c.Terminate(t)
			t.Fatalf("error setting up member: %v", err)
		}
	}
	// wait cluster to be stable to receive future client requests
	c.waitMembersMatch(t, c.HTTPMembers())
	c.waitVersion()
	for _, m := range c.Members {
		t.Logf(" - %v -> %v (%v)", m.Name, m.ID(), m.GRPCAddr())
	}
}

func (c *cluster) URL(i int) string {
	return c.Members[i].ClientURLs[0].String()
}

// URLs returns a list of all active client URLs in the cluster
func (c *cluster) URLs() []string {
	return getMembersURLs(c.Members)
}

func getMembersURLs(members []*member) []string {
	urls := make([]string, 0)
	for _, m := range members {
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
		pScheme := schemeFromTLSInfo(m.PeerTLSInfo)
		cScheme := schemeFromTLSInfo(m.ClientTLSInfo)
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

func (c *cluster) mustNewMember(t testutil.TB) *member {
	m := mustNewMember(t,
		memberConfig{
			name:                        c.generateMemberName(),
			authToken:                   c.cfg.AuthToken,
			peerTLS:                     c.cfg.PeerTLS,
			clientTLS:                   c.cfg.ClientTLS,
			quotaBackendBytes:           c.cfg.QuotaBackendBytes,
			maxTxnOps:                   c.cfg.MaxTxnOps,
			maxRequestBytes:             c.cfg.MaxRequestBytes,
			snapshotCount:               c.cfg.SnapshotCount,
			snapshotCatchUpEntries:      c.cfg.SnapshotCatchUpEntries,
			grpcKeepAliveMinTime:        c.cfg.GRPCKeepAliveMinTime,
			grpcKeepAliveInterval:       c.cfg.GRPCKeepAliveInterval,
			grpcKeepAliveTimeout:        c.cfg.GRPCKeepAliveTimeout,
			clientMaxCallSendMsgSize:    c.cfg.ClientMaxCallSendMsgSize,
			clientMaxCallRecvMsgSize:    c.cfg.ClientMaxCallRecvMsgSize,
			useIP:                       c.cfg.UseIP,
			enableLeaseCheckpoint:       c.cfg.EnableLeaseCheckpoint,
			leaseCheckpointInterval:     c.cfg.LeaseCheckpointInterval,
			WatchProgressNotifyInterval: c.cfg.WatchProgressNotifyInterval,
		})
	m.DiscoveryURL = c.cfg.DiscoveryURL
	if c.cfg.UseGRPC {
		if err := m.listenGRPC(); err != nil {
			t.Fatal(err)
		}
	}
	return m
}

// addMember return PeerURLs of the added member.
func (c *cluster) addMember(t testutil.TB) types.URLs {
	m := c.mustNewMember(t)

	scheme := schemeFromTLSInfo(c.cfg.PeerTLS)

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
	return m.PeerURLs
}

func (c *cluster) addMemberByURL(t testutil.TB, clientURL, peerURL string) error {
	cc := MustNewHTTPClient(t, []string{clientURL}, c.cfg.ClientTLS)
	ma := client.NewMembersAPI(cc)
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	_, err := ma.Add(ctx, peerURL)
	cancel()
	if err != nil {
		return err
	}

	// wait for the add node entry applied in the cluster
	members := append(c.HTTPMembers(), client.Member{PeerURLs: []string{peerURL}, ClientURLs: []string{}})
	c.waitMembersMatch(t, members)
	return nil
}

// AddMember return PeerURLs of the added member.
func (c *cluster) AddMember(t testutil.TB) types.URLs {
	return c.addMember(t)
}

func (c *cluster) RemoveMember(t testutil.TB, id uint64) {
	if err := c.removeMember(t, id); err != nil {
		t.Fatal(err)
	}
}

func (c *cluster) removeMember(t testutil.TB, id uint64) error {
	// send remove request to the cluster
	cc := MustNewHTTPClient(t, c.URLs(), c.cfg.ClientTLS)
	ma := client.NewMembersAPI(cc)
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	err := ma.Remove(ctx, types.ID(id).String())
	cancel()
	if err != nil {
		return err
	}
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
	return nil
}

func (c *cluster) Terminate(t testutil.TB) {
	var wg sync.WaitGroup
	wg.Add(len(c.Members))
	for _, m := range c.Members {
		go func(mm *member) {
			defer wg.Done()
			mm.Terminate(t)
		}(m)
	}
	wg.Wait()
}

func (c *cluster) waitMembersMatch(t testutil.TB, membs []client.Member) {
	for _, u := range c.URLs() {
		cc := MustNewHTTPClient(t, []string{u}, c.cfg.ClientTLS)
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
}

// WaitLeader returns index of the member in c.Members that is leader (or -1).
func (c *cluster) WaitLeader(t testutil.TB) int { return c.waitLeader(t, c.Members) }

// waitLeader waits until given members agree on the same leader,
// and returns its 'index' in the 'membs' list (or -1).
func (c *cluster) waitLeader(t testutil.TB, membs []*member) int {
	possibleLead := make(map[uint64]bool)
	var lead uint64
	for _, m := range membs {
		possibleLead[uint64(m.s.ID())] = true
	}
	cc := MustNewHTTPClient(t, getMembersURLs(membs), nil)
	kapi := client.NewKeysAPI(cc)

	// ensure leader is up via linearizable get
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 10*tickDuration+time.Second)
		_, err := kapi.Get(ctx, "0", &client.GetOptions{Quorum: true})
		cancel()
		if err == nil || strings.Contains(err.Error(), "Key not found") {
			break
		}
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
				time.Sleep(10 * tickDuration)
				break
			}
			lead = m.s.Lead()
		}
	}

	for i, m := range membs {
		if uint64(m.s.ID()) == lead {
			return i
		}
	}

	return -1
}

func (c *cluster) WaitNoLeader() { c.waitNoLeader(c.Members) }

// waitNoLeader waits until given members lose leader.
func (c *cluster) waitNoLeader(membs []*member) {
	noLeader := false
	for !noLeader {
		noLeader = true
		for _, m := range membs {
			select {
			case <-m.s.StopNotify():
				continue
			default:
			}
			if m.s.Lead() != 0 {
				noLeader = false
				time.Sleep(10 * tickDuration)
				break
			}
		}
	}
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

func newLocalListener(t testutil.TB) net.Listener {
	c := atomic.AddInt64(&localListenCount, 1)
	// Go 1.8+ allows only numbers in port
	addr := fmt.Sprintf("127.0.0.1:%05d%05d", c+basePort, os.Getpid())
	return NewListenerWithAddr(t, addr)
}

func NewListenerWithAddr(t testutil.TB, addr string) net.Listener {
	l, err := transport.NewUnixListener(addr)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

type member struct {
	config.ServerConfig
	PeerListeners, ClientListeners []net.Listener
	grpcListener                   net.Listener
	// PeerTLSInfo enables peer TLS when set
	PeerTLSInfo *transport.TLSInfo
	// ClientTLSInfo enables client TLS when set
	ClientTLSInfo *transport.TLSInfo
	DialOptions   []grpc.DialOption

	raftHandler   *testutil.PauseableHandler
	s             *etcdserver.EtcdServer
	serverClosers []func()

	grpcServerOpts []grpc.ServerOption
	grpcServer     *grpc.Server
	grpcServerPeer *grpc.Server
	grpcAddr       string
	grpcBridge     *bridge

	// serverClient is a clientv3 that directly calls the etcdserver.
	serverClient *clientv3.Client

	keepDataDirTerminate     bool
	clientMaxCallSendMsgSize int
	clientMaxCallRecvMsgSize int
	useIP                    bool

	isLearner bool
	closed    bool
}

func (m *member) GRPCAddr() string { return m.grpcAddr }

type memberConfig struct {
	name                        string
	peerTLS                     *transport.TLSInfo
	clientTLS                   *transport.TLSInfo
	authToken                   string
	quotaBackendBytes           int64
	maxTxnOps                   uint
	maxRequestBytes             uint
	snapshotCount               uint64
	snapshotCatchUpEntries      uint64
	grpcKeepAliveMinTime        time.Duration
	grpcKeepAliveInterval       time.Duration
	grpcKeepAliveTimeout        time.Duration
	clientMaxCallSendMsgSize    int
	clientMaxCallRecvMsgSize    int
	useIP                       bool
	enableLeaseCheckpoint       bool
	leaseCheckpointInterval     time.Duration
	WatchProgressNotifyInterval time.Duration
}

// mustNewMember return an inited member with the given name. If peerTLS is
// set, it will use https scheme to communicate between peers.
func mustNewMember(t testutil.TB, mcfg memberConfig) *member {
	var err error
	m := &member{}

	peerScheme := schemeFromTLSInfo(mcfg.peerTLS)
	clientScheme := schemeFromTLSInfo(mcfg.clientTLS)

	pln := newLocalListener(t)
	m.PeerListeners = []net.Listener{pln}
	m.PeerURLs, err = types.NewURLs([]string{peerScheme + "://" + pln.Addr().String()})
	if err != nil {
		t.Fatal(err)
	}
	m.PeerTLSInfo = mcfg.peerTLS

	cln := newLocalListener(t)
	m.ClientListeners = []net.Listener{cln}
	m.ClientURLs, err = types.NewURLs([]string{clientScheme + "://" + cln.Addr().String()})
	if err != nil {
		t.Fatal(err)
	}
	m.ClientTLSInfo = mcfg.clientTLS

	m.Name = mcfg.name

	m.DataDir, err = ioutil.TempDir(t.TempDir(), "etcd")
	if err != nil {
		t.Fatal(err)
	}
	clusterStr := fmt.Sprintf("%s=%s://%s", mcfg.name, peerScheme, pln.Addr().String())
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
	m.InitialElectionTickAdvance = true
	m.TickMs = uint(tickDuration / time.Millisecond)
	m.QuotaBackendBytes = mcfg.quotaBackendBytes
	m.MaxTxnOps = mcfg.maxTxnOps
	if m.MaxTxnOps == 0 {
		m.MaxTxnOps = embed.DefaultMaxTxnOps
	}
	m.MaxRequestBytes = mcfg.maxRequestBytes
	if m.MaxRequestBytes == 0 {
		m.MaxRequestBytes = embed.DefaultMaxRequestBytes
	}
	m.SnapshotCount = etcdserver.DefaultSnapshotCount
	if mcfg.snapshotCount != 0 {
		m.SnapshotCount = mcfg.snapshotCount
	}
	m.SnapshotCatchUpEntries = etcdserver.DefaultSnapshotCatchUpEntries
	if mcfg.snapshotCatchUpEntries != 0 {
		m.SnapshotCatchUpEntries = mcfg.snapshotCatchUpEntries
	}

	// for the purpose of integration testing, simple token is enough
	m.AuthToken = "simple"
	if mcfg.authToken != "" {
		m.AuthToken = mcfg.authToken
	}

	m.BcryptCost = uint(bcrypt.MinCost) // use min bcrypt cost to speedy up integration testing

	m.grpcServerOpts = []grpc.ServerOption{}
	if mcfg.grpcKeepAliveMinTime > time.Duration(0) {
		m.grpcServerOpts = append(m.grpcServerOpts, grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             mcfg.grpcKeepAliveMinTime,
			PermitWithoutStream: false,
		}))
	}
	if mcfg.grpcKeepAliveInterval > time.Duration(0) &&
		mcfg.grpcKeepAliveTimeout > time.Duration(0) {
		m.grpcServerOpts = append(m.grpcServerOpts, grpc.KeepaliveParams(keepalive.ServerParameters{
			Time:    mcfg.grpcKeepAliveInterval,
			Timeout: mcfg.grpcKeepAliveTimeout,
		}))
	}
	m.clientMaxCallSendMsgSize = mcfg.clientMaxCallSendMsgSize
	m.clientMaxCallRecvMsgSize = mcfg.clientMaxCallRecvMsgSize
	m.useIP = mcfg.useIP
	m.EnableLeaseCheckpoint = mcfg.enableLeaseCheckpoint
	m.LeaseCheckpointInterval = mcfg.leaseCheckpointInterval

	m.WatchProgressNotifyInterval = mcfg.WatchProgressNotifyInterval

	m.InitialCorruptCheck = true
	m.WarningApplyDuration = embed.DefaultWarningApplyDuration

	m.V2Deprecation = config.V2_DEPR_DEFAULT

	m.Logger = memberLogger(t, mcfg.name)
	t.Cleanup(func() {
		// if we didn't cleanup the logger, the consecutive test
		// might reuse this (t).
		raft.ResetDefaultLogger()
	})
	return m
}

func memberLogger(t testutil.TB, name string) *zap.Logger {
	level := zapcore.InfoLevel
	if os.Getenv("CLUSTER_DEBUG") != "" {
		level = zapcore.DebugLevel
	}

	options := zaptest.WrapOptions(zap.Fields(zap.String("member", name)))
	return zaptest.NewLogger(t, zaptest.Level(level), options).Named(name)
}

// listenGRPC starts a grpc server over a unix domain socket on the member
func (m *member) listenGRPC() error {
	// prefix with localhost so cert has right domain
	m.grpcAddr = "localhost:" + m.Name
	m.Logger.Info("LISTEN GRPC", zap.String("m.grpcAddr", m.grpcAddr), zap.String("m.Name", m.Name))
	if m.useIP { // for IP-only TLS certs
		m.grpcAddr = "127.0.0.1:" + m.Name
	}
	l, err := transport.NewUnixListener(m.grpcAddr)
	if err != nil {
		return fmt.Errorf("listen failed on grpc socket %s (%v)", m.grpcAddr, err)
	}
	m.grpcBridge, err = newBridge(m.grpcAddr)
	if err != nil {
		l.Close()
		return err
	}
	m.grpcAddr = schemeFromTLSInfo(m.ClientTLSInfo) + "://" + m.grpcBridge.inaddr
	m.grpcListener = l
	return nil
}

func (m *member) ElectionTimeout() time.Duration {
	return time.Duration(m.s.Cfg.ElectionTicks*int(m.s.Cfg.TickMs)) * time.Millisecond
}

func (m *member) ID() types.ID { return m.s.ID() }

func (m *member) DropConnections()    { m.grpcBridge.Reset() }
func (m *member) PauseConnections()   { m.grpcBridge.Pause() }
func (m *member) UnpauseConnections() { m.grpcBridge.Unpause() }
func (m *member) Blackhole()          { m.grpcBridge.Blackhole() }
func (m *member) Unblackhole()        { m.grpcBridge.Unblackhole() }

// NewClientV3 creates a new grpc client connection to the member
func NewClientV3(m *member) (*clientv3.Client, error) {
	if m.grpcAddr == "" {
		return nil, fmt.Errorf("member not configured for grpc")
	}

	cfg := clientv3.Config{
		Endpoints:          []string{m.grpcAddr},
		DialTimeout:        5 * time.Second,
		DialOptions:        []grpc.DialOption{grpc.WithBlock()},
		MaxCallSendMsgSize: m.clientMaxCallSendMsgSize,
		MaxCallRecvMsgSize: m.clientMaxCallRecvMsgSize,
	}

	if m.ClientTLSInfo != nil {
		tls, err := m.ClientTLSInfo.ClientConfig()
		if err != nil {
			return nil, err
		}
		cfg.TLS = tls
	}
	if m.DialOptions != nil {
		cfg.DialOptions = append(cfg.DialOptions, m.DialOptions...)
	}
	return newClientV3(cfg, m.Logger.Named("client"))
}

// Clone returns a member with the same server configuration. The returned
// member will not set PeerListeners and ClientListeners.
func (m *member) Clone(t testutil.TB) *member {
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
	mm.Logger = memberLogger(t, mm.Name+"c")
	return mm
}

// Launch starts a member based on ServerConfig, PeerListeners
// and ClientListeners.
func (m *member) Launch() error {
	m.Logger.Info(
		"launching a member",
		zap.String("name", m.Name),
		zap.Strings("advertise-peer-urls", m.PeerURLs.StringSlice()),
		zap.Strings("listen-client-urls", m.ClientURLs.StringSlice()),
		zap.String("grpc-address", m.grpcAddr),
	)
	var err error
	if m.s, err = etcdserver.NewServer(m.ServerConfig); err != nil {
		return fmt.Errorf("failed to initialize the etcd server: %v", err)
	}
	m.s.SyncTicker = time.NewTicker(500 * time.Millisecond)
	m.s.Start()

	var peerTLScfg *tls.Config
	if m.PeerTLSInfo != nil && !m.PeerTLSInfo.Empty() {
		if peerTLScfg, err = m.PeerTLSInfo.ServerConfig(); err != nil {
			return err
		}
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
		m.grpcServer = v3rpc.Server(m.s, tlscfg, m.grpcServerOpts...)
		m.grpcServerPeer = v3rpc.Server(m.s, peerTLScfg)
		m.serverClient = v3client.New(m.s)
		lockpb.RegisterLockServer(m.grpcServer, v3lock.NewLockServer(m.serverClient))
		epb.RegisterElectionServer(m.grpcServer, v3election.NewElectionServer(m.serverClient))
		go m.grpcServer.Serve(m.grpcListener)
	}

	m.raftHandler = &testutil.PauseableHandler{Next: etcdhttp.NewPeerHandler(m.Logger, m.s)}

	h := (http.Handler)(m.raftHandler)
	if m.grpcListener != nil {
		h = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.ProtoMajor == 2 && strings.Contains(r.Header.Get("Content-Type"), "application/grpc") {
				m.grpcServerPeer.ServeHTTP(w, r)
			} else {
				m.raftHandler.ServeHTTP(w, r)
			}
		})
	}

	for _, ln := range m.PeerListeners {
		cm := cmux.New(ln)
		// don't hang on matcher after closing listener
		cm.SetReadTimeout(time.Second)

		if m.grpcServer != nil {
			grpcl := cm.Match(cmux.HTTP2())
			go m.grpcServerPeer.Serve(grpcl)
		}

		// serve http1/http2 rafthttp/grpc
		ll := cm.Match(cmux.Any())
		if peerTLScfg != nil {
			if ll, err = transport.NewTLSListener(ll, m.PeerTLSInfo); err != nil {
				return err
			}
		}
		hs := &httptest.Server{
			Listener: ll,
			Config: &http.Server{
				Handler:   h,
				TLSConfig: peerTLScfg,
				ErrorLog:  log.New(ioutil.Discard, "net/http", 0),
			},
			TLS: peerTLScfg,
		}
		hs.Start()

		donec := make(chan struct{})
		go func() {
			defer close(donec)
			cm.Serve()
		}()
		closer := func() {
			ll.Close()
			hs.CloseClientConnections()
			hs.Close()
			<-donec
		}
		m.serverClosers = append(m.serverClosers, closer)
	}
	for _, ln := range m.ClientListeners {
		hs := &httptest.Server{
			Listener: ln,
			Config: &http.Server{
				Handler: v2http.NewClientHandler(
					m.Logger,
					m.s,
					m.ServerConfig.ReqTimeout(),
				),
				ErrorLog: log.New(ioutil.Discard, "net/http", 0),
			},
		}
		if m.ClientTLSInfo == nil {
			hs.Start()
		} else {
			info := m.ClientTLSInfo
			hs.TLS, err = info.ServerConfig()
			if err != nil {
				return err
			}

			// baseConfig is called on initial TLS handshake start.
			//
			// Previously,
			// 1. Server has non-empty (*tls.Config).Certificates on client hello
			// 2. Server calls (*tls.Config).GetCertificate iff:
			//    - Server's (*tls.Config).Certificates is not empty, or
			//    - Client supplies SNI; non-empty (*tls.ClientHelloInfo).ServerName
			//
			// When (*tls.Config).Certificates is always populated on initial handshake,
			// client is expected to provide a valid matching SNI to pass the TLS
			// verification, thus trigger server (*tls.Config).GetCertificate to reload
			// TLS assets. However, a cert whose SAN field does not include domain names
			// but only IP addresses, has empty (*tls.ClientHelloInfo).ServerName, thus
			// it was never able to trigger TLS reload on initial handshake; first
			// ceritifcate object was being used, never being updated.
			//
			// Now, (*tls.Config).Certificates is created empty on initial TLS client
			// handshake, in order to trigger (*tls.Config).GetCertificate and populate
			// rest of the certificates on every new TLS connection, even when client
			// SNI is empty (e.g. cert only includes IPs).
			//
			// This introduces another problem with "httptest.Server":
			// when server initial certificates are empty, certificates
			// are overwritten by Go's internal test certs, which have
			// different SAN fields (e.g. example.com). To work around,
			// re-overwrite (*tls.Config).Certificates before starting
			// test server.
			tlsCert, err := tlsutil.NewCert(info.CertFile, info.KeyFile, nil)
			if err != nil {
				return err
			}
			hs.TLS.Certificates = []tls.Certificate{*tlsCert}

			hs.StartTLS()
		}
		closer := func() {
			ln.Close()
			hs.CloseClientConnections()
			hs.Close()
		}
		m.serverClosers = append(m.serverClosers, closer)
	}

	m.Logger.Info(
		"launched a member",
		zap.String("name", m.Name),
		zap.Strings("advertise-peer-urls", m.PeerURLs.StringSlice()),
		zap.Strings("listen-client-urls", m.ClientURLs.StringSlice()),
		zap.String("grpc-address", m.grpcAddr),
	)
	return nil
}

func (m *member) WaitOK(t testutil.TB) {
	m.WaitStarted(t)
	for m.s.Leader() == 0 {
		time.Sleep(tickDuration)
	}
}

func (m *member) WaitStarted(t testutil.TB) {
	cc := MustNewHTTPClient(t, []string{m.URL()}, m.ClientTLSInfo)
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
}

func WaitClientV3(t testutil.TB, kv clientv3.KV) {
	timeout := time.Now().Add(requestTimeout)
	var err error
	for time.Now().Before(timeout) {
		ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
		_, err = kv.Get(ctx, "/")
		cancel()
		if err == nil {
			return
		}
		time.Sleep(tickDuration)
	}
	if err != nil {
		t.Fatalf("timed out waiting for client: %v", err)
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
	if m.grpcBridge != nil {
		m.grpcBridge.Close()
		m.grpcBridge = nil
	}
	if m.serverClient != nil {
		m.serverClient.Close()
		m.serverClient = nil
	}
	if m.grpcServer != nil {
		ch := make(chan struct{})
		go func() {
			defer close(ch)
			// close listeners to stop accepting new connections,
			// will block on any existing transports
			m.grpcServer.GracefulStop()
		}()
		// wait until all pending RPCs are finished
		select {
		case <-ch:
		case <-time.After(2 * time.Second):
			// took too long, manually close open transports
			// e.g. watch streams
			m.grpcServer.Stop()
			<-ch
		}
		m.grpcServer = nil
		m.grpcServerPeer.GracefulStop()
		m.grpcServerPeer.Stop()
		m.grpcServerPeer = nil
	}
	if m.s != nil {
		m.s.HardStop()
	}
	for _, f := range m.serverClosers {
		f()
	}
	if !m.closed {
		// Avoid verification of the same file multiple times
		// (that might not exist any longer)
		verify.MustVerifyIfEnabled(verify.Config{
			Logger:     m.Logger,
			DataDir:    m.DataDir,
			ExactIndex: false,
		})
	}
	m.closed = true
}

// Stop stops the member, but the data dir of the member is preserved.
func (m *member) Stop(_ testutil.TB) {
	m.Logger.Info(
		"stopping a member",
		zap.String("name", m.Name),
		zap.Strings("advertise-peer-urls", m.PeerURLs.StringSlice()),
		zap.Strings("listen-client-urls", m.ClientURLs.StringSlice()),
		zap.String("grpc-address", m.grpcAddr),
	)
	m.Close()
	m.serverClosers = nil
	m.Logger.Info(
		"stopped a member",
		zap.String("name", m.Name),
		zap.Strings("advertise-peer-urls", m.PeerURLs.StringSlice()),
		zap.Strings("listen-client-urls", m.ClientURLs.StringSlice()),
		zap.String("grpc-address", m.grpcAddr),
	)
}

// checkLeaderTransition waits for leader transition, returning the new leader ID.
func checkLeaderTransition(m *member, oldLead uint64) uint64 {
	interval := time.Duration(m.s.Cfg.TickMs) * time.Millisecond
	for m.s.Lead() == 0 || (m.s.Lead() == oldLead) {
		time.Sleep(interval)
	}
	return m.s.Lead()
}

// StopNotify unblocks when a member stop completes
func (m *member) StopNotify() <-chan struct{} {
	return m.s.StopNotify()
}

// Restart starts the member using the preserved data dir.
func (m *member) Restart(t testutil.TB) error {
	m.Logger.Info(
		"restarting a member",
		zap.String("name", m.Name),
		zap.Strings("advertise-peer-urls", m.PeerURLs.StringSlice()),
		zap.Strings("listen-client-urls", m.ClientURLs.StringSlice()),
		zap.String("grpc-address", m.grpcAddr),
	)
	newPeerListeners := make([]net.Listener, 0)
	for _, ln := range m.PeerListeners {
		newPeerListeners = append(newPeerListeners, NewListenerWithAddr(t, ln.Addr().String()))
	}
	m.PeerListeners = newPeerListeners
	newClientListeners := make([]net.Listener, 0)
	for _, ln := range m.ClientListeners {
		newClientListeners = append(newClientListeners, NewListenerWithAddr(t, ln.Addr().String()))
	}
	m.ClientListeners = newClientListeners

	if m.grpcListener != nil {
		if err := m.listenGRPC(); err != nil {
			t.Fatal(err)
		}
	}

	err := m.Launch()
	m.Logger.Info(
		"restarted a member",
		zap.String("name", m.Name),
		zap.Strings("advertise-peer-urls", m.PeerURLs.StringSlice()),
		zap.Strings("listen-client-urls", m.ClientURLs.StringSlice()),
		zap.String("grpc-address", m.grpcAddr),
		zap.Error(err),
	)
	return err
}

// Terminate stops the member and removes the data dir.
func (m *member) Terminate(t testutil.TB) {
	m.Logger.Info(
		"terminating a member",
		zap.String("name", m.Name),
		zap.Strings("advertise-peer-urls", m.PeerURLs.StringSlice()),
		zap.Strings("listen-client-urls", m.ClientURLs.StringSlice()),
		zap.String("grpc-address", m.grpcAddr),
	)
	m.Close()
	if !m.keepDataDirTerminate {
		if err := os.RemoveAll(m.ServerConfig.DataDir); err != nil {
			t.Fatal(err)
		}
	}
	m.Logger.Info(
		"terminated a member",
		zap.String("name", m.Name),
		zap.Strings("advertise-peer-urls", m.PeerURLs.StringSlice()),
		zap.Strings("listen-client-urls", m.ClientURLs.StringSlice()),
		zap.String("grpc-address", m.grpcAddr),
	)
}

// Metric gets the metric value for a member
func (m *member) Metric(metricName string, expectLabels ...string) (string, error) {
	cfgtls := transport.TLSInfo{}
	tr, err := transport.NewTimeoutTransport(cfgtls, time.Second, time.Second, time.Second)
	if err != nil {
		return "", err
	}
	cli := &http.Client{Transport: tr}
	resp, err := cli.Get(m.ClientURLs[0].String() + "/metrics")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	b, rerr := ioutil.ReadAll(resp.Body)
	if rerr != nil {
		return "", rerr
	}
	lines := strings.Split(string(b), "\n")
	for _, l := range lines {
		if !strings.HasPrefix(l, metricName) {
			continue
		}
		ok := true
		for _, lv := range expectLabels {
			if !strings.Contains(l, lv) {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		return strings.Split(l, " ")[1], nil
	}
	return "", nil
}

// InjectPartition drops connections from m to others, vice versa.
func (m *member) InjectPartition(t testutil.TB, others ...*member) {
	for _, other := range others {
		m.s.CutPeer(other.s.ID())
		other.s.CutPeer(m.s.ID())
		t.Logf("network partition injected between: %v <-> %v", m.s.ID(), other.s.ID())
	}
}

// RecoverPartition recovers connections from m to others, vice versa.
func (m *member) RecoverPartition(t testutil.TB, others ...*member) {
	for _, other := range others {
		m.s.MendPeer(other.s.ID())
		other.s.MendPeer(m.s.ID())
		t.Logf("network partition between: %v <-> %v", m.s.ID(), other.s.ID())
	}
}

func (m *member) ReadyNotify() <-chan struct{} {
	return m.s.ReadyNotify()
}

func MustNewHTTPClient(t testutil.TB, eps []string, tls *transport.TLSInfo) client.Client {
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

func mustNewTransport(t testutil.TB, tlsInfo transport.TLSInfo) *http.Transport {
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

	mu      sync.Mutex
	clients []*clientv3.Client
}

// NewClusterV3 returns a launched cluster with a grpc client connection
// for each cluster member.
func NewClusterV3(t testutil.TB, cfg *ClusterConfig) *ClusterV3 {
	t.Helper()

	assertInTestContext(t)

	cfg.UseGRPC = true

	clus := &ClusterV3{
		cluster: NewClusterByConfig(t, cfg),
	}
	clus.Launch(t)

	if !cfg.SkipCreatingClient {
		for _, m := range clus.Members {
			client, err := NewClientV3(m)
			if err != nil {
				t.Fatalf("cannot create client: %v", err)
			}
			clus.clients = append(clus.clients, client)
		}
	}

	return clus
}

func (c *ClusterV3) TakeClient(idx int) {
	c.mu.Lock()
	c.clients[idx] = nil
	c.mu.Unlock()
}

func (c *ClusterV3) Terminate(t testutil.TB) {
	c.mu.Lock()
	for _, client := range c.clients {
		if client == nil {
			continue
		}
		if err := client.Close(); err != nil {
			t.Error(err)
		}
	}
	c.mu.Unlock()
	c.cluster.Terminate(t)
}

func (c *ClusterV3) RandClient() *clientv3.Client {
	return c.clients[rand.Intn(len(c.clients))]
}

func (c *ClusterV3) Client(i int) *clientv3.Client {
	return c.clients[i]
}

// NewClientV3 creates a new grpc client connection to the member
func (c *ClusterV3) NewClientV3(memberIndex int) (*clientv3.Client, error) {
	return NewClientV3(c.Members[memberIndex])
}

func makeClients(t testutil.TB, clus *ClusterV3, clients *[]*clientv3.Client, chooseMemberIndex func() int) func() *clientv3.Client {
	var mu sync.Mutex
	*clients = nil
	return func() *clientv3.Client {
		cli, err := clus.NewClientV3(chooseMemberIndex())
		if err != nil {
			t.Fatalf("cannot create client: %v", err)
		}
		mu.Lock()
		*clients = append(*clients, cli)
		mu.Unlock()
		return cli
	}
}

// MakeSingleNodeClients creates factory of clients that all connect to member 0.
// All the created clients are put on the 'clients' list. The factory is thread-safe.
func MakeSingleNodeClients(t testutil.TB, clus *ClusterV3, clients *[]*clientv3.Client) func() *clientv3.Client {
	return makeClients(t, clus, clients, func() int { return 0 })
}

// MakeMultiNodeClients creates factory of clients that all connect to random members.
// All the created clients are put on the 'clients' list. The factory is thread-safe.
func MakeMultiNodeClients(t testutil.TB, clus *ClusterV3, clients *[]*clientv3.Client) func() *clientv3.Client {
	return makeClients(t, clus, clients, func() int { return rand.Intn(len(clus.Members)) })
}

// CloseClients closes all the clients from the 'clients' list.
func CloseClients(t testutil.TB, clients []*clientv3.Client) {
	for _, cli := range clients {
		if err := cli.Close(); err != nil {
			t.Fatal(err)
		}
	}
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
	// Auth is the authentication API for the client's connection.
	Auth pb.AuthClient
	// Lock is the lock API for the client's connection.
	Lock lockpb.LockClient
	// Election is the election API for the client's connection.
	Election epb.ElectionClient
}

// GetLearnerMembers returns the list of learner members in cluster using MemberList API.
func (c *ClusterV3) GetLearnerMembers() ([]*pb.Member, error) {
	cli := c.Client(0)
	resp, err := cli.MemberList(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to list member %v", err)
	}
	var learners []*pb.Member
	for _, m := range resp.Members {
		if m.IsLearner {
			learners = append(learners, m)
		}
	}
	return learners, nil
}

// AddAndLaunchLearnerMember creates a leaner member, adds it to cluster
// via v3 MemberAdd API, and then launches the new member.
func (c *ClusterV3) AddAndLaunchLearnerMember(t testutil.TB) {
	m := c.mustNewMember(t)
	m.isLearner = true

	scheme := schemeFromTLSInfo(c.cfg.PeerTLS)
	peerURLs := []string{scheme + "://" + m.PeerListeners[0].Addr().String()}

	cli := c.Client(0)
	_, err := cli.MemberAddAsLearner(context.Background(), peerURLs)
	if err != nil {
		t.Fatalf("failed to add learner member %v", err)
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

	c.waitMembersMatch(t)
}

// getMembers returns a list of members in cluster, in format of etcdserverpb.Member
func (c *ClusterV3) getMembers() []*pb.Member {
	var mems []*pb.Member
	for _, m := range c.Members {
		mem := &pb.Member{
			Name:       m.Name,
			PeerURLs:   m.PeerURLs.StringSlice(),
			ClientURLs: m.ClientURLs.StringSlice(),
			IsLearner:  m.isLearner,
		}
		mems = append(mems, mem)
	}
	return mems
}

// waitMembersMatch waits until v3rpc MemberList returns the 'same' members info as the
// local 'c.Members', which is the local recording of members in the testing cluster. With
// the exception that the local recording c.Members does not have info on Member.ID, which
// is generated when the member is been added to cluster.
//
// Note:
// A successful match means the Member.clientURLs are matched. This means member has already
// finished publishing its server attributes to cluster. Publishing attributes is a cluster-wide
// write request (in v2 server). Therefore, at this point, any raft log entries prior to this
// would have already been applied.
//
// If a new member was added to an existing cluster, at this point, it has finished publishing
// its own server attributes to the cluster. And therefore by the same argument, it has already
// applied the raft log entries (especially those of type raftpb.ConfChangeType). At this point,
// the new member has the correct view of the cluster configuration.
//
// Special note on learner member:
// Learner member is only added to a cluster via v3rpc MemberAdd API (as of v3.4). When starting
// the learner member, its initial view of the cluster created by peerURLs map does not have info
// on whether or not the new member itself is learner. But at this point, a successful match does
// indicate that the new learner member has applied the raftpb.ConfChangeAddLearnerNode entry
// which was used to add the learner itself to the cluster, and therefore it has the correct info
// on learner.
func (c *ClusterV3) waitMembersMatch(t testutil.TB) {
	wMembers := c.getMembers()
	sort.Sort(SortableProtoMemberSliceByPeerURLs(wMembers))
	cli := c.Client(0)
	for {
		resp, err := cli.MemberList(context.Background())
		if err != nil {
			t.Fatalf("failed to list member %v", err)
		}

		if len(resp.Members) != len(wMembers) {
			continue
		}
		sort.Sort(SortableProtoMemberSliceByPeerURLs(resp.Members))
		for _, m := range resp.Members {
			m.ID = 0
		}
		if reflect.DeepEqual(resp.Members, wMembers) {
			return
		}

		time.Sleep(tickDuration)
	}
}

type SortableProtoMemberSliceByPeerURLs []*pb.Member

func (p SortableProtoMemberSliceByPeerURLs) Len() int { return len(p) }
func (p SortableProtoMemberSliceByPeerURLs) Less(i, j int) bool {
	return p[i].PeerURLs[0] < p[j].PeerURLs[0]
}
func (p SortableProtoMemberSliceByPeerURLs) Swap(i, j int) { p[i], p[j] = p[j], p[i] }

// MustNewMember creates a new member instance based on the response of V3 Member Add API.
func (c *ClusterV3) MustNewMember(t testutil.TB, resp *clientv3.MemberAddResponse) *member {
	m := c.mustNewMember(t)
	m.isLearner = resp.Member.IsLearner
	m.NewCluster = false

	m.InitialPeerURLsMap = types.URLsMap{}
	for _, mm := range c.Members {
		m.InitialPeerURLsMap[mm.Name] = mm.PeerURLs
	}
	m.InitialPeerURLsMap[m.Name] = types.MustNewURLs(resp.Member.PeerURLs)
	c.Members = append(c.Members, m)
	return m
}
