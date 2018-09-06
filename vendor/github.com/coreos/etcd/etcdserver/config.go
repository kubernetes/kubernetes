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

package etcdserver

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/net/context"

	"github.com/coreos/etcd/pkg/netutil"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/etcd/pkg/types"
)

// ServerConfig holds the configuration of etcd as taken from the command line or discovery.
type ServerConfig struct {
	Name           string
	DiscoveryURL   string
	DiscoveryProxy string
	ClientURLs     types.URLs
	PeerURLs       types.URLs
	DataDir        string
	// DedicatedWALDir config will make the etcd to write the WAL to the WALDir
	// rather than the dataDir/member/wal.
	DedicatedWALDir     string
	SnapCount           uint64
	MaxSnapFiles        uint
	MaxWALFiles         uint
	InitialPeerURLsMap  types.URLsMap
	InitialClusterToken string
	NewCluster          bool
	ForceNewCluster     bool
	PeerTLSInfo         transport.TLSInfo

	TickMs        uint
	ElectionTicks int

	// InitialElectionTickAdvance is true, then local member fast-forwards
	// election ticks to speed up "initial" leader election trigger. This
	// benefits the case of larger election ticks. For instance, cross
	// datacenter deployment may require longer election timeout of 10-second.
	// If true, local node does not need wait up to 10-second. Instead,
	// forwards its election ticks to 8-second, and have only 2-second left
	// before leader election.
	//
	// Major assumptions are that:
	//  - cluster has no active leader thus advancing ticks enables faster
	//    leader election, or
	//  - cluster already has an established leader, and rejoining follower
	//    is likely to receive heartbeats from the leader after tick advance
	//    and before election timeout.
	//
	// However, when network from leader to rejoining follower is congested,
	// and the follower does not receive leader heartbeat within left election
	// ticks, disruptive election has to happen thus affecting cluster
	// availabilities.
	//
	// Disabling this would slow down initial bootstrap process for cross
	// datacenter deployments. Make your own tradeoffs by configuring
	// --initial-election-tick-advance at the cost of slow initial bootstrap.
	//
	// If single-node, it advances ticks regardless.
	//
	// See https://github.com/coreos/etcd/issues/9333 for more detail.
	InitialElectionTickAdvance bool

	BootstrapTimeout time.Duration

	AutoCompactionRetention int
	QuotaBackendBytes       int64

	// MaxRequestBytes is the maximum request size to send over raft.
	MaxRequestBytes uint

	StrictReconfigCheck bool

	// ClientCertAuthEnabled is true when cert has been signed by the client CA.
	ClientCertAuthEnabled bool

	AuthToken string

	Debug bool
}

// VerifyBootstrap sanity-checks the initial config for bootstrap case
// and returns an error for things that should never happen.
func (c *ServerConfig) VerifyBootstrap() error {
	if err := c.hasLocalMember(); err != nil {
		return err
	}
	if err := c.advertiseMatchesCluster(); err != nil {
		return err
	}
	if checkDuplicateURL(c.InitialPeerURLsMap) {
		return fmt.Errorf("initial cluster %s has duplicate url", c.InitialPeerURLsMap)
	}
	if c.InitialPeerURLsMap.String() == "" && c.DiscoveryURL == "" {
		return fmt.Errorf("initial cluster unset and no discovery URL found")
	}
	return nil
}

// VerifyJoinExisting sanity-checks the initial config for join existing cluster
// case and returns an error for things that should never happen.
func (c *ServerConfig) VerifyJoinExisting() error {
	// The member has announced its peer urls to the cluster before starting; no need to
	// set the configuration again.
	if err := c.hasLocalMember(); err != nil {
		return err
	}
	if checkDuplicateURL(c.InitialPeerURLsMap) {
		return fmt.Errorf("initial cluster %s has duplicate url", c.InitialPeerURLsMap)
	}
	if c.DiscoveryURL != "" {
		return fmt.Errorf("discovery URL should not be set when joining existing initial cluster")
	}
	return nil
}

// hasLocalMember checks that the cluster at least contains the local server.
func (c *ServerConfig) hasLocalMember() error {
	if urls := c.InitialPeerURLsMap[c.Name]; urls == nil {
		return fmt.Errorf("couldn't find local name %q in the initial cluster configuration", c.Name)
	}
	return nil
}

// advertiseMatchesCluster confirms peer URLs match those in the cluster peer list.
func (c *ServerConfig) advertiseMatchesCluster() error {
	urls, apurls := c.InitialPeerURLsMap[c.Name], c.PeerURLs.StringSlice()
	urls.Sort()
	sort.Strings(apurls)
	ctx, cancel := context.WithTimeout(context.TODO(), 30*time.Second)
	defer cancel()
	if !netutil.URLStringsEqual(ctx, apurls, urls.StringSlice()) {
		umap := map[string]types.URLs{c.Name: c.PeerURLs}
		return fmt.Errorf("--initial-cluster must include %s given --initial-advertise-peer-urls=%s", types.URLsMap(umap).String(), strings.Join(apurls, ","))
	}
	return nil
}

func (c *ServerConfig) MemberDir() string { return filepath.Join(c.DataDir, "member") }

func (c *ServerConfig) WALDir() string {
	if c.DedicatedWALDir != "" {
		return c.DedicatedWALDir
	}
	return filepath.Join(c.MemberDir(), "wal")
}

func (c *ServerConfig) SnapDir() string { return filepath.Join(c.MemberDir(), "snap") }

func (c *ServerConfig) ShouldDiscover() bool { return c.DiscoveryURL != "" }

// ReqTimeout returns timeout for request to finish.
func (c *ServerConfig) ReqTimeout() time.Duration {
	// 5s for queue waiting, computation and disk IO delay
	// + 2 * election timeout for possible leader election
	return 5*time.Second + 2*time.Duration(c.ElectionTicks)*time.Duration(c.TickMs)*time.Millisecond
}

func (c *ServerConfig) electionTimeout() time.Duration {
	return time.Duration(c.ElectionTicks) * time.Duration(c.TickMs) * time.Millisecond
}

func (c *ServerConfig) peerDialTimeout() time.Duration {
	// 1s for queue wait and system delay
	// + one RTT, which is smaller than 1/5 election timeout
	return time.Second + time.Duration(c.ElectionTicks)*time.Duration(c.TickMs)*time.Millisecond/5
}

func (c *ServerConfig) PrintWithInitial() { c.print(true) }

func (c *ServerConfig) Print() { c.print(false) }

func (c *ServerConfig) print(initial bool) {
	plog.Infof("name = %s", c.Name)
	if c.ForceNewCluster {
		plog.Infof("force new cluster")
	}
	plog.Infof("data dir = %s", c.DataDir)
	plog.Infof("member dir = %s", c.MemberDir())
	if c.DedicatedWALDir != "" {
		plog.Infof("dedicated WAL dir = %s", c.DedicatedWALDir)
	}
	plog.Infof("heartbeat = %dms", c.TickMs)
	plog.Infof("election = %dms", c.ElectionTicks*int(c.TickMs))
	plog.Infof("snapshot count = %d", c.SnapCount)
	if len(c.DiscoveryURL) != 0 {
		plog.Infof("discovery URL= %s", c.DiscoveryURL)
		if len(c.DiscoveryProxy) != 0 {
			plog.Infof("discovery proxy = %s", c.DiscoveryProxy)
		}
	}
	plog.Infof("advertise client URLs = %s", c.ClientURLs)
	if initial {
		plog.Infof("initial advertise peer URLs = %s", c.PeerURLs)
		plog.Infof("initial cluster = %s", c.InitialPeerURLsMap)
	}
}

func checkDuplicateURL(urlsmap types.URLsMap) bool {
	um := make(map[string]bool)
	for _, urls := range urlsmap {
		for _, url := range urls {
			u := url.String()
			if um[u] {
				return true
			}
			um[u] = true
		}
	}
	return false
}

func (c *ServerConfig) bootstrapTimeout() time.Duration {
	if c.BootstrapTimeout != 0 {
		return c.BootstrapTimeout
	}
	return time.Second
}

func (c *ServerConfig) backendPath() string { return filepath.Join(c.SnapDir(), "db") }
