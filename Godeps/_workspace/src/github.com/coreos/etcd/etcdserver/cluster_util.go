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

package etcdserver

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"sort"
	"time"

	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
)

// isMemberBootstrapped tries to check if the given member has been bootstrapped
// in the given cluster.
func isMemberBootstrapped(cl *cluster, member string, tr *http.Transport) bool {
	rcl, err := getClusterFromRemotePeers(getRemotePeerURLs(cl, member), time.Second, false, tr)
	if err != nil {
		return false
	}
	id := cl.MemberByName(member).ID
	m := rcl.Member(id)
	if m == nil {
		return false
	}
	if len(m.ClientURLs) > 0 {
		return true
	}
	return false
}

// GetClusterFromRemotePeers takes a set of URLs representing etcd peers, and
// attempts to construct a Cluster by accessing the members endpoint on one of
// these URLs. The first URL to provide a response is used. If no URLs provide
// a response, or a Cluster cannot be successfully created from a received
// response, an error is returned.
// Each request has a 10-second timeout. Because the upper limit of TTL is 5s,
// 10 second is enough for building connection and finishing request.
func GetClusterFromRemotePeers(urls []string, tr *http.Transport) (*cluster, error) {
	return getClusterFromRemotePeers(urls, 10*time.Second, true, tr)
}

// If logerr is true, it prints out more error messages.
func getClusterFromRemotePeers(urls []string, timeout time.Duration, logerr bool, tr *http.Transport) (*cluster, error) {
	cc := &http.Client{
		Transport: tr,
		Timeout:   timeout,
	}
	for _, u := range urls {
		resp, err := cc.Get(u + "/members")
		if err != nil {
			if logerr {
				plog.Warningf("could not get cluster response from %s: %v", u, err)
			}
			continue
		}
		b, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			if logerr {
				plog.Warningf("could not read the body of cluster response: %v", err)
			}
			continue
		}
		var membs []*Member
		if err := json.Unmarshal(b, &membs); err != nil {
			if logerr {
				plog.Warningf("could not unmarshal cluster response: %v", err)
			}
			continue
		}
		id, err := types.IDFromString(resp.Header.Get("X-Etcd-Cluster-ID"))
		if err != nil {
			if logerr {
				plog.Warningf("could not parse the cluster ID from cluster res: %v", err)
			}
			continue
		}
		return newClusterFromMembers("", id, membs), nil
	}
	return nil, fmt.Errorf("could not retrieve cluster information from the given urls")
}

// getRemotePeerURLs returns peer urls of remote members in the cluster. The
// returned list is sorted in ascending lexicographical order.
func getRemotePeerURLs(cl Cluster, local string) []string {
	us := make([]string, 0)
	for _, m := range cl.Members() {
		if m.Name == local {
			continue
		}
		us = append(us, m.PeerURLs...)
	}
	sort.Strings(us)
	return us
}

// getVersions returns the versions of the members in the given cluster.
// The key of the returned map is the member's ID. The value of the returned map
// is the semver versions string, including server and cluster.
// If it fails to get the version of a member, the key will be nil.
func getVersions(cl Cluster, local types.ID, tr *http.Transport) map[string]*version.Versions {
	members := cl.Members()
	vers := make(map[string]*version.Versions)
	for _, m := range members {
		if m.ID == local {
			cv := "not_decided"
			if cl.Version() != nil {
				cv = cl.Version().String()
			}
			vers[m.ID.String()] = &version.Versions{Server: version.Version, Cluster: cv}
			continue
		}
		ver, err := getVersion(m, tr)
		if err != nil {
			plog.Warningf("cannot get the version of member %s (%v)", m.ID, err)
			vers[m.ID.String()] = nil
		} else {
			vers[m.ID.String()] = ver
		}
	}
	return vers
}

// decideClusterVersion decides the cluster version based on the versions map.
// The returned version is the min server version in the map, or nil if the min
// version in unknown.
func decideClusterVersion(vers map[string]*version.Versions) *semver.Version {
	var cv *semver.Version
	lv := semver.Must(semver.NewVersion(version.Version))

	for mid, ver := range vers {
		if ver == nil {
			return nil
		}
		v, err := semver.NewVersion(ver.Server)
		if err != nil {
			plog.Errorf("cannot understand the version of member %s (%v)", mid, err)
			return nil
		}
		if lv.LessThan(*v) {
			plog.Warningf("the local etcd version %s is not up-to-date", lv.String())
			plog.Warningf("member %s has a higher version %s", mid, ver.Server)
		}
		if cv == nil {
			cv = v
		} else if v.LessThan(*cv) {
			cv = v
		}
	}
	return cv
}

// isCompatibleWithCluster return true if the local member has a compitable version with
// the current running cluster.
// The version is considered as compitable when at least one of the other members in the cluster has a
// cluster version in the range of [MinClusterVersion, Version] and no known members has a cluster version
// out of the range.
// We set this rule since when the local member joins, another member might be offline.
func isCompatibleWithCluster(cl Cluster, local types.ID, tr *http.Transport) bool {
	vers := getVersions(cl, local, tr)
	minV := semver.Must(semver.NewVersion(version.MinClusterVersion))
	maxV := semver.Must(semver.NewVersion(version.Version))
	maxV = &semver.Version{
		Major: maxV.Major,
		Minor: maxV.Minor,
	}

	return isCompatibleWithVers(vers, local, minV, maxV)
}

func isCompatibleWithVers(vers map[string]*version.Versions, local types.ID, minV, maxV *semver.Version) bool {
	var ok bool
	for id, v := range vers {
		// ignore comparasion with local version
		if id == local.String() {
			continue
		}
		if v == nil {
			continue
		}
		clusterv, err := semver.NewVersion(v.Cluster)
		if err != nil {
			plog.Errorf("cannot understand the cluster version of member %s (%v)", id, err)
			continue
		}
		if clusterv.LessThan(*minV) {
			plog.Warningf("the running cluster version(%v) is lower than the minimal cluster version(%v) supported", clusterv.String(), minV.String())
			return false
		}
		if maxV.LessThan(*clusterv) {
			plog.Warningf("the running cluster version(%v) is higher than the maximum cluster version(%v) supported", clusterv.String(), maxV.String())
			return false
		}
		ok = true
	}
	return ok
}

// getVersion returns the Versions of the given member via its
// peerURLs. Returns the last error if it fails to get the version.
func getVersion(m *Member, tr *http.Transport) (*version.Versions, error) {
	cc := &http.Client{
		Transport: tr,
	}
	var (
		err  error
		resp *http.Response
	)

	for _, u := range m.PeerURLs {
		resp, err = cc.Get(u + "/version")
		if err != nil {
			plog.Warningf("failed to reach the peerURL(%s) of member %s (%v)", u, m.ID, err)
			continue
		}
		// etcd 2.0 does not have version endpoint on peer url.
		if resp.StatusCode == http.StatusNotFound {
			resp.Body.Close()
			return &version.Versions{
				Server:  "2.0.0",
				Cluster: "2.0.0",
			}, nil
		}

		var b []byte
		b, err = ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			plog.Warningf("failed to read out the response body from the peerURL(%s) of member %s (%v)", u, m.ID, err)
			continue
		}
		var vers version.Versions
		if err := json.Unmarshal(b, &vers); err != nil {
			plog.Warningf("failed to unmarshal the response body got from the peerURL(%s) of member %s (%v)", u, m.ID, err)
			continue
		}
		return &vers, nil
	}
	return nil, err
}

func MustDetectDowngrade(cv *semver.Version) {
	lv := semver.Must(semver.NewVersion(version.Version))
	// only keep major.minor version for comparison against cluster version
	lv = &semver.Version{Major: lv.Major, Minor: lv.Minor}
	if cv != nil && lv.LessThan(*cv) {
		plog.Fatalf("cluster cannot be downgraded (current version: %s is lower than determined cluster version: %s).", version.Version, version.Cluster(cv.String()))
	}
}
