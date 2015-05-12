/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package zoo

import (
	"fmt"
	"math"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/detector"
	mesos "github.com/mesos/mesos-go/mesosproto"
)

const (
	// prefix for nodes listed at the ZK URL path
	nodePrefix                    = "info_"
	defaultMinDetectorCyclePeriod = 1 * time.Second
)

// reasonable default for a noop change listener
var ignoreChanged = detector.OnMasterChanged(func(*mesos.MasterInfo) {})

// Detector uses ZooKeeper to detect new leading master.
type MasterDetector struct {
	client     *Client
	leaderNode string

	// for one-time zk client initiation
	bootstrap sync.Once

	// latch: only install, at most, one ignoreChanged listener; see MasterDetector.Detect
	ignoreInstalled int32

	// detection should not signal master change listeners more frequently than this
	minDetectorCyclePeriod time.Duration
}

// Internal constructor function
func NewMasterDetector(zkurls string) (*MasterDetector, error) {
	zkHosts, zkPath, err := parseZk(zkurls)
	if err != nil {
		log.Fatalln("Failed to parse url", err)
		return nil, err
	}

	client, err := newClient(zkHosts, zkPath)
	if err != nil {
		return nil, err
	}

	detector := &MasterDetector{
		client:                 client,
		minDetectorCyclePeriod: defaultMinDetectorCyclePeriod,
	}

	log.V(2).Infoln("Created new detector, watching ", zkHosts, zkPath)
	return detector, nil
}

func parseZk(zkurls string) ([]string, string, error) {
	u, err := url.Parse(zkurls)
	if err != nil {
		log.V(1).Infof("failed to parse url: %v", err)
		return nil, "", err
	}
	if u.Scheme != "zk" {
		return nil, "", fmt.Errorf("invalid url scheme for zk url: '%v'", u.Scheme)
	}
	return strings.Split(u.Host, ","), u.Path, nil
}

// returns a chan that, when closed, indicates termination of the detector
func (md *MasterDetector) Done() <-chan struct{} {
	return md.client.stopped()
}

func (md *MasterDetector) Cancel() {
	md.client.stop()
}

//TODO(jdef) execute async because we don't want to stall our client's event loop? if so
//then we also probably want serial event delivery (aka. delivery via a chan) but then we
//have to deal with chan buffer sizes .. ugh. This is probably the least painful for now.
func (md *MasterDetector) childrenChanged(zkc *Client, path string, obs detector.MasterChanged) {
	log.V(2).Infof("fetching children at path '%v'", path)
	list, err := zkc.list(path)
	if err != nil {
		log.Warning(err)
		return
	}

	topNode := selectTopNode(list)

	if md.leaderNode == topNode {
		log.V(2).Infof("ignoring children-changed event, leader has not changed: %v", path)
		return
	}

	log.V(2).Infof("changing leader node from %s -> %s", md.leaderNode, topNode)
	md.leaderNode = topNode

	var masterInfo *mesos.MasterInfo
	if md.leaderNode != "" {
		data, err := zkc.data(fmt.Sprintf("%s/%s", path, topNode))
		if err != nil {
			log.Errorf("unable to retrieve leader data: %v", err.Error())
			return
		}

		masterInfo = new(mesos.MasterInfo)
		err = proto.Unmarshal(data, masterInfo)
		if err != nil {
			log.Errorf("unable to unmarshall MasterInfo data from zookeeper: %v", err)
			return
		}
	}
	log.V(2).Infof("detected master info: %+v", masterInfo)
	obs.OnMasterChanged(masterInfo)
}

// the first call to Detect will kickstart a connection to zookeeper. a nil change listener may
// be spec'd, result of which is a detector that will still listen for master changes and record
// leaderhip changes internally but no listener would be notified. Detect may be called more than
// once, and each time the spec'd listener will be added to the list of those receiving notifications.
func (md *MasterDetector) Detect(f detector.MasterChanged) (err error) {
	// kickstart zk client connectivity
	md.bootstrap.Do(func() { go md.client.connect() })

	if f == nil {
		// only ever install, at most, one ignoreChanged listener. multiple instances of it
		// just consume resources and generate misleading log messages.
		if !atomic.CompareAndSwapInt32(&md.ignoreInstalled, 0, 1) {
			return
		}
		f = ignoreChanged
	}

	go md.detect(f)
	return nil
}

func (md *MasterDetector) detect(f detector.MasterChanged) {
detectLoop:
	for {
		started := time.Now()
		select {
		case <-md.Done():
			return
		case <-md.client.connections():
			// we let the golang runtime manage our listener list for us, in form of goroutines that
			// callback to the master change notification listen func's
			if watchEnded, err := md.client.watchChildren(currentPath, ChildWatcher(func(zkc *Client, path string) {
				md.childrenChanged(zkc, path, f)
			})); err == nil {
				log.V(2).Infoln("detector listener installed")
				select {
				case <-watchEnded:
					if md.leaderNode != "" {
						log.V(1).Infof("child watch ended, signaling master lost")
						md.leaderNode = ""
						f.OnMasterChanged(nil)
					}
				case <-md.client.stopped():
					return
				}
			} else {
				log.V(1).Infof("child watch ended with error: %v", err)
				continue detectLoop
			}
		}
		// rate-limit master changes
		if elapsed := time.Now().Sub(started); elapsed > 0 {
			log.V(2).Infoln("resting before next detection cycle")
			select {
			case <-md.Done():
				return
			case <-time.After(md.minDetectorCyclePeriod - elapsed): // noop
			}
		}
	}
}

func selectTopNode(list []string) (node string) {
	var leaderSeq uint64 = math.MaxUint64

	for _, v := range list {
		if !strings.HasPrefix(v, nodePrefix) {
			continue // only care about participants
		}
		seqStr := strings.TrimPrefix(v, nodePrefix)
		seq, err := strconv.ParseUint(seqStr, 10, 64)
		if err != nil {
			log.Warningf("unexpected zk node format '%s': %v", seqStr, err)
			continue
		}
		if seq < leaderSeq {
			leaderSeq = seq
			node = v
		}
	}

	if node == "" {
		log.V(3).Infoln("No top node found.")
	} else {
		log.V(3).Infof("Top node selected: '%s'", node)
	}
	return node
}
