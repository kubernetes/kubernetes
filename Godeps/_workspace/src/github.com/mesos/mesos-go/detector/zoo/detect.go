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
	"encoding/json"
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
	nodeJSONPrefix                = "json.info_"
	defaultMinDetectorCyclePeriod = 1 * time.Second
)

// reasonable default for a noop change listener
var ignoreChanged = detector.OnMasterChanged(func(*mesos.MasterInfo) {})

type zkInterface interface {
	stopped() <-chan struct{}
	stop()
	data(string) ([]byte, error)
	watchChildren(string) (string, <-chan []string, <-chan error)
}

type infoCodec func(path, node string) (*mesos.MasterInfo, error)

// Detector uses ZooKeeper to detect new leading master.
type MasterDetector struct {
	client     zkInterface
	leaderNode string

	bootstrapLock sync.RWMutex // guard against concurrent invocations of bootstrapFunc
	bootstrapFunc func() error // for one-time zk client initiation

	// latch: only install, at most, one ignoreChanged listener; see MasterDetector.Detect
	ignoreInstalled int32

	// detection should not signal master change listeners more frequently than this
	minDetectorCyclePeriod time.Duration
	done                   chan struct{}
	cancel                 func()
}

// Internal constructor function
func NewMasterDetector(zkurls string) (*MasterDetector, error) {
	zkHosts, zkPath, err := parseZk(zkurls)
	if err != nil {
		log.Fatalln("Failed to parse url", err)
		return nil, err
	}

	detector := &MasterDetector{
		minDetectorCyclePeriod: defaultMinDetectorCyclePeriod,
		done:   make(chan struct{}),
		cancel: func() {},
	}

	detector.bootstrapFunc = func() (err error) {
		if detector.client == nil {
			detector.client, err = connect2(zkHosts, zkPath)
		}
		return
	}

	log.V(2).Infoln("Created new detector to watch", zkHosts, zkPath)
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
	return md.done
}

func (md *MasterDetector) Cancel() {
	md.bootstrapLock.RLock()
	defer md.bootstrapLock.RUnlock()
	md.cancel()
}

func (md *MasterDetector) childrenChanged(path string, list []string, obs detector.MasterChanged) {
	md.notifyMasterChanged(path, list, obs)
	md.notifyAllMasters(path, list, obs)
}

func (md *MasterDetector) notifyMasterChanged(path string, list []string, obs detector.MasterChanged) {
	// mesos v0.24 writes JSON only, v0.23 writes json and protobuf, v0.22 and prior only write protobuf
	topNode, codec := md.selectTopNode(list)
	if md.leaderNode == topNode {
		log.V(2).Infof("ignoring children-changed event, leader has not changed: %v", path)
		return
	}

	log.V(2).Infof("changing leader node from %q -> %q", md.leaderNode, topNode)
	md.leaderNode = topNode

	var masterInfo *mesos.MasterInfo
	if md.leaderNode != "" {
		var err error
		if masterInfo, err = codec(path, topNode); err != nil {
			log.Errorln(err.Error())
		}
	}
	log.V(2).Infof("detected master info: %+v", masterInfo)
	logPanic(func() { obs.OnMasterChanged(masterInfo) })
}

// logPanic safely executes the given func, recovering from and logging a panic if one occurs.
func logPanic(f func()) {
	defer func() {
		if r := recover(); r != nil {
			log.Errorf("recovered from client panic: %v", r)
		}
	}()
	f()
}

func (md *MasterDetector) pullMasterInfo(path, node string) (*mesos.MasterInfo, error) {
	data, err := md.client.data(fmt.Sprintf("%s/%s", path, node))
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve leader data: %v", err)
	}

	masterInfo := &mesos.MasterInfo{}
	err = proto.Unmarshal(data, masterInfo)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal protobuf MasterInfo data from zookeeper: %v", err)
	}
	return masterInfo, nil
}

func (md *MasterDetector) pullMasterJsonInfo(path, node string) (*mesos.MasterInfo, error) {
	data, err := md.client.data(fmt.Sprintf("%s/%s", path, node))
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve leader data: %v", err)
	}

	masterInfo := &mesos.MasterInfo{}
	err = json.Unmarshal(data, masterInfo)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal json MasterInfo data from zookeeper: %v", err)
	}
	return masterInfo, nil
}

func (md *MasterDetector) notifyAllMasters(path string, list []string, obs detector.MasterChanged) {
	all, ok := obs.(detector.AllMasters)
	if !ok {
		// not interested in entire master list
		return
	}

	// mesos v0.24 writes JSON only, v0.23 writes json and protobuf, v0.22 and prior only write protobuf
	masters := map[string]*mesos.MasterInfo{}
	tryStore := func(node string, codec infoCodec) {
		info, err := codec(path, node)
		if err != nil {
			log.Errorln(err.Error())
		} else {
			masters[info.GetId()] = info
		}
	}
	for _, node := range list {
		// compare https://github.com/apache/mesos/blob/0.23.0/src/master/detector.cpp#L437
		if strings.HasPrefix(node, nodePrefix) {
			tryStore(node, md.pullMasterInfo)
		} else if strings.HasPrefix(node, nodeJSONPrefix) {
			tryStore(node, md.pullMasterJsonInfo)
		} else {
			continue
		}
	}
	masterList := make([]*mesos.MasterInfo, 0, len(masters))
	for _, v := range masters {
		masterList = append(masterList, v)
	}

	log.V(2).Infof("notifying of master membership change: %+v", masterList)
	logPanic(func() { all.UpdatedMasters(masterList) })
}

func (md *MasterDetector) callBootstrap() (e error) {
	log.V(2).Infoln("invoking detector boostrap")
	md.bootstrapLock.Lock()
	defer md.bootstrapLock.Unlock()

	clientConfigured := md.client != nil
	if e = md.bootstrapFunc(); e == nil && !clientConfigured && md.client != nil {
		// chain the lifetime of this detector to that of the newly created client impl
		client := md.client
		md.cancel = client.stop
		go func() {
			defer close(md.done)
			<-client.stopped()
		}()
	}
	return
}

// the first call to Detect will kickstart a connection to zookeeper. a nil change listener may
// be spec'd, result of which is a detector that will still listen for master changes and record
// leaderhip changes internally but no listener would be notified. Detect may be called more than
// once, and each time the spec'd listener will be added to the list of those receiving notifications.
func (md *MasterDetector) Detect(f detector.MasterChanged) (err error) {
	// kickstart zk client connectivity
	if err := md.callBootstrap(); err != nil {
		log.V(3).Infoln("failed to execute bootstrap function", err.Error())
		return err
	}

	if f == nil {
		// only ever install, at most, one ignoreChanged listener. multiple instances of it
		// just consume resources and generate misleading log messages.
		if !atomic.CompareAndSwapInt32(&md.ignoreInstalled, 0, 1) {
			log.V(3).Infoln("ignoreChanged listener already installed")
			return
		}
		f = ignoreChanged
	}

	log.V(3).Infoln("spawning detect()")
	go md.detect(f)
	return nil
}

func (md *MasterDetector) detect(f detector.MasterChanged) {
	log.V(3).Infoln("detecting children at", currentPath)
detectLoop:
	for {
		select {
		case <-md.Done():
			return
		default:
		}
		log.V(3).Infoln("watching children at", currentPath)
		path, childrenCh, errCh := md.client.watchChildren(currentPath)
		rewatch := false
		for {
			started := time.Now()
			select {
			case children := <-childrenCh:
				md.childrenChanged(path, children, f)
			case err, ok := <-errCh:
				// check for a tie first (required for predictability (tests)); the downside of
				// doing this is that a listener might get two callbacks back-to-back ("new leader",
				// followed by "no leader").
				select {
				case children := <-childrenCh:
					md.childrenChanged(path, children, f)
				default:
				}
				if ok {
					log.V(1).Infoln("child watch ended with error, master lost; error was:", err.Error())
				} else {
					// detector shutdown likely...
					log.V(1).Infoln("child watch ended, master lost")
				}
				select {
				case <-md.Done():
					return
				default:
					if md.leaderNode != "" {
						log.V(2).Infof("changing leader node from %q -> \"\"", md.leaderNode)
						md.leaderNode = ""
						f.OnMasterChanged(nil)
					}
				}
				rewatch = true
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
			if rewatch {
				continue detectLoop
			}
		}
	}
}

func (md *MasterDetector) selectTopNode(list []string) (topNode string, codec infoCodec) {
	// mesos v0.24 writes JSON only, v0.23 writes json and protobuf, v0.22 and prior only write protobuf
	topNode = selectTopNodePrefix(list, nodeJSONPrefix)
	codec = md.pullMasterJsonInfo
	if topNode == "" {
		topNode = selectTopNodePrefix(list, nodePrefix)
		codec = md.pullMasterInfo

		if topNode != "" {
			log.Warningf("Leading master is using a Protobuf binary format when registering "+
				"with Zookeeper (%s): this will be deprecated as of Mesos 0.24 (see MESOS-2340).",
				topNode)
		}
	}
	return
}

func selectTopNodePrefix(list []string, pre string) (node string) {
	var leaderSeq uint64 = math.MaxUint64

	for _, v := range list {
		if !strings.HasPrefix(v, pre) {
			continue // only care about participants
		}
		seqStr := strings.TrimPrefix(v, pre)
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
