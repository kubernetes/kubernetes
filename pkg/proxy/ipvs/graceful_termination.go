//go:build linux
// +build linux

/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package ipvs

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	utilipvs "k8s.io/kubernetes/pkg/proxy/ipvs/util"
)

const (
	rsCheckDeleteInterval = 1 * time.Minute
)

// listItem stores real server information and the process time.
// If nothing special happened, real server will be delete after process time.
type listItem struct {
	VirtualServer *utilipvs.VirtualServer
	RealServer    *utilipvs.RealServer
}

// String return the unique real server name(with virtual server information)
func (g *listItem) String() string {
	return GetUniqueRSName(g.VirtualServer, g.RealServer)
}

// GetUniqueRSName return a string type unique rs name with vs information
func GetUniqueRSName(vs *utilipvs.VirtualServer, rs *utilipvs.RealServer) string {
	return vs.String() + "/" + rs.String()
}

type graceTerminateRSList struct {
	lock sync.Mutex
	list map[string]*listItem
}

// add push an new element to the rsList
func (q *graceTerminateRSList) add(rs *listItem) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	uniqueRS := rs.String()
	if _, ok := q.list[uniqueRS]; ok {
		return false
	}

	klog.V(5).InfoS("Adding real server to graceful delete real server list", "realServer", rs)
	q.list[uniqueRS] = rs
	return true
}

// remove remove an element from the rsList
func (q *graceTerminateRSList) remove(rs *listItem) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	uniqueRS := rs.String()
	if _, ok := q.list[uniqueRS]; ok {
		delete(q.list, uniqueRS)
		return true
	}
	return false
}

// return the size of the list
func (q *graceTerminateRSList) len() int {
	q.lock.Lock()
	defer q.lock.Unlock()

	return len(q.list)
}

func (q *graceTerminateRSList) flushList(handler func(rsToDelete *listItem) (bool, error)) bool {
	q.lock.Lock()
	defer q.lock.Unlock()
	success := true
	for name, rs := range q.list {
		deleted, err := handler(rs)
		if err != nil {
			klog.ErrorS(err, "Error in deleting real server", "realServer", name)
			success = false
		}
		if deleted {
			klog.InfoS("Removed real server from graceful delete real server list", "realServer", name)
			delete(q.list, rs.String())
		}
	}
	return success
}

// exist check whether the specified unique RS is in the rsList
func (q *graceTerminateRSList) exist(uniqueRS string) (*listItem, bool) {
	q.lock.Lock()
	defer q.lock.Unlock()

	if rs, ok := q.list[uniqueRS]; ok {
		return rs, true
	}
	return nil, false
}

// GracefulTerminationManager manage rs graceful termination information and do graceful termination work
// rsList is the rs list to graceful termination, ipvs is the ipvsinterface to do ipvs delete/update work
type GracefulTerminationManager struct {
	rsList graceTerminateRSList
	ipvs   utilipvs.Interface
}

// NewGracefulTerminationManager create a gracefulTerminationManager to manage ipvs rs graceful termination work
func NewGracefulTerminationManager(ipvs utilipvs.Interface) *GracefulTerminationManager {
	l := make(map[string]*listItem)
	return &GracefulTerminationManager{
		rsList: graceTerminateRSList{
			list: l,
		},
		ipvs: ipvs,
	}
}

// InTerminationList to check whether specified unique rs name is in graceful termination list
func (m *GracefulTerminationManager) InTerminationList(uniqueRS string) bool {
	_, exist := m.rsList.exist(uniqueRS)
	return exist
}

// GracefulDeleteRS to update rs weight to 0, and add rs to graceful terminate list
func (m *GracefulTerminationManager) GracefulDeleteRS(vs *utilipvs.VirtualServer, rs *utilipvs.RealServer) error {
	// Try to delete rs before add it to graceful delete list
	ele := &listItem{
		VirtualServer: vs,
		RealServer:    rs,
	}
	deleted, err := m.deleteRsFunc(ele)
	if err != nil {
		klog.ErrorS(err, "Error in deleting real server", "realServer", ele)
	}
	if deleted {
		return nil
	}
	rs.Weight = 0
	err = m.ipvs.UpdateRealServer(vs, rs)
	if err != nil {
		return err
	}
	klog.V(5).InfoS("Adding real server to graceful delete real server list", "realServer", ele)
	m.rsList.add(ele)
	return nil
}

func (m *GracefulTerminationManager) deleteRsFunc(rsToDelete *listItem) (bool, error) {
	klog.V(5).InfoS("Trying to delete real server", "realServer", rsToDelete)
	rss, err := m.ipvs.GetRealServers(rsToDelete.VirtualServer)
	if err != nil {
		return false, err
	}
	for _, rs := range rss {
		if rsToDelete.RealServer.Equal(rs) {
			// For UDP and SCTP traffic, no graceful termination, we immediately delete the RS
			//     (existing connections will be deleted on the next packet because sysctlExpireNoDestConn=1)
			// For other protocols, don't delete until all connections have expired)
			if utilipvs.IsRsGracefulTerminationNeeded(rsToDelete.VirtualServer.Protocol) && rs.ActiveConn+rs.InactiveConn != 0 {
				klog.V(5).InfoS("Skip deleting real server till all connection have expired", "realServer", rsToDelete, "activeConnection", rs.ActiveConn, "inactiveConnection", rs.InactiveConn)
				return false, nil
			}
			klog.V(5).InfoS("Deleting real server", "realServer", rsToDelete)
			err := m.ipvs.DeleteRealServer(rsToDelete.VirtualServer, rs)
			if err != nil {
				return false, fmt.Errorf("delete destination %q err: %w", rs.String(), err)
			}
			return true, nil
		}
	}
	return true, fmt.Errorf("failed to delete rs %q, can't find the real server", rsToDelete.String())
}

func (m *GracefulTerminationManager) tryDeleteRs() {
	if !m.rsList.flushList(m.deleteRsFunc) {
		klog.ErrorS(nil, "Try flush graceful termination list error")
	}
}

// MoveRSOutofGracefulDeleteList to delete an rs and remove it from the rsList immediately
func (m *GracefulTerminationManager) MoveRSOutofGracefulDeleteList(uniqueRS string) error {
	rsToDelete, find := m.rsList.exist(uniqueRS)
	if !find || rsToDelete == nil {
		return fmt.Errorf("failed to find rs: %q", uniqueRS)
	}
	err := m.ipvs.DeleteRealServer(rsToDelete.VirtualServer, rsToDelete.RealServer)
	if err != nil {
		return err
	}
	m.rsList.remove(rsToDelete)
	return nil
}

// Run start a goroutine to try to delete rs in the graceful delete rsList with an interval 1 minute
func (m *GracefulTerminationManager) Run() {
	go wait.Until(m.tryDeleteRs, rsCheckDeleteInterval, wait.NeverStop)
}
