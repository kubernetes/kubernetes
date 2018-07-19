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
	"container/list"
	"sync"
	"time"

	"fmt"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
)

const (
	rsGracefulDeletePeriod = 15 * time.Minute
	rsCheckDeleteInterval  = 1 * time.Minute
)

// listItem stores real server information and the process time.
// If nothing special happened, real server will be delete after process time.
type listItem struct {
	VirtualServer *utilipvs.VirtualServer
	RealServer    *utilipvs.RealServer
	ProcessAt     time.Time
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
	list *list.List
	set  sets.String
}

// add push an new element to the rsList
func (q *graceTerminateRSList) add(rs *listItem) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	uniqueRS := rs.String()
	if q.set.Has(uniqueRS) {
		return false
	}
	glog.V(5).Infof("Pushing rs %v to graceful delete rsList: %+v", rs)

	q.list.PushBack(rs)
	q.set.Insert(uniqueRS)
	return true
}

// remove remove an element from the rsList
func (q *graceTerminateRSList) remove(rs *listItem) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	uniqueRS := rs.String()
	if !q.set.Has(uniqueRS) {
		return false
	}
	q.set.Delete(uniqueRS)
	for e := q.list.Front(); e.Next() == nil; e = e.Next() {
		val := e.Value.(*listItem)
		if val.String() == uniqueRS {
			q.list.Remove(e)
			return true
		}
	}
	return false
}

// head return the first element from the rsList
func (q *graceTerminateRSList) head() (*listItem, bool) {
	q.lock.Lock()
	defer q.lock.Unlock()
	if q.list.Len() == 0 {
		return nil, false
	}
	result := q.list.Front().Value.(*listItem)
	return result, true
}

// exist check whether the specified unique RS is in the rsList
func (q *graceTerminateRSList) exist(uniqueRS string) (*listItem, bool) {
	q.lock.Lock()
	defer q.lock.Unlock()

	if !q.set.Has(uniqueRS) {
		return nil, false
	}
	for e := q.list.Front(); e.Next() == nil; e = e.Next() {
		val := e.Value.(*listItem)
		if val.String() == uniqueRS {
			return val, true
		}
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
	l := list.New()
	l.Init()
	return &GracefulTerminationManager{
		rsList: graceTerminateRSList{
			list: l,
			set:  sets.NewString(),
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
	rs.Weight = 0
	err := m.ipvs.UpdateRealServer(vs, rs)
	if err != nil {
		return err
	}

	ele := &listItem{
		VirtualServer: vs,
		RealServer:    rs,
		ProcessAt:     time.Now().Add(rsGracefulDeletePeriod),
	}
	glog.V(5).Infof("Adding an element to graceful delete rsList: %+v", ele)
	m.rsList.add(ele)
	return nil
}

func (m *GracefulTerminationManager) tryDeleteRs() {
	for {
		rsToDelete, _ := m.rsList.head()
		glog.V(5).Infof("Trying to delete rs")
		if rsToDelete == nil || rsToDelete.ProcessAt.After(time.Now()) {
			break
		}

		glog.V(5).Infof("Deleting rs: %s", rsToDelete.String())
		err := m.ipvs.DeleteRealServer(rsToDelete.VirtualServer, rsToDelete.RealServer)
		if err != nil {
			glog.Errorf("Failed to delete destination: %v, error: %v", rsToDelete.RealServer, err)
		}
		if !m.rsList.remove(rsToDelete) {
			glog.Errorf("Failed to pop out rsList.")
		}
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
	// before start, add leftover in delete rs to graceful delete rsList
	vss, err := m.ipvs.GetVirtualServers()
	if err != nil {
		glog.Errorf("IPVS graceful delete manager failed to get IPVS virtualserver")
	}
	for _, vs := range vss {
		rss, err := m.ipvs.GetRealServers(vs)
		if err != nil {
			glog.Errorf("IPVS graceful delete manager failed to get %v realserver", vs)
			continue
		}
		for _, rs := range rss {
			if rs.Weight == 0 {
				ele := &listItem{
					VirtualServer: vs,
					RealServer:    rs,
					ProcessAt:     time.Now().Add(rsGracefulDeletePeriod),
				}
				m.rsList.add(ele)
			}
		}
	}

	go wait.Until(m.tryDeleteRs, rsCheckDeleteInterval, wait.NeverStop)
}
