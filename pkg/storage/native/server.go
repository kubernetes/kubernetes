/*
Copyright 2016 The Kubernetes Authors.

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

package native

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/golang/glog"
	"github.com/hashicorp/raft"
	"github.com/hashicorp/raft-boltdb"
	"google.golang.org/grpc"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

// ServerOptions contains Options for a state server
type ServerOptions struct {
	EnableSingle        bool
	RaftBind            string
	ClientBind          string
	RaftDir             string
	RetainSnapshotCount int32
}

func (s *ServerOptions) InitDefaults() {
	s.EnableSingle = true
	s.RaftBind = "127.0.0.1:12000"
	s.ClientBind = "127.0.0.1:12001"
	s.RaftDir = "/var/run/kubernetes/stateserver"
	s.RetainSnapshotCount = 5
}

type Server struct {
	options *ServerOptions
	started int32
	raft    *raft.Raft
}

func NewServer(options *ServerOptions) *Server {
	return &Server{
		options: options,
	}
}

func (s *Server) IsStarted() bool {
	return 1 == atomic.LoadInt32(&s.started)
}

func (s *Server) IsLeader() bool {
	r := s.raft
	return r != nil && r.State() == raft.Leader
}

func (s *Server) Run() error {
	o := s.options

	// Setup Raft configuration.
	config := raft.DefaultConfig()

	// Check for any existing peers.
	peers, err := readPeersJSON(filepath.Join(o.RaftDir, "peers.json"))
	if err != nil {
		return err
	}

	// Allow the node to entry single-mode, potentially electing itself, if
	// explicitly enabled and there is only 1 node in the cluster already.
	if o.EnableSingle && len(peers) <= 1 {
		glog.Infof("enabling single-node mode")
		config.EnableSingleNode = true
		config.DisableBootstrapAfterElect = false
	}

	addr, err := net.ResolveTCPAddr("tcp", o.RaftBind)
	if err != nil {
		return err
	}
	transport, err := raft.NewTCPTransport(o.RaftBind, addr, 3, 10*time.Second, os.Stderr)
	if err != nil {
		return err
	}

	// Create peer storage.
	peerStore := raft.NewJSONPeers(o.RaftDir, transport)

	// Create the snapshot store. This allows the Raft to truncate the log.
	snapshots, err := raft.NewFileSnapshotStore(o.RaftDir, int(o.RetainSnapshotCount), os.Stderr)
	if err != nil {
		return fmt.Errorf("file snapshot store: %s", err)
	}

	// Create the log store and stable store.
	logStore, err := raftboltdb.NewBoltStore(filepath.Join(o.RaftDir, "raft.db"))
	if err != nil {
		return fmt.Errorf("new bolt store: %s", err)
	}

	fsm := NewRaftFSM(logStore)

	// Instantiate the Raft systems.
	raft, err := raft.NewRaft(config, fsm, logStore, logStore, snapshots, peerStore, transport)
	if err != nil {
		return fmt.Errorf("error creating raft server: %v", err)
	}
	s.raft = raft
	backend := NewRaftBackend(raft, fsm)

	// TODO: Reuse primary grpc server?
	grpcBindAddress := o.ClientBind
	lis, err := net.Listen("tcp", grpcBindAddress)
	if err != nil {
		return fmt.Errorf("failed to listen on %q: %v", grpcBindAddress, err)
	}

	grpcServer := grpc.NewServer()
	RegisterStorageServiceServer(grpcServer, backend)
	atomic.StoreInt32(&s.started, 1)
	err = grpcServer.Serve(lis)

	return err
}

func readPeersJSON(path string) ([]string, error) {
	b, err := ioutil.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	if len(b) == 0 {
		return nil, nil
	}

	var peers []string
	dec := json.NewDecoder(bytes.NewReader(b))
	if err := dec.Decode(&peers); err != nil {
		return nil, err
	}

	return peers, nil
}
