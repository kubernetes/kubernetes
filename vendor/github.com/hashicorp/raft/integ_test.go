package raft

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"testing"
	"time"
)

// CheckInteg will skip a test if integration testing is not enabled.
func CheckInteg(t *testing.T) {
	if !IsInteg() {
		t.SkipNow()
	}
}

// IsInteg returns a boolean telling you if we're in integ testing mode.
func IsInteg() bool {
	return os.Getenv("INTEG_TESTS") != ""
}

type RaftEnv struct {
	dir      string
	conf     *Config
	fsm      *MockFSM
	store    *InmemStore
	snapshot *FileSnapshotStore
	peers    *JSONPeers
	trans    *NetworkTransport
	raft     *Raft
}

func (r *RaftEnv) Release() {
	log.Printf("[WARN] Release node at %v", r.raft.localAddr)
	f := r.raft.Shutdown()
	if err := f.Error(); err != nil {
		panic(err)
	}
	r.trans.Close()
	os.RemoveAll(r.dir)
}

func MakeRaft(t *testing.T, conf *Config) *RaftEnv {
	env := &RaftEnv{}

	// Set the config
	if conf == nil {
		conf = inmemConfig(t)
	}
	env.conf = conf

	dir, err := ioutil.TempDir("", "raft")
	if err != nil {
		t.Fatalf("err: %v ", err)
	}
	env.dir = dir

	stable := NewInmemStore()
	env.store = stable

	snap, err := NewFileSnapshotStore(dir, 3, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	env.snapshot = snap

	env.fsm = &MockFSM{}

	trans, err := NewTCPTransport("127.0.0.1:0", nil, 2, time.Second, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	env.trans = trans

	env.peers = NewJSONPeers(dir, trans)

	log.Printf("[INFO] Starting node at %v", trans.LocalAddr())
	raft, err := NewRaft(conf, env.fsm, stable, stable, snap, env.peers, trans)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	env.raft = raft
	return env
}

func WaitFor(env *RaftEnv, state RaftState) error {
	limit := time.Now().Add(200 * time.Millisecond)
	for env.raft.State() != state {
		if time.Now().Before(limit) {
			time.Sleep(10 * time.Millisecond)
		} else {
			return fmt.Errorf("failed to transition to state %v", state)
		}
	}
	return nil
}

func WaitForAny(state RaftState, envs []*RaftEnv) (*RaftEnv, error) {
	limit := time.Now().Add(200 * time.Millisecond)
CHECK:
	for _, env := range envs {
		if env.raft.State() == state {
			return env, nil
		}
	}
	if time.Now().Before(limit) {
		goto WAIT
	}
	return nil, fmt.Errorf("failed to find node in %v state", state)
WAIT:
	time.Sleep(10 * time.Millisecond)
	goto CHECK
}

func WaitFuture(f Future, t *testing.T) error {
	timer := time.AfterFunc(200*time.Millisecond, func() {
		panic(fmt.Errorf("timeout waiting for future %v", f))
	})
	defer timer.Stop()
	return f.Error()
}

func NoErr(err error, t *testing.T) {
	if err != nil {
		t.Fatalf("err: %v", err)
	}
}

func CheckConsistent(envs []*RaftEnv, t *testing.T) {
	limit := time.Now().Add(400 * time.Millisecond)
	first := envs[0]
	var err error
CHECK:
	l1 := len(first.fsm.logs)
	for i := 1; i < len(envs); i++ {
		env := envs[i]
		l2 := len(env.fsm.logs)
		if l1 != l2 {
			err = fmt.Errorf("log length mismatch %d %d", l1, l2)
			goto ERR
		}
		for idx, log := range first.fsm.logs {
			other := env.fsm.logs[idx]
			if bytes.Compare(log, other) != 0 {
				err = fmt.Errorf("log %d mismatch %v %v", idx, log, other)
				goto ERR
			}
		}
	}
	return
ERR:
	if time.Now().After(limit) {
		t.Fatalf("%v", err)
	}
	time.Sleep(20 * time.Millisecond)
	goto CHECK
}

// Tests Raft by creating a cluster, growing it to 5 nodes while
// causing various stressful conditions
func TestRaft_Integ(t *testing.T) {
	CheckInteg(t)
	conf := DefaultConfig()
	conf.HeartbeatTimeout = 50 * time.Millisecond
	conf.ElectionTimeout = 50 * time.Millisecond
	conf.LeaderLeaseTimeout = 50 * time.Millisecond
	conf.CommitTimeout = 5 * time.Millisecond
	conf.SnapshotThreshold = 100
	conf.TrailingLogs = 10
	conf.EnableSingleNode = true

	// Create a single node
	env1 := MakeRaft(t, conf)
	NoErr(WaitFor(env1, Leader), t)

	// Do some commits
	var futures []Future
	for i := 0; i < 100; i++ {
		futures = append(futures, env1.raft.Apply([]byte(fmt.Sprintf("test%d", i)), 0))
	}
	for _, f := range futures {
		NoErr(WaitFuture(f, t), t)
		log.Printf("[DEBUG] Applied %v", f)
	}

	// Do a snapshot
	NoErr(WaitFuture(env1.raft.Snapshot(), t), t)

	// Join a few nodes!
	var envs []*RaftEnv
	for i := 0; i < 4; i++ {
		env := MakeRaft(t, conf)
		addr := env.trans.LocalAddr()
		NoErr(WaitFuture(env1.raft.AddPeer(addr), t), t)
		envs = append(envs, env)
	}

	// Wait for a leader
	leader, err := WaitForAny(Leader, append([]*RaftEnv{env1}, envs...))
	NoErr(err, t)

	// Do some more commits
	futures = nil
	for i := 0; i < 100; i++ {
		futures = append(futures, leader.raft.Apply([]byte(fmt.Sprintf("test%d", i)), 0))
	}
	for _, f := range futures {
		NoErr(WaitFuture(f, t), t)
		log.Printf("[DEBUG] Applied %v", f)
	}

	// Shoot two nodes in the head!
	rm1, rm2 := envs[0], envs[1]
	rm1.Release()
	rm2.Release()
	envs = envs[2:]
	time.Sleep(10 * time.Millisecond)

	// Wait for a leader
	leader, err = WaitForAny(Leader, append([]*RaftEnv{env1}, envs...))
	NoErr(err, t)

	// Do some more commits
	futures = nil
	for i := 0; i < 100; i++ {
		futures = append(futures, leader.raft.Apply([]byte(fmt.Sprintf("test%d", i)), 0))
	}
	for _, f := range futures {
		NoErr(WaitFuture(f, t), t)
		log.Printf("[DEBUG] Applied %v", f)
	}

	// Join a few new nodes!
	for i := 0; i < 2; i++ {
		env := MakeRaft(t, conf)
		addr := env.trans.LocalAddr()
		NoErr(WaitFuture(leader.raft.AddPeer(addr), t), t)
		envs = append(envs, env)
	}

	// Remove the old nodes
	NoErr(WaitFuture(leader.raft.RemovePeer(rm1.raft.localAddr), t), t)
	NoErr(WaitFuture(leader.raft.RemovePeer(rm2.raft.localAddr), t), t)

	// Shoot the leader
	env1.Release()
	time.Sleep(3 * conf.HeartbeatTimeout)

	// Wait for a leader
	leader, err = WaitForAny(Leader, envs)
	NoErr(err, t)

	allEnvs := append([]*RaftEnv{env1}, envs...)
	CheckConsistent(allEnvs, t)

	if len(env1.fsm.logs) != 300 {
		t.Fatalf("should apply 300 logs! %d", len(env1.fsm.logs))
	}

	for _, e := range envs {
		e.Release()
	}
}
