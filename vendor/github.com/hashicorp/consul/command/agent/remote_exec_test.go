package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/go-uuid"
)

func generateUUID() (ret string) {
	var err error
	if ret, err = uuid.GenerateUUID(); err != nil {
		panic(fmt.Sprintf("Unable to generate a UUID, %v", err))
	}
	return ret
}

func TestRexecWriter(t *testing.T) {
	writer := &rexecWriter{
		BufCh:    make(chan []byte, 16),
		BufSize:  16,
		BufIdle:  10 * time.Millisecond,
		CancelCh: make(chan struct{}),
	}

	// Write short, wait for idle
	start := time.Now()
	n, err := writer.Write([]byte("test"))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if n != 4 {
		t.Fatalf("bad: %v", n)
	}

	select {
	case b := <-writer.BufCh:
		if len(b) != 4 {
			t.Fatalf("Bad: %v", b)
		}
		if time.Now().Sub(start) < writer.BufIdle {
			t.Fatalf("too early")
		}
	case <-time.After(2 * writer.BufIdle):
		t.Fatalf("timeout")
	}

	// Write in succession to prevent the timeout
	writer.Write([]byte("test"))
	time.Sleep(writer.BufIdle / 2)
	writer.Write([]byte("test"))
	time.Sleep(writer.BufIdle / 2)
	start = time.Now()
	writer.Write([]byte("test"))

	select {
	case b := <-writer.BufCh:
		if len(b) != 12 {
			t.Fatalf("Bad: %v", b)
		}
		if time.Now().Sub(start) < writer.BufIdle {
			t.Fatalf("too early")
		}
	case <-time.After(2 * writer.BufIdle):
		t.Fatalf("timeout")
	}

	// Write large values, multiple flushes required
	writer.Write([]byte("01234567890123456789012345678901"))

	select {
	case b := <-writer.BufCh:
		if string(b) != "0123456789012345" {
			t.Fatalf("bad: %s", b)
		}
	default:
		t.Fatalf("should have buf")
	}
	select {
	case b := <-writer.BufCh:
		if string(b) != "6789012345678901" {
			t.Fatalf("bad: %s", b)
		}
	default:
		t.Fatalf("should have buf")
	}
}

func TestRemoteExecGetSpec(t *testing.T) {
	config := nextConfig()
	testRemoteExecGetSpec(t, config)
}

func TestRemoteExecGetSpec_ACLToken(t *testing.T) {
	config := nextConfig()
	config.ACLDatacenter = "dc1"
	config.ACLToken = "root"
	config.ACLDefaultPolicy = "deny"
	testRemoteExecGetSpec(t, config)
}

func testRemoteExecGetSpec(t *testing.T, c *Config) {
	dir, agent := makeAgent(t, c)
	defer os.RemoveAll(dir)
	defer agent.Shutdown()
	testutil.WaitForLeader(t, agent.RPC, "dc1")

	event := &remoteExecEvent{
		Prefix:  "_rexec",
		Session: makeRexecSession(t, agent),
	}
	defer destroySession(t, agent, event.Session)

	spec := &remoteExecSpec{
		Command: "uptime",
		Script:  []byte("#!/bin/bash"),
		Wait:    time.Second,
	}
	buf, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	key := "_rexec/" + event.Session + "/job"
	setKV(t, agent, key, buf)

	var out remoteExecSpec
	if !agent.remoteExecGetSpec(event, &out) {
		t.Fatalf("bad")
	}
	if !reflect.DeepEqual(spec, &out) {
		t.Fatalf("bad spec")
	}
}

func TestRemoteExecWrites(t *testing.T) {
	config := nextConfig()
	testRemoteExecWrites(t, config)
}

func TestRemoteExecWrites_ACLToken(t *testing.T) {
	config := nextConfig()
	config.ACLDatacenter = "dc1"
	config.ACLToken = "root"
	config.ACLDefaultPolicy = "deny"
	testRemoteExecWrites(t, config)
}

func testRemoteExecWrites(t *testing.T, c *Config) {
	dir, agent := makeAgent(t, c)
	defer os.RemoveAll(dir)
	defer agent.Shutdown()
	testutil.WaitForLeader(t, agent.RPC, "dc1")

	event := &remoteExecEvent{
		Prefix:  "_rexec",
		Session: makeRexecSession(t, agent),
	}
	defer destroySession(t, agent, event.Session)

	if !agent.remoteExecWriteAck(event) {
		t.Fatalf("bad")
	}

	output := []byte("testing")
	if !agent.remoteExecWriteOutput(event, 0, output) {
		t.Fatalf("bad")
	}
	if !agent.remoteExecWriteOutput(event, 10, output) {
		t.Fatalf("bad")
	}

	exitCode := 1
	if !agent.remoteExecWriteExitCode(event, &exitCode) {
		t.Fatalf("bad")
	}

	key := "_rexec/" + event.Session + "/" + agent.config.NodeName + "/ack"
	d := getKV(t, agent, key)
	if d == nil || d.Session != event.Session {
		t.Fatalf("bad ack: %#v", d)
	}

	key = "_rexec/" + event.Session + "/" + agent.config.NodeName + "/out/00000"
	d = getKV(t, agent, key)
	if d == nil || d.Session != event.Session || !bytes.Equal(d.Value, output) {
		t.Fatalf("bad output: %#v", d)
	}

	key = "_rexec/" + event.Session + "/" + agent.config.NodeName + "/out/0000a"
	d = getKV(t, agent, key)
	if d == nil || d.Session != event.Session || !bytes.Equal(d.Value, output) {
		t.Fatalf("bad output: %#v", d)
	}

	key = "_rexec/" + event.Session + "/" + agent.config.NodeName + "/exit"
	d = getKV(t, agent, key)
	if d == nil || d.Session != event.Session || string(d.Value) != "1" {
		t.Fatalf("bad output: %#v", d)
	}
}

func testHandleRemoteExec(t *testing.T, command string, expectedSubstring string, expectedReturnCode string) {
	dir, agent := makeAgent(t, nextConfig())
	defer os.RemoveAll(dir)
	defer agent.Shutdown()
	testutil.WaitForLeader(t, agent.RPC, "dc1")

	event := &remoteExecEvent{
		Prefix:  "_rexec",
		Session: makeRexecSession(t, agent),
	}
	defer destroySession(t, agent, event.Session)

	spec := &remoteExecSpec{
		Command: command,
		Wait:    time.Second,
	}
	buf, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	key := "_rexec/" + event.Session + "/job"
	setKV(t, agent, key, buf)

	buf, err = json.Marshal(event)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	msg := &UserEvent{
		ID:      generateUUID(),
		Payload: buf,
	}

	// Handle the event...
	agent.handleRemoteExec(msg)

	// Verify we have an ack
	key = "_rexec/" + event.Session + "/" + agent.config.NodeName + "/ack"
	d := getKV(t, agent, key)
	if d == nil || d.Session != event.Session {
		t.Fatalf("bad ack: %#v", d)
	}

	// Verify we have output
	key = "_rexec/" + event.Session + "/" + agent.config.NodeName + "/out/00000"
	d = getKV(t, agent, key)
	if d == nil || d.Session != event.Session ||
		!bytes.Contains(d.Value, []byte(expectedSubstring)) {
		t.Fatalf("bad output: %#v", d)
	}

	// Verify we have an exit code
	key = "_rexec/" + event.Session + "/" + agent.config.NodeName + "/exit"
	d = getKV(t, agent, key)
	if d == nil || d.Session != event.Session || string(d.Value) != expectedReturnCode {
		t.Fatalf("bad output: %#v", d)
	}
}

func TestHandleRemoteExec(t *testing.T) {
	testHandleRemoteExec(t, "uptime", "load", "0")
}

func TestHandleRemoteExecFailed(t *testing.T) {
	testHandleRemoteExec(t, "echo failing;exit 2", "failing", "2")
}

func makeRexecSession(t *testing.T, agent *Agent) string {
	args := structs.SessionRequest{
		Datacenter: agent.config.Datacenter,
		Op:         structs.SessionCreate,
		Session: structs.Session{
			Node:      agent.config.NodeName,
			LockDelay: 15 * time.Second,
		},
	}
	var out string
	if err := agent.RPC("Session.Apply", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	return out
}

func destroySession(t *testing.T, agent *Agent, session string) {
	args := structs.SessionRequest{
		Datacenter: agent.config.Datacenter,
		Op:         structs.SessionDestroy,
		Session: structs.Session{
			ID: session,
		},
	}
	var out string
	if err := agent.RPC("Session.Apply", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func setKV(t *testing.T, agent *Agent, key string, val []byte) {
	write := structs.KVSRequest{
		Datacenter: agent.config.Datacenter,
		Op:         structs.KVSSet,
		DirEnt: structs.DirEntry{
			Key:   key,
			Value: val,
		},
	}
	write.Token = agent.config.ACLToken
	var success bool
	if err := agent.RPC("KVS.Apply", &write, &success); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func getKV(t *testing.T, agent *Agent, key string) *structs.DirEntry {
	req := structs.KeyRequest{
		Datacenter: agent.config.Datacenter,
		Key:        key,
	}
	req.Token = agent.config.ACLToken
	var out structs.IndexedDirEntries
	if err := agent.RPC("KVS.Get", &req, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out.Entries) > 0 {
		return out.Entries[0]
	}
	return nil
}
