package networkdb

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/docker/go-events"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	dbPort             int32 = 10000
	runningInContainer       = flag.Bool("incontainer", false, "Indicates if the test is running in a container")
)

func TestMain(m *testing.M) {
	ioutil.WriteFile("/proc/sys/net/ipv6/conf/lo/disable_ipv6", []byte{'0', '\n'}, 0644)
	logrus.SetLevel(logrus.ErrorLevel)
	os.Exit(m.Run())
}

func createNetworkDBInstances(t *testing.T, num int, namePrefix string) []*NetworkDB {
	var dbs []*NetworkDB
	for i := 0; i < num; i++ {
		conf := DefaultConfig()
		conf.NodeName = fmt.Sprintf("%s%d", namePrefix, i+1)
		conf.BindPort = int(atomic.AddInt32(&dbPort, 1))
		db, err := New(conf)
		require.NoError(t, err)

		if i != 0 {
			err = db.Join([]string{fmt.Sprintf("localhost:%d", db.config.BindPort-1)})
			assert.NoError(t, err)
		}

		dbs = append(dbs, db)
	}

	return dbs
}

func closeNetworkDBInstances(dbs []*NetworkDB) {
	for _, db := range dbs {
		db.Close()
	}
}

func (db *NetworkDB) verifyNodeExistence(t *testing.T, node string, present bool) {
	for i := 0; i < 80; i++ {
		db.RLock()
		_, ok := db.nodes[node]
		db.RUnlock()
		if present && ok {
			return
		}

		if !present && !ok {
			return
		}

		time.Sleep(50 * time.Millisecond)
	}

	assert.Fail(t, fmt.Sprintf("%s: Node existence verification for node %s failed", db.config.NodeName, node))
}

func (db *NetworkDB) verifyNetworkExistence(t *testing.T, node string, id string, present bool) {
	for i := 0; i < 80; i++ {
		db.RLock()
		nn, nnok := db.networks[node]
		db.RUnlock()
		if nnok {
			n, ok := nn[id]
			if present && ok {
				return
			}

			if !present &&
				((ok && n.leaving) ||
					!ok) {
				return
			}
		}

		time.Sleep(50 * time.Millisecond)
	}

	assert.Fail(t, "Network existence verification failed")
}

func (db *NetworkDB) verifyEntryExistence(t *testing.T, tname, nid, key, value string, present bool) {
	n := 80
	for i := 0; i < n; i++ {
		entry, err := db.getEntry(tname, nid, key)
		if present && err == nil && string(entry.value) == value {
			return
		}

		if !present &&
			((err == nil && entry.deleting) ||
				(err != nil)) {
			return
		}

		if i == n-1 && !present && err != nil {
			return
		}

		time.Sleep(50 * time.Millisecond)
	}

	assert.Fail(t, fmt.Sprintf("Entry existence verification test failed for %s", db.config.NodeName))
}

func testWatch(t *testing.T, ch chan events.Event, ev interface{}, tname, nid, key, value string) {
	select {
	case rcvdEv := <-ch:
		assert.Equal(t, fmt.Sprintf("%T", rcvdEv), fmt.Sprintf("%T", ev))
		switch rcvdEv.(type) {
		case CreateEvent:
			assert.Equal(t, tname, rcvdEv.(CreateEvent).Table)
			assert.Equal(t, nid, rcvdEv.(CreateEvent).NetworkID)
			assert.Equal(t, key, rcvdEv.(CreateEvent).Key)
			assert.Equal(t, value, string(rcvdEv.(CreateEvent).Value))
		case UpdateEvent:
			assert.Equal(t, tname, rcvdEv.(UpdateEvent).Table)
			assert.Equal(t, nid, rcvdEv.(UpdateEvent).NetworkID)
			assert.Equal(t, key, rcvdEv.(UpdateEvent).Key)
			assert.Equal(t, value, string(rcvdEv.(UpdateEvent).Value))
		case DeleteEvent:
			assert.Equal(t, tname, rcvdEv.(DeleteEvent).Table)
			assert.Equal(t, nid, rcvdEv.(DeleteEvent).NetworkID)
			assert.Equal(t, key, rcvdEv.(DeleteEvent).Key)
		}
	case <-time.After(time.Second):
		t.Fail()
		return
	}
}

func TestNetworkDBSimple(t *testing.T) {
	dbs := createNetworkDBInstances(t, 2, "node")
	closeNetworkDBInstances(dbs)
}

func TestNetworkDBJoinLeaveNetwork(t *testing.T) {
	dbs := createNetworkDBInstances(t, 2, "node")

	err := dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	dbs[1].verifyNetworkExistence(t, "node1", "network1", true)

	err = dbs[0].LeaveNetwork("network1")
	assert.NoError(t, err)

	dbs[1].verifyNetworkExistence(t, "node1", "network1", false)
	closeNetworkDBInstances(dbs)
}

func TestNetworkDBJoinLeaveNetworks(t *testing.T) {
	dbs := createNetworkDBInstances(t, 2, "node")

	n := 10
	for i := 1; i <= n; i++ {
		err := dbs[0].JoinNetwork(fmt.Sprintf("network0%d", i))
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		err := dbs[1].JoinNetwork(fmt.Sprintf("network1%d", i))
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		dbs[1].verifyNetworkExistence(t, "node1", fmt.Sprintf("network0%d", i), true)
	}

	for i := 1; i <= n; i++ {
		dbs[0].verifyNetworkExistence(t, "node2", fmt.Sprintf("network1%d", i), true)
	}

	for i := 1; i <= n; i++ {
		err := dbs[0].LeaveNetwork(fmt.Sprintf("network0%d", i))
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		err := dbs[1].LeaveNetwork(fmt.Sprintf("network1%d", i))
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		dbs[1].verifyNetworkExistence(t, "node1", fmt.Sprintf("network0%d", i), false)
	}

	for i := 1; i <= n; i++ {
		dbs[0].verifyNetworkExistence(t, "node2", fmt.Sprintf("network1%d", i), false)
	}

	closeNetworkDBInstances(dbs)
}

func TestNetworkDBCRUDTableEntry(t *testing.T) {
	dbs := createNetworkDBInstances(t, 3, "node")

	err := dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	dbs[1].verifyNetworkExistence(t, "node1", "network1", true)

	err = dbs[1].JoinNetwork("network1")
	assert.NoError(t, err)

	err = dbs[0].CreateEntry("test_table", "network1", "test_key", []byte("test_value"))
	assert.NoError(t, err)

	dbs[1].verifyEntryExistence(t, "test_table", "network1", "test_key", "test_value", true)
	dbs[2].verifyEntryExistence(t, "test_table", "network1", "test_key", "test_value", false)

	err = dbs[0].UpdateEntry("test_table", "network1", "test_key", []byte("test_updated_value"))
	assert.NoError(t, err)

	dbs[1].verifyEntryExistence(t, "test_table", "network1", "test_key", "test_updated_value", true)

	err = dbs[0].DeleteEntry("test_table", "network1", "test_key")
	assert.NoError(t, err)

	dbs[1].verifyEntryExistence(t, "test_table", "network1", "test_key", "", false)

	closeNetworkDBInstances(dbs)
}

func TestNetworkDBCRUDTableEntries(t *testing.T) {
	dbs := createNetworkDBInstances(t, 2, "node")

	err := dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	dbs[1].verifyNetworkExistence(t, "node1", "network1", true)

	err = dbs[1].JoinNetwork("network1")
	assert.NoError(t, err)

	n := 10
	for i := 1; i <= n; i++ {
		err = dbs[0].CreateEntry("test_table", "network1",
			fmt.Sprintf("test_key0%d", i),
			[]byte(fmt.Sprintf("test_value0%d", i)))
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		err = dbs[1].CreateEntry("test_table", "network1",
			fmt.Sprintf("test_key1%d", i),
			[]byte(fmt.Sprintf("test_value1%d", i)))
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		dbs[0].verifyEntryExistence(t, "test_table", "network1",
			fmt.Sprintf("test_key1%d", i),
			fmt.Sprintf("test_value1%d", i), true)
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		dbs[1].verifyEntryExistence(t, "test_table", "network1",
			fmt.Sprintf("test_key0%d", i),
			fmt.Sprintf("test_value0%d", i), true)
		assert.NoError(t, err)
	}

	// Verify deletes
	for i := 1; i <= n; i++ {
		err = dbs[0].DeleteEntry("test_table", "network1",
			fmt.Sprintf("test_key0%d", i))
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		err = dbs[1].DeleteEntry("test_table", "network1",
			fmt.Sprintf("test_key1%d", i))
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		dbs[0].verifyEntryExistence(t, "test_table", "network1",
			fmt.Sprintf("test_key1%d", i), "", false)
		assert.NoError(t, err)
	}

	for i := 1; i <= n; i++ {
		dbs[1].verifyEntryExistence(t, "test_table", "network1",
			fmt.Sprintf("test_key0%d", i), "", false)
		assert.NoError(t, err)
	}

	closeNetworkDBInstances(dbs)
}

func TestNetworkDBNodeLeave(t *testing.T) {
	dbs := createNetworkDBInstances(t, 2, "node")

	err := dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	err = dbs[1].JoinNetwork("network1")
	assert.NoError(t, err)

	err = dbs[0].CreateEntry("test_table", "network1", "test_key", []byte("test_value"))
	assert.NoError(t, err)

	dbs[1].verifyEntryExistence(t, "test_table", "network1", "test_key", "test_value", true)

	dbs[0].Close()
	dbs[1].verifyEntryExistence(t, "test_table", "network1", "test_key", "test_value", false)
	dbs[1].Close()
}

func TestNetworkDBWatch(t *testing.T) {
	dbs := createNetworkDBInstances(t, 2, "node")
	err := dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	err = dbs[1].JoinNetwork("network1")
	assert.NoError(t, err)

	ch, cancel := dbs[1].Watch("", "", "")

	err = dbs[0].CreateEntry("test_table", "network1", "test_key", []byte("test_value"))
	assert.NoError(t, err)

	testWatch(t, ch.C, CreateEvent{}, "test_table", "network1", "test_key", "test_value")

	err = dbs[0].UpdateEntry("test_table", "network1", "test_key", []byte("test_updated_value"))
	assert.NoError(t, err)

	testWatch(t, ch.C, UpdateEvent{}, "test_table", "network1", "test_key", "test_updated_value")

	err = dbs[0].DeleteEntry("test_table", "network1", "test_key")
	assert.NoError(t, err)

	testWatch(t, ch.C, DeleteEvent{}, "test_table", "network1", "test_key", "")

	cancel()
	closeNetworkDBInstances(dbs)
}

func TestNetworkDBBulkSync(t *testing.T) {
	dbs := createNetworkDBInstances(t, 2, "node")

	err := dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	dbs[1].verifyNetworkExistence(t, "node1", "network1", true)

	n := 1000
	for i := 1; i <= n; i++ {
		err = dbs[0].CreateEntry("test_table", "network1",
			fmt.Sprintf("test_key0%d", i),
			[]byte(fmt.Sprintf("test_value0%d", i)))
		assert.NoError(t, err)
	}

	err = dbs[1].JoinNetwork("network1")
	assert.NoError(t, err)

	dbs[0].verifyNetworkExistence(t, "node2", "network1", true)

	for i := 1; i <= n; i++ {
		dbs[1].verifyEntryExistence(t, "test_table", "network1",
			fmt.Sprintf("test_key0%d", i),
			fmt.Sprintf("test_value0%d", i), true)
		assert.NoError(t, err)
	}

	closeNetworkDBInstances(dbs)
}

func TestNetworkDBCRUDMediumCluster(t *testing.T) {
	n := 5

	dbs := createNetworkDBInstances(t, n, "node")

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}

			dbs[i].verifyNodeExistence(t, fmt.Sprintf("node%d", j+1), true)
		}
	}

	for i := 0; i < n; i++ {
		err := dbs[i].JoinNetwork("network1")
		assert.NoError(t, err)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			dbs[i].verifyNetworkExistence(t, fmt.Sprintf("node%d", j+1), "network1", true)
		}
	}

	err := dbs[0].CreateEntry("test_table", "network1", "test_key", []byte("test_value"))
	assert.NoError(t, err)

	for i := 1; i < n; i++ {
		dbs[i].verifyEntryExistence(t, "test_table", "network1", "test_key", "test_value", true)
	}

	err = dbs[0].UpdateEntry("test_table", "network1", "test_key", []byte("test_updated_value"))
	assert.NoError(t, err)

	for i := 1; i < n; i++ {
		dbs[i].verifyEntryExistence(t, "test_table", "network1", "test_key", "test_updated_value", true)
	}

	err = dbs[0].DeleteEntry("test_table", "network1", "test_key")
	assert.NoError(t, err)

	for i := 1; i < n; i++ {
		dbs[i].verifyEntryExistence(t, "test_table", "network1", "test_key", "", false)
	}

	log.Print("Closing DB instances...")
	closeNetworkDBInstances(dbs)
}

func TestNetworkDBNodeJoinLeaveIteration(t *testing.T) {
	maxRetry := 5
	dbs := createNetworkDBInstances(t, 2, "node")

	// Single node Join/Leave
	err := dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	if len(dbs[0].networkNodes["network1"]) != 1 {
		t.Fatalf("The networkNodes list has to have be 1 instead of %d", len(dbs[0].networkNodes["network1"]))
	}

	err = dbs[0].LeaveNetwork("network1")
	assert.NoError(t, err)

	if len(dbs[0].networkNodes["network1"]) != 0 {
		t.Fatalf("The networkNodes list has to have be 0 instead of %d", len(dbs[0].networkNodes["network1"]))
	}

	// Multiple nodes Join/Leave
	err = dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	err = dbs[1].JoinNetwork("network1")
	assert.NoError(t, err)

	// Wait for the propagation on db[0]
	for i := 0; i < maxRetry; i++ {
		if len(dbs[0].networkNodes["network1"]) == 2 {
			break
		}
		time.Sleep(1 * time.Second)
	}
	if len(dbs[0].networkNodes["network1"]) != 2 {
		t.Fatalf("The networkNodes list has to have be 2 instead of %d - %v", len(dbs[0].networkNodes["network1"]), dbs[0].networkNodes["network1"])
	}

	// Wait for the propagation on db[1]
	for i := 0; i < maxRetry; i++ {
		if len(dbs[1].networkNodes["network1"]) == 2 {
			break
		}
		time.Sleep(1 * time.Second)
	}
	if len(dbs[1].networkNodes["network1"]) != 2 {
		t.Fatalf("The networkNodes list has to have be 2 instead of %d - %v", len(dbs[1].networkNodes["network1"]), dbs[1].networkNodes["network1"])
	}

	// Try a quick leave/join
	err = dbs[0].LeaveNetwork("network1")
	assert.NoError(t, err)
	err = dbs[0].JoinNetwork("network1")
	assert.NoError(t, err)

	for i := 0; i < maxRetry; i++ {
		if len(dbs[0].networkNodes["network1"]) == 2 {
			break
		}
		time.Sleep(1 * time.Second)
	}
	if len(dbs[0].networkNodes["network1"]) != 2 {
		t.Fatalf("The networkNodes list has to have be 2 instead of %d - %v", len(dbs[0].networkNodes["network1"]), dbs[0].networkNodes["network1"])
	}

	for i := 0; i < maxRetry; i++ {
		if len(dbs[1].networkNodes["network1"]) == 2 {
			break
		}
		time.Sleep(1 * time.Second)
	}
	if len(dbs[1].networkNodes["network1"]) != 2 {
		t.Fatalf("The networkNodes list has to have be 2 instead of %d - %v", len(dbs[1].networkNodes["network1"]), dbs[1].networkNodes["network1"])
	}

	dbs[0].Close()
	dbs[1].Close()
}
