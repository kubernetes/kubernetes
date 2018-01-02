package cluster

import (
	"bytes"
	"encoding/json"
	"strings"

	"go.pedge.io/dlog"

	"github.com/libopenstorage/openstorage/api"
	"github.com/portworx/kvdb"
)

const (
	// ClusterDBKey is the key at which cluster info is store in kvdb
	ClusterDBKey = "cluster/database"
)

func snapAndReadClusterInfo() (*ClusterInitState, error) {
	kv := kvdb.Instance()

	// To work-around a kvdb issue with watches, try snapshot in a loop
	var (
		collector kvdb.UpdatesCollector
		err       error
		version   uint64
		snap      kvdb.Kvdb
	)
	for i := 0; i < 3; i++ {
		if i > 0 {
			dlog.Infof("Retrying snapshot")
		}
		// Start the watch before the snapshot
		collector, err = kvdb.NewUpdatesCollector(kv, "", 0)
		if err != nil {
			dlog.Errorf("Failed to start collector for cluster db: %v", err)
			collector = nil
			continue
		}
		// Create the snapshot
		snap, version, err = kv.Snapshot("")
		if err != nil {
			dlog.Errorf("Snapshot failed for cluster db: %v", err)
			collector.Stop()
			collector = nil
		} else {
			break
		}
	}
	if err != nil {
		return nil, err
	}
	dlog.Infof("Cluster db snapshot at: %v", version)

	clusterDB, err := snap.Get(ClusterDBKey)
	if err != nil && !strings.Contains(err.Error(), "Key not found") {
		dlog.Warnln("Warning, could not read cluster database")
		return nil, err
	}

	db := ClusterInfo{
		Status:      api.Status_STATUS_INIT,
		NodeEntries: make(map[string]NodeEntry),
	}
	state := &ClusterInitState{
		ClusterInfo: &db,
		InitDb:      snap,
		Version:     version,
		Collector:   collector,
	}

	if clusterDB == nil || bytes.Compare(clusterDB.Value, []byte("{}")) == 0 {
		dlog.Infoln("Cluster is uninitialized...")
		return state, nil
	}
	if err := json.Unmarshal(clusterDB.Value, &db); err != nil {
		dlog.Warnln("Fatal, Could not parse cluster database ", kv)
		return state, err
	}

	return state, nil
}

func readClusterInfo() (ClusterInfo, uint64, error) {
	kvdb := kvdb.Instance()

	db := ClusterInfo{
		Status:      api.Status_STATUS_INIT,
		NodeEntries: make(map[string]NodeEntry),
	}
	kv, err := kvdb.Get(ClusterDBKey)

	if err != nil && !strings.Contains(err.Error(), "Key not found") {
		dlog.Warnln("Warning, could not read cluster database")
		return db, 0, err
	}

	if kv == nil || bytes.Compare(kv.Value, []byte("{}")) == 0 {
		dlog.Infoln("Cluster is uninitialized...")
		return db, 0, nil
	}
	if err := json.Unmarshal(kv.Value, &db); err != nil {
		dlog.Warnln("Fatal, Could not parse cluster database ", kv)
		return db, 0, err
	}

	return db, kv.KVDBIndex, nil
}

func writeClusterInfo(db *ClusterInfo) (*kvdb.KVPair, error) {
	kvdb := kvdb.Instance()
	b, err := json.Marshal(db)
	if err != nil {
		dlog.Warnf("Fatal, Could not marshal cluster database to JSON: %v", err)
		return nil, err
	}

	kvp, err := kvdb.Put(ClusterDBKey, b, 0)
	if err != nil {
		dlog.Warnf("Fatal, Could not marshal cluster database to JSON: %v", err)
		return nil, err
	}
	return kvp, nil
}
