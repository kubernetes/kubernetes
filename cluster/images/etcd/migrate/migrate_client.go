/*
Copyright 2018 The Kubernetes Authors.

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

package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"context"

	clientv2 "github.com/coreos/etcd/client"
	"github.com/coreos/etcd/clientv3"
	"k8s.io/klog"
)

// CombinedEtcdClient provides an implementation of EtcdMigrateClient using a combination of the etcd v2 client, v3 client
// and etcdctl commands called via the shell.
type CombinedEtcdClient struct {
	cfg *EtcdMigrateCfg
}

// NewEtcdMigrateClient creates a new EtcdMigrateClient from the given EtcdMigrateCfg.
func NewEtcdMigrateClient(cfg *EtcdMigrateCfg) (EtcdMigrateClient, error) {
	return &CombinedEtcdClient{cfg}, nil
}

// Close closes the client and releases any resources it holds.
func (e *CombinedEtcdClient) Close() error {
	return nil
}

// SetEtcdVersionKeyValue writes the given version to the etcd 'etcd_version' key.
// If no error is returned, the write was successful, indicating the etcd server is available
// and able to perform consensus writes.
func (e *CombinedEtcdClient) SetEtcdVersionKeyValue(version *EtcdVersion) error {
	return e.Put(version, "etcd_version", version.String())
}

// Put write a single key value pair to etcd.
func (e *CombinedEtcdClient) Put(version *EtcdVersion, key, value string) error {
	if version.Major == 2 {
		v2client, err := e.clientV2()
		if err != nil {
			return err
		}
		_, err = v2client.Set(context.Background(), key, value, nil)
		return err
	}
	v3client, err := e.clientV3()
	if err != nil {
		return err
	}
	defer v3client.Close()
	_, err = v3client.KV.Put(context.Background(), key, value)
	return err
}

// Get reads a single value for a given key.
func (e *CombinedEtcdClient) Get(version *EtcdVersion, key string) (string, error) {
	if version.Major == 2 {
		v2client, err := e.clientV2()
		if err != nil {
			return "", err
		}
		resp, err := v2client.Get(context.Background(), key, nil)
		if err != nil {
			return "", err
		}
		return resp.Node.Value, nil
	}
	v3client, err := e.clientV3()
	if err != nil {
		return "", err
	}
	defer v3client.Close()
	resp, err := v3client.KV.Get(context.Background(), key)
	if err != nil {
		return "", err
	}
	kvs := resp.Kvs
	if len(kvs) != 1 {
		return "", fmt.Errorf("expected exactly one value for key %s but got %d", key, len(kvs))
	}

	return string(kvs[0].Value), nil
}

func (e *CombinedEtcdClient) clientV2() (clientv2.KeysAPI, error) {
	v2client, err := clientv2.New(clientv2.Config{Endpoints: []string{e.endpoint()}})
	if err != nil {
		return nil, err
	}
	return clientv2.NewKeysAPI(v2client), nil
}

func (e *CombinedEtcdClient) clientV3() (*clientv3.Client, error) {
	return clientv3.New(clientv3.Config{Endpoints: []string{e.endpoint()}})
}

// Backup creates a backup of an etcd2 data directory at the given backupDir.
func (e *CombinedEtcdClient) Backup(version *EtcdVersion, backupDir string) error {
	// We cannot use etcd/client (v2) to make this call. It is implemented in the etcdctl client code.
	if version.Major != 2 {
		return fmt.Errorf("etcd 2.x required but got version '%s'", version)
	}
	return e.runEtcdctlCommand(version,
		"--debug",
		"backup",
		"--data-dir", e.cfg.dataDirectory,
		"--backup-dir", backupDir,
	)
}

// Snapshot captures a snapshot from a running etcd3 server and saves it to the given snapshotFile.
// We cannot use etcd/clientv3 to make this call. It is implemented in the etcdctl client code.
func (e *CombinedEtcdClient) Snapshot(version *EtcdVersion, snapshotFile string) error {
	if version.Major != 3 {
		return fmt.Errorf("etcd 3.x required but got version '%s'", version)
	}
	return e.runEtcdctlCommand(version,
		"--endpoints", e.endpoint(),
		"snapshot", "save", snapshotFile,
	)
}

// Restore restores a given snapshotFile into the data directory specified this clients config.
func (e *CombinedEtcdClient) Restore(version *EtcdVersion, snapshotFile string) error {
	// We cannot use etcd/clientv3 to make this call. It is implemented in the etcdctl client code.
	if version.Major != 3 {
		return fmt.Errorf("etcd 3.x required but got version '%s'", version)
	}
	return e.runEtcdctlCommand(version,
		"snapshot", "restore", snapshotFile,
		"--data-dir", e.cfg.dataDirectory,
		"--name", e.cfg.name,
		"--initial-advertise-peer-urls", e.cfg.peerAdvertiseUrls,
		"--initial-cluster", e.cfg.initialCluster,
	)
}

// Migrate upgrades a 'etcd2' storage version data directory to a 'etcd3' storage version
// data directory.
func (e *CombinedEtcdClient) Migrate(version *EtcdVersion) error {
	// We cannot use etcd/clientv3 to make this call as it is implemented in etcd/etcdctl.
	if version.Major != 3 {
		return fmt.Errorf("etcd 3.x required but got version '%s'", version)
	}
	return e.runEtcdctlCommand(version,
		"migrate",
		"--data-dir", e.cfg.dataDirectory,
	)
}

func (e *CombinedEtcdClient) runEtcdctlCommand(version *EtcdVersion, args ...string) error {
	etcdctlCmd := exec.Command(filepath.Join(e.cfg.binPath, fmt.Sprintf("etcdctl-%s", version)), args...)
	etcdctlCmd.Env = []string{fmt.Sprintf("ETCDCTL_API=%d", version.Major)}
	etcdctlCmd.Stdout = os.Stdout
	etcdctlCmd.Stderr = os.Stderr
	return etcdctlCmd.Run()
}

// AttachLease attaches leases of the given leaseDuration to all the  etcd objects under
// ttlKeysDirectory specified in this client's config.
func (e *CombinedEtcdClient) AttachLease(leaseDuration time.Duration) error {
	ttlKeysPrefix := e.cfg.ttlKeysDirectory
	// Make sure that ttlKeysPrefix is ended with "/" so that we only get children "directories".
	if !strings.HasSuffix(ttlKeysPrefix, "/") {
		ttlKeysPrefix += "/"
	}
	ctx := context.Background()

	v3client, err := e.clientV3()
	if err != nil {
		return err
	}
	defer v3client.Close()
	objectsResp, err := v3client.KV.Get(ctx, ttlKeysPrefix, clientv3.WithPrefix())
	if err != nil {
		return fmt.Errorf("Error while getting objects to attach to the lease")
	}

	lease, err := v3client.Lease.Grant(ctx, int64(leaseDuration/time.Second))
	if err != nil {
		return fmt.Errorf("Error while creating lease: %v", err)
	}
	klog.Infof("Lease with TTL: %v created", lease.TTL)

	klog.Infof("Attaching lease to %d entries", len(objectsResp.Kvs))
	for _, kv := range objectsResp.Kvs {
		putResp, err := v3client.KV.Put(ctx, string(kv.Key), string(kv.Value), clientv3.WithLease(lease.ID), clientv3.WithPrevKV())
		if err != nil {
			klog.Errorf("Error while attaching lease to: %s", string(kv.Key))
		}
		if bytes.Compare(putResp.PrevKv.Value, kv.Value) != 0 {
			return fmt.Errorf("concurrent access to key detected when setting lease on %s, expected previous value of %s but got %s",
				kv.Key, kv.Value, putResp.PrevKv.Value)
		}
	}
	return nil
}

func (e *CombinedEtcdClient) endpoint() string {
	return fmt.Sprintf("http://127.0.0.1:%d", e.cfg.port)
}
