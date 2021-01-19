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
	"fmt"
	"os"
	"time"

	"github.com/blang/semver"
	"k8s.io/klog/v2"
)

// EtcdMigrateCfg provides all configuration required to perform etcd data upgrade/downgrade migrations.
type EtcdMigrateCfg struct {
	binPath           string
	name              string
	initialCluster    string
	port              uint64
	peerListenUrls    string
	peerAdvertiseUrls string
	clientListenUrls  string
	etcdDataPrefix    string
	ttlKeysDirectory  string
	supportedVersions SupportedVersions
	dataDirectory     string
	etcdServerArgs    string
}

// EtcdMigrateClient defines the etcd client operations required to perform migrations.
type EtcdMigrateClient interface {
	SetEtcdVersionKeyValue(version *EtcdVersion) error
	Get(version *EtcdVersion, key string) (string, error)
	Put(version *EtcdVersion, key, value string) error
	Backup(version *EtcdVersion, backupDir string) error
	Snapshot(version *EtcdVersion, snapshotFile string) error
	Restore(version *EtcdVersion, snapshotFile string) error
	Migrate(version *EtcdVersion) error
	AttachLease(leaseDuration time.Duration) error
	Close() error
}

// Migrator manages etcd data migrations.
type Migrator struct {
	cfg           *EtcdMigrateCfg // TODO: don't wire this directly in
	dataDirectory *DataDirectory
	client        EtcdMigrateClient
}

// MigrateIfNeeded upgrades or downgrades the etcd data directory to the given target version.
func (m *Migrator) MigrateIfNeeded(target *EtcdVersionPair) error {
	klog.InfoS("Starting migration to target version", "targetVersion", target)
	err := m.dataDirectory.Initialize(target)
	if err != nil {
		return fmt.Errorf("failed to initialize data directory %s: %v", m.dataDirectory.path, err)
	}

	var current *EtcdVersionPair
	vfExists, err := m.dataDirectory.versionFile.Exists()
	if err != nil {
		return err
	}
	if vfExists {
		current, err = m.dataDirectory.versionFile.Read()
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("existing data directory '%s' is missing version.txt file, unable to migrate", m.dataDirectory.path)
	}

	for {
		klog.InfoS("Converging current version to target version", "currentVersion", current, "targetVersion", target)
		currentNextMinorVersion := &EtcdVersion{Version: semver.Version{Major: current.version.Major, Minor: current.version.Minor + 1}}
		switch {
		case current.version.MajorMinorEquals(target.version) || currentNextMinorVersion.MajorMinorEquals(target.version):
			klog.InfoS("Current version equals or is one minor version previous of target version - migration complete", "currentVersion", current, "targetVersion", target)
			err = m.dataDirectory.versionFile.Write(target)
			if err != nil {
				return fmt.Errorf("failed to write version.txt to '%s': %v", m.dataDirectory.path, err)
			}
			return nil
		case current.storageVersion == storageEtcd2 && target.storageVersion == storageEtcd3:
			return fmt.Errorf("upgrading from etcd2 storage to etcd3 storage is not supported")
		case current.version.Major == 3 && target.version.Major == 2:
			return fmt.Errorf("downgrading from etcd 3.x to 2.x is not supported")
		case current.version.Major == target.version.Major && current.version.Minor < target.version.Minor:
			stepVersion := m.cfg.supportedVersions.NextVersionPair(current)
			klog.InfoS("Upgrading etcd from current version to next version", "currentVersion", current, "nextVersion", stepVersion)
			current, err = m.minorVersionUpgrade(current, stepVersion)
		case current.version.Major == 3 && target.version.Major == 3 && current.version.Minor > target.version.Minor:
			klog.InfoS("Rolling etcd back from current version to target version", "currentVersion", current, "targetVersion", target)
			current, err = m.rollbackEtcd3MinorVersion(current, target)
		}
		if err != nil {
			return err
		}
	}
}

func (m *Migrator) rollbackEtcd3MinorVersion(current *EtcdVersionPair, target *EtcdVersionPair) (*EtcdVersionPair, error) {
	if target.version.Minor != current.version.Minor-1 {
		return nil, fmt.Errorf("rollback from %s to %s not supported, only rollbacks to the previous minor version are supported", current.version, target.version)
	}

	klog.InfoS("Performing etcd current version -> target version, rollback", "currentVersion", current.version, "targetVersion", target.version)
	err := m.dataDirectory.Backup()
	if err != nil {
		return nil, err
	}

	snapshotFilename := fmt.Sprintf("%s.snapshot.db", m.dataDirectory.path)
	err = os.Remove(snapshotFilename)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("failed to clean snapshot file before rollback: %v", err)
	}

	// Start current version of etcd.
	runner := m.newServer()
	klog.InfoS("Starting etcd version to capture rollback snapshot", "currentVersion", current.version)
	err = runner.Start(current.version)
	if err != nil {
		klog.Fatalf("Unable to automatically downgrade etcd: starting etcd version %s to capture rollback snapshot failed: %v", current.version, err)
		return nil, err
	}

	klog.InfoS("Snapshotting etcd current version to snapshot file", "currentVersion", current.version, "snapshotFile", snapshotFilename)
	err = m.client.Snapshot(current.version, snapshotFilename)
	if err != nil {
		return nil, err
	}

	err = runner.Stop()
	if err != nil {
		return nil, err
	}

	klog.Info("Backing up data before rolling back")
	backupDir := fmt.Sprintf("%s.bak", m.dataDirectory)
	err = os.RemoveAll(backupDir)
	if err != nil {
		return nil, err
	}
	origInfo, err := os.Stat(m.dataDirectory.path)
	if err != nil {
		return nil, err
	}
	err = os.Rename(m.dataDirectory.path, backupDir)
	if err != nil {
		return nil, err
	}

	klog.InfoS("Restoring etcd with target version from snapshot file", "targetVersion", target.version, "snapshotFile", snapshotFilename)
	err = m.client.Restore(target.version, snapshotFilename)
	if err != nil {
		return nil, err
	}
	err = os.Chmod(m.dataDirectory.path, origInfo.Mode())
	if err != nil {
		return nil, err
	}

	return target, nil
}

func (m *Migrator) minorVersionUpgrade(current *EtcdVersionPair, target *EtcdVersionPair) (*EtcdVersionPair, error) {
	runner := m.newServer()

	// Do the migration step, by just starting etcd in the target version.
	err := runner.Start(target.version)
	if err != nil {
		return nil, err
	}
	err = runner.Stop()
	return target, err
}

func (m *Migrator) newServer() *EtcdMigrateServer {
	return NewEtcdMigrateServer(m.cfg, m.client)
}
