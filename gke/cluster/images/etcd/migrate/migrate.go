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
	"path/filepath"

	"github.com/spf13/cobra"
	"k8s.io/klog/v2"
)

const (
	versionFilename = "version.txt"
)

var (
	migrateCmd = &cobra.Command{
		Short: "Upgrade/downgrade etcd data across multiple versions",
		Long: `Upgrade or downgrade etcd data across multiple versions to the target version

Given a 'bin-dir' directory of etcd and etcdctl binaries, an etcd 'data-dir' with a 'version.txt' file and
a target etcd version, this tool will upgrade or downgrade the etcd data from the version specified in
'version.txt' to the target version.
`,
		Run: func(cmd *cobra.Command, args []string) {
			runMigrate()
		},
	}
	opts = migrateOpts{}
)

func main() {
	registerFlags(migrateCmd.Flags(), &opts)
	err := migrateCmd.Execute()
	if err != nil {
		klog.Errorf("Failed to execute migratecmd: %s", err)
	}
}

// runMigrate starts the migration.
func runMigrate() {
	if err := opts.validateAndDefault(); err != nil {
		klog.Fatalf("%v", err)
	}
	copyBinaries()

	target := &EtcdVersionPair{
		version:        MustParseEtcdVersion(opts.targetVersion),
		storageVersion: MustParseEtcdStorageVersion(opts.targetStorage),
	}

	migrate(
		opts.name, opts.port, opts.peerListenUrls, opts.peerAdvertiseUrls, opts.clientListenUrls,
		opts.binDir, opts.dataDir, opts.etcdDataPrefix, opts.ttlKeysDirectory, opts.initialCluster,
		target, opts.supportedVersions, opts.etcdServerArgs)
}

func copyBinaries() {
	if val, err := lookupEnv("DO_NOT_MOVE_BINARIES"); err != nil || val != "true" {
		etcdVersioned := fmt.Sprintf("etcd-%s", opts.targetVersion)
		etcdctlVersioned := fmt.Sprintf("etcdctl-%s", opts.targetVersion)
		if err := copyFile(filepath.Join(opts.binDir, etcdVersioned), filepath.Join(opts.binDir, "etcd")); err != nil {
			klog.Fatalf("Failed to copy %s: %v", etcdVersioned, err)
		}
		if err := copyFile(filepath.Join(opts.binDir, etcdctlVersioned), filepath.Join(opts.binDir, "etcdctl")); err != nil {
			klog.Fatalf("Failed to copy %s: %v", etcdctlVersioned, err)
		}
	}
}

// migrate opens or initializes the etcd data directory, configures the migrator, and starts the migration.
func migrate(name string, port uint64, peerListenUrls string, peerAdvertiseUrls string, clientListenUrls string,
	binPath string, dataDirPath string, etcdDataPrefix string, ttlKeysDirectory string,
	initialCluster string, target *EtcdVersionPair, bundledVersions SupportedVersions, etcdServerArgs string) {

	dataDir, err := OpenOrCreateDataDirectory(dataDirPath)
	if err != nil {
		klog.Fatalf("Error opening or creating data directory %s: %v", dataDirPath, err)
	}

	cfg := &EtcdMigrateCfg{
		binPath:           binPath,
		name:              name,
		port:              port,
		peerListenUrls:    peerListenUrls,
		peerAdvertiseUrls: peerAdvertiseUrls,
		clientListenUrls:  clientListenUrls,
		etcdDataPrefix:    etcdDataPrefix,
		ttlKeysDirectory:  ttlKeysDirectory,
		initialCluster:    initialCluster,
		supportedVersions: bundledVersions,
		dataDirectory:     dataDirPath,
		etcdServerArgs:    etcdServerArgs,
	}
	client, err := NewEtcdMigrateClient(cfg)
	if err != nil {
		klog.Fatalf("Migration failed: %v", err)
	}
	defer client.Close()

	migrator := &Migrator{cfg, dataDir, client}

	err = migrator.MigrateIfNeeded(target)
	if err != nil {
		klog.Fatalf("Migration failed: %v", err)
	}
}
