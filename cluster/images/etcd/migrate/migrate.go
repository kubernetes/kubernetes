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
	"path/filepath"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

const (
	versionFilename        = "version.txt"
	defaultPort     uint64 = 18629
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

type migrateOpts struct {
	name                 string
	port                 uint64
	peerListenUrls       string
	peerAdvertiseUrls    string
	binDir               string
	dataDir              string
	bundledVersionString string
	etcdDataPrefix       string
	ttlKeysDirectory     string
	initialCluster       string
	targetVersion        string
	targetStorage        string
	etcdServerArgs       string
}

func main() {
	flags := migrateCmd.Flags()
	flags.StringVar(&opts.name, "name", "", "etcd cluster member name. Defaults to etcd-{hostname}")
	flags.Uint64Var(&opts.port, "port", defaultPort, "etcd client port to use during migration operations. This should be a different port than typically used by etcd to avoid clients accidentally connecting during upgrade/downgrade operations.")
	flags.StringVar(&opts.peerListenUrls, "listen-peer-urls", "", "etcd --listen-peer-urls flag, required for HA clusters")
	flags.StringVar(&opts.peerAdvertiseUrls, "initial-advertise-peer-urls", "", "etcd --initial-advertise-peer-urls flag, required for HA clusters")
	flags.StringVar(&opts.binDir, "bin-dir", "/usr/local/bin", "directory of etcd and etcdctl binaries, must contain etcd-<version> and etcdctl-<version> for each version listed in bindled-versions")
	flags.StringVar(&opts.dataDir, "data-dir", "", "etcd data directory of etcd server to migrate")
	flags.StringVar(&opts.bundledVersionString, "bundled-versions", "", "comma separated list of etcd binary versions present under the bin-dir")
	flags.StringVar(&opts.etcdDataPrefix, "etcd-data-prefix", "/registry", "etcd key prefix under which all objects are kept")
	flags.StringVar(&opts.ttlKeysDirectory, "ttl-keys-directory", "", "etcd key prefix under which all keys with TTLs are kept. Defaults to {etcd-data-prefix}/events")
	flags.StringVar(&opts.initialCluster, "initial-cluster", "", "comma separated list of name=endpoint pairs. Defaults to etcd-{hostname}=http://localhost:2380")
	flags.StringVar(&opts.targetVersion, "target-version", "", "version of etcd to migrate to. Format must be '<major>.<minor>.<patch>'")
	flags.StringVar(&opts.targetStorage, "target-storage", "", "storage version of etcd to migrate to, one of: etcd2, etcd3")
	flags.StringVar(&opts.etcdServerArgs, "etcd-server-extra-args", "", "additional etcd server args for starting etcd servers during migration steps, --peer-* TLS cert flags should be added for etcd clusters with more than 1 member that use mutual TLS for peer communication.")
	migrateCmd.Execute()
}

// runMigrate validates the command line flags and starts the migration.
func runMigrate() {
	if opts.name == "" {
		hostname, err := os.Hostname()
		if err != nil {
			glog.Errorf("Error while getting hostname to supply default --name: %v", err)
			os.Exit(1)
		}
		opts.name = fmt.Sprintf("etcd-%s", hostname)
	}

	if opts.ttlKeysDirectory == "" {
		opts.ttlKeysDirectory = fmt.Sprintf("%s/events", opts.etcdDataPrefix)
	}
	if opts.initialCluster == "" {
		opts.initialCluster = fmt.Sprintf("%s=http://localhost:2380", opts.name)
	}
	if opts.targetStorage == "" {
		glog.Errorf("--target-storage is required")
		os.Exit(1)
	}
	if opts.targetVersion == "" {
		glog.Errorf("--target-version is required")
		os.Exit(1)
	}
	if opts.dataDir == "" {
		glog.Errorf("--data-dir is required")
		os.Exit(1)
	}
	if opts.bundledVersionString == "" {
		glog.Errorf("--bundled-versions is required")
		os.Exit(1)
	}

	bundledVersions, err := ParseSupportedVersions(opts.bundledVersionString)
	if err != nil {
		glog.Errorf("Failed to parse --supported-versions: %v", err)
	}
	err = validateBundledVersions(bundledVersions, opts.binDir)
	if err != nil {
		glog.Errorf("Failed to validate that 'etcd-<version>' and 'etcdctl-<version>' binaries exist in --bin-dir '%s' for all --bundled-verions '%s': %v",
			opts.binDir, opts.bundledVersionString, err)
		os.Exit(1)
	}

	target := &EtcdVersionPair{
		version:        MustParseEtcdVersion(opts.targetVersion),
		storageVersion: MustParseEtcdStorageVersion(opts.targetStorage),
	}

	migrate(opts.name, opts.port, opts.peerListenUrls, opts.peerAdvertiseUrls, opts.binDir, opts.dataDir, opts.etcdDataPrefix, opts.ttlKeysDirectory, opts.initialCluster, target, bundledVersions, opts.etcdServerArgs)
}

// migrate opens or initializes the etcd data directory, configures the migrator, and starts the migration.
func migrate(name string, port uint64, peerListenUrls string, peerAdvertiseUrls string, binPath string, dataDirPath string, etcdDataPrefix string, ttlKeysDirectory string,
	initialCluster string, target *EtcdVersionPair, bundledVersions SupportedVersions, etcdServerArgs string) {

	dataDir, err := OpenOrCreateDataDirectory(dataDirPath)
	if err != nil {
		glog.Errorf("Error opening or creating data directory %s: %v", dataDirPath, err)
		os.Exit(1)
	}

	cfg := &EtcdMigrateCfg{
		binPath:           binPath,
		name:              name,
		port:              port,
		peerListenUrls:    peerListenUrls,
		peerAdvertiseUrls: peerAdvertiseUrls,
		etcdDataPrefix:    etcdDataPrefix,
		ttlKeysDirectory:  ttlKeysDirectory,
		initialCluster:    initialCluster,
		supportedVersions: bundledVersions,
		dataDirectory:     dataDirPath,
		etcdServerArgs:    etcdServerArgs,
	}
	client, err := NewEtcdMigrateClient(cfg)
	if err != nil {
		glog.Errorf("Migration failed: %v", err)
		os.Exit(1)
	}
	defer client.Close()

	migrator := &Migrator{cfg, dataDir, client}

	err = migrator.MigrateIfNeeded(target)
	if err != nil {
		glog.Errorf("Migration failed: %v", err)
		os.Exit(1)
	}
}

// validateBundledVersions checks that 'etcd-<version>' and 'etcdctl-<version>' binaries exist in the binDir
// for each version in the bundledVersions list.
func validateBundledVersions(bundledVersions SupportedVersions, binDir string) error {
	for _, v := range bundledVersions {
		for _, binaryName := range []string{"etcd", "etcdctl"} {
			fn := filepath.Join(binDir, fmt.Sprintf("%s-%s", binaryName, v))
			if _, err := os.Stat(fn); err != nil {
				return fmt.Errorf("failed to validate '%s' binary exists for bundled-version '%s': %v", fn, v, err)
			}

		}
	}
	return nil
}
