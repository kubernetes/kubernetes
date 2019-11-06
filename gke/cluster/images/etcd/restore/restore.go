/*
Copyright 2021 The Kubernetes Authors.

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

// The restore binary is responsible for checking
// and restoring etcd data directory from backup snapshot.
package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/klog/v2"
)

const (
	versionFilename = "version.txt"
	// Snapshot file in backup-dir to restore the backup from.
	snapshotFilename = "snapshot.db"
)

var (
	backupDir         string
	dataDir           string
	binDir            string
	memberName        string
	initialCluster    string
	peerAdvertiseUrls string
)

var restoreCmd = &cobra.Command{
	Short: "Restore etcd from backup if backup is present",
	Long: `Restore etcd from backup if backup is present.

Given a 'backup-dir' directory with snapshot.db and version.txt files, this tool will run etcdctl
snapshot restore with specified parameters.
`,
	Run: func(cmd *cobra.Command, args []string) {
		runRestore()
	},
}

func init() {
	flags := restoreCmd.Flags()
	flags.StringVar(&backupDir, "backup-dir", "", "directory from which backup can be restored")
	flags.StringVar(&dataDir, "data-dir", "", "directory where the restored etcd data will be placed")
	flags.StringVar(&binDir, "bin-dir", "/usr/local/bin", "directory where etcdctl binaries are")
	flags.StringVar(&memberName, "member-name", "", "etcd cluster member name")
	flags.StringVar(&initialCluster, "initial-cluster", "", "comma separated list of name=endpoint pairs")
	flags.StringVar(&peerAdvertiseUrls, "peer-advertise-urls", "", "etcd --initial-advertise-peer-urls flag")

	restoreCmd.MarkFlagRequired("backup-dir")
	restoreCmd.MarkFlagRequired("data-dir")
	restoreCmd.MarkFlagRequired("member-name")
	restoreCmd.MarkFlagRequired("initial-cluster")
	restoreCmd.MarkFlagRequired("peer-advertise-urls")
}

func main() {
	err := restoreCmd.Execute()
	if err != nil {
		cleanupAndExit(fmt.Sprintf("Failed to execute restorecmd: %s", err))
	}
}

func runRestore() {
	if _, err := os.Stat(filepath.Join(backupDir, versionFilename)); os.IsNotExist(err) {
		klog.Info("no backup to restore, exiting")
		return
	} else if err != nil {
		cleanupAndExit(fmt.Sprintf("error checking the backup version file: %v", err))
	}

	klog.Info("attempting to restore etcd backup")

	version, storageVersion, err := parseVersionFile(filepath.Join(backupDir, versionFilename))
	if err != nil {
		cleanupAndExit(fmt.Sprintf("error reading backup version file: %v", err))
	}
	klog.Info("version file parsed successfully")

	etcdctlBin, etcdctlApi, err := findEtcdctlBinary(version)
	if err != nil {
		cleanupAndExit(fmt.Sprintf("error finding the etcdctl binary: %v", err))
	}
	klog.Info("found correct etcdctl binary")

	if err := os.Rename(dataDir, fmt.Sprintf("%s.bak", dataDir)); err != nil && !os.IsNotExist(err) {
		cleanupAndExit(fmt.Sprintf("error moving current data directory to %v: %v", fmt.Sprintf("%s.bak", dataDir), err))
	}
	klog.Infof("moved current data directory to %v", fmt.Sprintf("%s.bak", dataDir))

	if err = runEtcdctlRestore(etcdctlBin, etcdctlApi); err != nil {
		// If restoring the snapshot failed, attempt to move back the original data.
		os.Rename(fmt.Sprintf("%s.bak", dataDir), dataDir)
		cleanupAndExit(fmt.Sprintf("error restoring backup from snapshot: %v", err))
	}
	klog.Info("etcdctl restore ran successfully")

	if err = writeVersionFile(filepath.Join(dataDir, versionFilename), version, storageVersion); err != nil {
		cleanupAndExit(fmt.Sprintf("error writing new version file after restoring backup: %v", err))
	}
	klog.Info("successfully written new version file - restore done, exiting")

	cleanupAndExit("")
}

func cleanupAndExit(err string) {
	os.RemoveAll(fmt.Sprintf("%s.bak", dataDir))
	os.RemoveAll(backupDir)

	if err != "" {
		klog.Fatalf("%v", err)
	}
}

// findEtcdctlBinary looks for binary which matches selected minor and major version.
// Also returns etcdctl api version.
func findEtcdctlBinary(version string) (string, string, error) {
	versionSplit := strings.Split(version, ".")
	if len(versionSplit) != 3 {
		return "", "", fmt.Errorf("malformed version, expected <major>.<minor>.<patch>, but got %s", version)
	}
	etcdctlBins, err := filepath.Glob(fmt.Sprintf("%v/etcdctl-%v.%v.*", binDir, versionSplit[0], versionSplit[1]))
	if err != nil || len(etcdctlBins) == 0 {
		return "", "", fmt.Errorf("no available etcdctl binaries: %v", err)
	}

	return etcdctlBins[0], versionSplit[0], nil
}

func runEtcdctlRestore(binary, etcdctlApi string) error {
	cmd := exec.Command(
		binary,
		"snapshot", "restore",
		filepath.Join(backupDir, snapshotFilename),
		"--data-dir", dataDir,
		"--name", memberName,
		"--initial-advertise-peer-urls", peerAdvertiseUrls,
		"--initial-cluster", initialCluster,
	)
	cmd.Env = []string{fmt.Sprintf("ETCDCTL_API=%v", etcdctlApi)}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// parseVersionFile, given an etcd version file path, returns etcd version
// (in form of <major>.<minor>.<patch>) and storage version.
// It returns two empty strings and error on failure.
func parseVersionFile(path string) (string, string, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return "", "", fmt.Errorf("failed to read version file %s: %v", path, err)
	}

	txt := strings.TrimSpace(string(data))
	parts := strings.Split(txt, "/")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("malformed version file, expected <major>.<minor>.<patch>/<storage> but got %s", txt)
	}

	return parts[0], parts[1], nil
}

func writeVersionFile(path, version, storageVersion string) error {
	data := []byte(fmt.Sprintf("%s/%s", version, storageVersion))
	return ioutil.WriteFile(path, data, 0666)
}
