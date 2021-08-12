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
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/klog/v2"
)

const (
	versionFilename = "version.txt"
	// Snapshot file in backup-dir to restore the backup from.
	snapshotFilename = "snapshot.db"
	// File in the backup-dir containing etcd initial cluster token.
	tokenFilename = "initial-cluster-token.txt"
	// File in the backup-dir to which end result will be written.
	resultFilename = "result.txt"
	// Path of the db file within etcd data directory.
	dbFilepath = "/member/snap/db"
)

var (
	backupDir         string
	dataDir           string
	binDir            string
	memberName        string
	initialCluster    string
	peerAdvertiseUrls string
)

type SnapshotStatus struct {
	Hash      int
	Revision  int
	TotalKey  int
	TotalSize int
}

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

	tokenData, err := ioutil.ReadFile(filepath.Join(backupDir, tokenFilename))
	if err != nil {
		cleanupAndExit(fmt.Sprintf("error reading token file: %v", err))
	}
	token := strings.TrimSpace(string(tokenData))

	if err := os.Rename(dataDir, fmt.Sprintf("%s.bak", dataDir)); err != nil && !os.IsNotExist(err) {
		cleanupAndExit(fmt.Sprintf("error moving current data directory to %v: %v", fmt.Sprintf("%s.bak", dataDir), err))
	}
	klog.Infof("moved current data directory to %v", fmt.Sprintf("%s.bak", dataDir))

	if err = runEtcdctlRestore(etcdctlBin, etcdctlApi, token); err != nil {
		// If restoring the snapshot failed, attempt to move back the original data.
		os.Rename(fmt.Sprintf("%s.bak", dataDir), dataDir)
		cleanupAndExit(fmt.Sprintf("error restoring backup from snapshot: %v", err))
	}
	klog.Info("etcdctl restore ran successfully")

	if err = writeVersionFile(filepath.Join(dataDir, versionFilename), version, storageVersion); err != nil {
		cleanupAndExit(fmt.Sprintf("error writing new version file after restoring backup: %v", err))
	}
	klog.Info("successfully written new version file")

	if err = verifyDataIntegrity(etcdctlBin, etcdctlApi); err != nil {
		cleanupAndExit(fmt.Sprintf("failed to verify integity of the restored data: %v", err))
	}
	klog.Info("successfully verified integity of the recovered data - restore done, exiting")

	cleanupAndExit("")
}

func cleanupAndExit(err string) {
	os.RemoveAll(fmt.Sprintf("%s.bak", dataDir))
	os.Remove(filepath.Join(backupDir, versionFilename))
	os.Remove(filepath.Join(backupDir, snapshotFilename))
	os.Remove(filepath.Join(backupDir, tokenFilename))

	e := ioutil.WriteFile(filepath.Join(backupDir, resultFilename), []byte(err), 0666)
	if e != nil {
		klog.Fatalf("failed to write result to the file: %v", e)
	}

	if err != "" {
		klog.Fatalf("%v", err)
	}
}

func verifyDataIntegrity(etcdctlBin, etcdctlApi string) error {
	statusBackup, err := runEtcdctlStatus(etcdctlBin, etcdctlApi, filepath.Join(backupDir, snapshotFilename))
	if err != nil {
		return fmt.Errorf("error running etcdctl status on snapshot file: %v", err)
	}
	statusDB, err := runEtcdctlStatus(etcdctlBin, etcdctlApi, filepath.Join(dataDir, dbFilepath))
	if err != nil {
		return fmt.Errorf("error running etcdctl status on restored db file: %v", err)
	}

	if statusBackup.TotalKey != statusDB.TotalKey || statusDB.TotalSize != statusBackup.TotalSize {
		return fmt.Errorf("restored db (hash may differ) want: %v, got: %v", statusBackup, statusDB)
	}

	return nil
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

func runEtcdctlRestore(binary, etcdctlApi, token string) error {
	hasHash, err := hasIntegrityHash()
	if err != nil {
		return fmt.Errorf("error running etcdctl restore: %v", err)
	}
	if !hasHash {
		klog.Info("no integrity hash in the backup file - this can happen with backups made via 'backup before start'")
	}

	cmd := exec.Command(
		binary,
		"snapshot", "restore",
		filepath.Join(backupDir, snapshotFilename),
		"--data-dir", dataDir,
		"--name", memberName,
		"--initial-advertise-peer-urls", peerAdvertiseUrls,
		"--initial-cluster", initialCluster,
		"--initial-cluster-token", token,
		fmt.Sprintf("--skip-hash-check=%s", strconv.FormatBool(!hasHash)),
	)
	cmd.Env = []string{fmt.Sprintf("ETCDCTL_API=%v", etcdctlApi)}

	klog.Infof("about to run %s", cmd.String())
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func runEtcdctlStatus(binary, etcdctlApi, file string) (*SnapshotStatus, error) {
	cmd := exec.Command(
		binary,
		"snapshot", "status",
		file,
		"--write-out", "json",
	)

	cmd.Env = []string{fmt.Sprintf("ETCDCTL_API=%v", etcdctlApi)}
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	status := SnapshotStatus{}
	if err = json.Unmarshal(out, &status); err != nil {
		return nil, err
	}

	return &status, nil
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

// See https://github.com/etcd-io/etcd/blob/706f256a054b2158ba5dc2e59cbab45826063829/etcdutl/snapshot/v3_snapshot.go#L93.
func hasIntegrityHash() (bool, error) {
	stat, err := os.Stat(filepath.Join(backupDir, snapshotFilename))
	if err != nil {
		return false, fmt.Errorf("error checking if snapshot has integity hash: %v", err)
	}
	return (stat.Size() % 512) == sha256.Size, nil
}
