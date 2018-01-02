// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package e2e

import (
	"io/ioutil"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coreos/etcd/pkg/fileutil"
	"github.com/coreos/etcd/pkg/testutil"
)

func TestCtlV2Set(t *testing.T)          { testCtlV2Set(t, &configNoTLS, false) }
func TestCtlV2SetQuorum(t *testing.T)    { testCtlV2Set(t, &configNoTLS, true) }
func TestCtlV2SetClientTLS(t *testing.T) { testCtlV2Set(t, &configClientTLS, false) }
func TestCtlV2SetPeerTLS(t *testing.T)   { testCtlV2Set(t, &configPeerTLS, false) }
func TestCtlV2SetTLS(t *testing.T)       { testCtlV2Set(t, &configTLS, false) }
func testCtlV2Set(t *testing.T, cfg *etcdProcessClusterConfig, quorum bool) {
	defer testutil.AfterTest(t)

	epc := setupEtcdctlTest(t, cfg, quorum)
	defer func() {
		if errC := epc.Close(); errC != nil {
			t.Fatalf("error closing etcd processes (%v)", errC)
		}
	}()

	key, value := "foo", "bar"

	if err := etcdctlSet(epc, key, value); err != nil {
		t.Fatalf("failed set (%v)", err)
	}

	if err := etcdctlGet(epc, key, value, quorum); err != nil {
		t.Fatalf("failed get (%v)", err)
	}
}

func TestCtlV2Mk(t *testing.T)       { testCtlV2Mk(t, &configNoTLS, false) }
func TestCtlV2MkQuorum(t *testing.T) { testCtlV2Mk(t, &configNoTLS, true) }
func TestCtlV2MkTLS(t *testing.T)    { testCtlV2Mk(t, &configTLS, false) }
func testCtlV2Mk(t *testing.T, cfg *etcdProcessClusterConfig, quorum bool) {
	defer testutil.AfterTest(t)

	epc := setupEtcdctlTest(t, cfg, quorum)
	defer func() {
		if errC := epc.Close(); errC != nil {
			t.Fatalf("error closing etcd processes (%v)", errC)
		}
	}()

	key, value := "foo", "bar"

	if err := etcdctlMk(epc, key, value, true); err != nil {
		t.Fatalf("failed mk (%v)", err)
	}
	if err := etcdctlMk(epc, key, value, false); err != nil {
		t.Fatalf("failed mk (%v)", err)
	}

	if err := etcdctlGet(epc, key, value, quorum); err != nil {
		t.Fatalf("failed get (%v)", err)
	}
}

func TestCtlV2Rm(t *testing.T)    { testCtlV2Rm(t, &configNoTLS) }
func TestCtlV2RmTLS(t *testing.T) { testCtlV2Rm(t, &configTLS) }
func testCtlV2Rm(t *testing.T, cfg *etcdProcessClusterConfig) {
	defer testutil.AfterTest(t)

	epc := setupEtcdctlTest(t, cfg, true)
	defer func() {
		if errC := epc.Close(); errC != nil {
			t.Fatalf("error closing etcd processes (%v)", errC)
		}
	}()

	key, value := "foo", "bar"

	if err := etcdctlSet(epc, key, value); err != nil {
		t.Fatalf("failed set (%v)", err)
	}

	if err := etcdctlRm(epc, key, value, true); err != nil {
		t.Fatalf("failed rm (%v)", err)
	}
	if err := etcdctlRm(epc, key, value, false); err != nil {
		t.Fatalf("failed rm (%v)", err)
	}
}

func TestCtlV2Ls(t *testing.T)       { testCtlV2Ls(t, &configNoTLS, false) }
func TestCtlV2LsQuorum(t *testing.T) { testCtlV2Ls(t, &configNoTLS, true) }
func TestCtlV2LsTLS(t *testing.T)    { testCtlV2Ls(t, &configTLS, false) }
func testCtlV2Ls(t *testing.T, cfg *etcdProcessClusterConfig, quorum bool) {
	defer testutil.AfterTest(t)

	epc := setupEtcdctlTest(t, cfg, quorum)
	defer func() {
		if errC := epc.Close(); errC != nil {
			t.Fatalf("error closing etcd processes (%v)", errC)
		}
	}()

	key, value := "foo", "bar"

	if err := etcdctlSet(epc, key, value); err != nil {
		t.Fatalf("failed set (%v)", err)
	}

	if err := etcdctlLs(epc, key, quorum); err != nil {
		t.Fatalf("failed ls (%v)", err)
	}
}

func TestCtlV2Watch(t *testing.T)                { testCtlV2Watch(t, &configNoTLS, false) }
func TestCtlV2WatchTLS(t *testing.T)             { testCtlV2Watch(t, &configTLS, false) }
func TestCtlV2WatchWithProxy(t *testing.T)       { testCtlV2Watch(t, &configWithProxy, false) }
func TestCtlV2WatchWithProxyNoSync(t *testing.T) { testCtlV2Watch(t, &configWithProxy, true) }
func testCtlV2Watch(t *testing.T, cfg *etcdProcessClusterConfig, noSync bool) {
	defer testutil.AfterTest(t)

	epc := setupEtcdctlTest(t, cfg, true)
	defer func() {
		if errC := epc.Close(); errC != nil {
			t.Fatalf("error closing etcd processes (%v)", errC)
		}
	}()

	key, value := "foo", "bar"
	errc := etcdctlWatch(epc, key, value, noSync)
	if err := etcdctlSet(epc, key, value); err != nil {
		t.Fatalf("failed set (%v)", err)
	}

	select {
	case err := <-errc:
		if err != nil {
			t.Fatalf("failed watch (%v)", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("watch timed out")
	}
}

func TestCtlV2GetRoleUser(t *testing.T)          { testCtlV2GetRoleUser(t, &configNoTLS) }
func TestCtlV2GetRoleUserWithProxy(t *testing.T) { testCtlV2GetRoleUser(t, &configWithProxy) }
func testCtlV2GetRoleUser(t *testing.T, cfg *etcdProcessClusterConfig) {
	defer testutil.AfterTest(t)

	epc := setupEtcdctlTest(t, cfg, true)
	defer func() {
		if err := epc.Close(); err != nil {
			t.Fatalf("error closing etcd processes (%v)", err)
		}
	}()

	if err := etcdctlRoleAdd(epc, "foo"); err != nil {
		t.Fatalf("failed to add role (%v)", err)
	}
	if err := etcdctlUserAdd(epc, "username", "password"); err != nil {
		t.Fatalf("failed to add user (%v)", err)
	}
	if err := etcdctlUserGrant(epc, "username", "foo"); err != nil {
		t.Fatalf("failed to grant role (%v)", err)
	}
	if err := etcdctlUserGet(epc, "username"); err != nil {
		t.Fatalf("failed to get user (%v)", err)
	}

	// ensure double grant gives an error; was crashing in 2.3.1
	regrantArgs := etcdctlPrefixArgs(epc)
	regrantArgs = append(regrantArgs, "user", "grant", "--roles", "foo", "username")
	if err := spawnWithExpect(regrantArgs, "duplicate"); err != nil {
		t.Fatalf("missing duplicate error on double grant role (%v)", err)
	}
}

func TestCtlV2UserListUsername(t *testing.T) { testCtlV2UserList(t, "username") }
func TestCtlV2UserListRoot(t *testing.T)     { testCtlV2UserList(t, "root") }
func testCtlV2UserList(t *testing.T, username string) {
	defer testutil.AfterTest(t)

	epc := setupEtcdctlTest(t, &configWithProxy, false)
	defer func() {
		if err := epc.Close(); err != nil {
			t.Fatalf("error closing etcd processes (%v)", err)
		}
	}()

	if err := etcdctlUserAdd(epc, username, "password"); err != nil {
		t.Fatalf("failed to add user (%v)", err)
	}
	if err := etcdctlUserList(epc, username); err != nil {
		t.Fatalf("failed to list users (%v)", err)
	}
}

func TestCtlV2RoleList(t *testing.T) {
	defer testutil.AfterTest(t)

	epc := setupEtcdctlTest(t, &configWithProxy, false)
	defer func() {
		if err := epc.Close(); err != nil {
			t.Fatalf("error closing etcd processes (%v)", err)
		}
	}()

	if err := etcdctlRoleAdd(epc, "foo"); err != nil {
		t.Fatalf("failed to add role (%v)", err)
	}
	if err := etcdctlRoleList(epc, "foo"); err != nil {
		t.Fatalf("failed to list roles (%v)", err)
	}
}

func TestCtlV2Backup(t *testing.T) { // For https://github.com/coreos/etcd/issues/5360
	defer testutil.AfterTest(t)

	backupDir, err := ioutil.TempDir("", "testbackup0.etcd")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(backupDir)

	epc1 := setupEtcdctlTest(t, &configNoTLS, false)
	if err := etcdctlSet(epc1, "foo1", "bar"); err != nil {
		t.Fatal(err)
	}

	if err := etcdctlBackup(epc1, epc1.procs[0].cfg.dataDirPath, backupDir); err != nil {
		t.Fatal(err)
	}

	if err := epc1.Close(); err != nil {
		t.Fatalf("error closing etcd processes (%v)", err)
	}

	// restart from the backup directory
	cfg2 := configNoTLS
	cfg2.dataDirPath = backupDir
	cfg2.keepDataDir = true
	cfg2.forceNewCluster = true
	epc2 := setupEtcdctlTest(t, &cfg2, false)

	// check if backup went through correctly
	if err := etcdctlGet(epc2, "foo1", "bar", false); err != nil {
		t.Fatal(err)
	}

	// check if it can serve client requests
	if err := etcdctlSet(epc2, "foo2", "bar"); err != nil {
		t.Fatal(err)
	}
	if err := etcdctlGet(epc2, "foo2", "bar", false); err != nil {
		t.Fatal(err)
	}

	if err := epc2.Close(); err != nil {
		t.Fatalf("error closing etcd processes (%v)", err)
	}
}

func TestCtlV2AuthWithCommonName(t *testing.T) {
	defer testutil.AfterTest(t)

	copiedCfg := configClientTLS
	copiedCfg.clientCertAuthEnabled = true

	epc := setupEtcdctlTest(t, &copiedCfg, false)
	defer func() {
		if err := epc.Close(); err != nil {
			t.Fatalf("error closing etcd processes (%v)", err)
		}
	}()

	if err := etcdctlRoleAdd(epc, "testrole"); err != nil {
		t.Fatalf("failed to add role (%v)", err)
	}
	if err := etcdctlRoleGrant(epc, "testrole", "--rw", "--path=/foo"); err != nil {
		t.Fatalf("failed to grant role (%v)", err)
	}
	if err := etcdctlUserAdd(epc, "root", "123"); err != nil {
		t.Fatalf("failed to add user (%v)", err)
	}
	if err := etcdctlUserAdd(epc, "Autogenerated CA", "123"); err != nil {
		t.Fatalf("failed to add user (%v)", err)
	}
	if err := etcdctlUserGrant(epc, "Autogenerated CA", "testrole"); err != nil {
		t.Fatalf("failed to grant role (%v)", err)
	}
	if err := etcdctlAuthEnable(epc); err != nil {
		t.Fatalf("failed to enable auth (%v)", err)
	}
	if err := etcdctlSet(epc, "foo", "bar"); err != nil {
		t.Fatalf("failed to write (%v)", err)
	}
}

func TestCtlV2ClusterHealth(t *testing.T) {
	defer testutil.AfterTest(t)
	epc := setupEtcdctlTest(t, &configNoTLS, true)
	defer func() {
		if err := epc.Close(); err != nil {
			t.Fatalf("error closing etcd processes (%v)", err)
		}
	}()

	// has quorum
	if err := etcdctlClusterHealth(epc, "cluster is healthy"); err != nil {
		t.Fatalf("cluster-health expected to be healthy (%v)", err)
	}

	// cut quorum
	epc.procs[0].Stop()
	epc.procs[1].Stop()
	if err := etcdctlClusterHealth(epc, "cluster is unhealthy"); err != nil {
		t.Fatalf("cluster-health expected to be unhealthy (%v)", err)
	}
	epc.procs[0], epc.procs[1] = nil, nil
}

func etcdctlPrefixArgs(clus *etcdProcessCluster) []string {
	endpoints := ""
	if proxies := clus.proxies(); len(proxies) != 0 {
		endpoints = proxies[0].cfg.acurl
	} else if processes := clus.processes(); len(processes) != 0 {
		es := []string{}
		for _, b := range processes {
			es = append(es, b.cfg.acurl)
		}
		endpoints = strings.Join(es, ",")
	}
	cmdArgs := []string{ctlBinPath, "--endpoints", endpoints}
	if clus.cfg.clientTLS == clientTLS {
		cmdArgs = append(cmdArgs, "--ca-file", caPath, "--cert-file", certPath, "--key-file", privateKeyPath)
	}
	return cmdArgs
}

func etcdctlClusterHealth(clus *etcdProcessCluster, val string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "cluster-health")
	return spawnWithExpect(cmdArgs, val)
}

func etcdctlSet(clus *etcdProcessCluster, key, value string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "set", key, value)
	return spawnWithExpect(cmdArgs, value)
}

func etcdctlMk(clus *etcdProcessCluster, key, value string, first bool) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "mk", key, value)
	if first {
		return spawnWithExpect(cmdArgs, value)
	}
	return spawnWithExpect(cmdArgs, "Error:  105: Key already exists")
}

func etcdctlGet(clus *etcdProcessCluster, key, value string, quorum bool) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "get", key)
	if quorum {
		cmdArgs = append(cmdArgs, "--quorum")
	}
	return spawnWithExpect(cmdArgs, value)
}

func etcdctlRm(clus *etcdProcessCluster, key, value string, first bool) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "rm", key)
	if first {
		return spawnWithExpect(cmdArgs, "PrevNode.Value: "+value)
	}
	return spawnWithExpect(cmdArgs, "Error:  100: Key not found")
}

func etcdctlLs(clus *etcdProcessCluster, key string, quorum bool) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "ls")
	if quorum {
		cmdArgs = append(cmdArgs, "--quorum")
	}
	return spawnWithExpect(cmdArgs, key)
}

func etcdctlWatch(clus *etcdProcessCluster, key, value string, noSync bool) <-chan error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "watch", "--after-index=1", key)
	if noSync {
		cmdArgs = append(cmdArgs, "--no-sync")
	}
	errc := make(chan error, 1)
	go func() {
		errc <- spawnWithExpect(cmdArgs, value)
	}()
	return errc
}

func etcdctlRoleAdd(clus *etcdProcessCluster, role string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "role", "add", role)
	return spawnWithExpect(cmdArgs, role)
}

func etcdctlRoleGrant(clus *etcdProcessCluster, role string, perms ...string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "role", "grant")
	cmdArgs = append(cmdArgs, perms...)
	cmdArgs = append(cmdArgs, role)
	return spawnWithExpect(cmdArgs, role)
}

func etcdctlRoleList(clus *etcdProcessCluster, expectedRole string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "role", "list")
	return spawnWithExpect(cmdArgs, expectedRole)
}

func etcdctlUserAdd(clus *etcdProcessCluster, user, pass string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "user", "add", user+":"+pass)
	return spawnWithExpect(cmdArgs, "User "+user+" created")
}

func etcdctlUserGrant(clus *etcdProcessCluster, user, role string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "user", "grant", "--roles", role, user)
	return spawnWithExpect(cmdArgs, "User "+user+" updated")
}

func etcdctlUserGet(clus *etcdProcessCluster, user string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "user", "get", user)
	return spawnWithExpect(cmdArgs, "User: "+user)
}

func etcdctlUserList(clus *etcdProcessCluster, expectedUser string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "user", "list")
	return spawnWithExpect(cmdArgs, expectedUser)
}

func etcdctlAuthEnable(clus *etcdProcessCluster) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "auth", "enable")
	return spawnWithExpect(cmdArgs, "Authentication Enabled")
}

func etcdctlBackup(clus *etcdProcessCluster, dataDir, backupDir string) error {
	cmdArgs := append(etcdctlPrefixArgs(clus), "backup", "--data-dir", dataDir, "--backup-dir", backupDir)
	return spawnWithExpects(cmdArgs)
}

func mustEtcdctl(t *testing.T) {
	if !fileutil.Exist(binDir + "/etcdctl") {
		t.Fatalf("could not find etcdctl binary")
	}
}

func setupEtcdctlTest(t *testing.T, cfg *etcdProcessClusterConfig, quorum bool) *etcdProcessCluster {
	mustEtcdctl(t)
	if !quorum {
		cfg = configStandalone(*cfg)
	}
	epc, err := newEtcdProcessCluster(cfg)
	if err != nil {
		t.Fatalf("could not start etcd process cluster (%v)", err)
	}
	return epc
}
