/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package master

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util"

	"github.com/stretchr/testify/assert"
)

// TestSecondsSinceSync verifies that proper results are returned
// when checking the time between syncs
func TestSecondsSinceSync(t *testing.T) {
	tunneler := &SSHTunneler{}
	assert := assert.New(t)

	tunneler.lastSync = time.Date(2015, time.January, 1, 1, 1, 1, 1, time.UTC).Unix()

	// Nano Second. No difference.
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.January, 1, 1, 1, 1, 2, time.UTC)}
	assert.Equal(int64(0), tunneler.SecondsSinceSync())

	// Second
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.January, 1, 1, 1, 2, 1, time.UTC)}
	assert.Equal(int64(1), tunneler.SecondsSinceSync())

	// Minute
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.January, 1, 1, 2, 1, 1, time.UTC)}
	assert.Equal(int64(60), tunneler.SecondsSinceSync())

	// Hour
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.January, 1, 2, 1, 1, 1, time.UTC)}
	assert.Equal(int64(3600), tunneler.SecondsSinceSync())

	// Day
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.January, 2, 1, 1, 1, 1, time.UTC)}
	assert.Equal(int64(86400), tunneler.SecondsSinceSync())

	// Month
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.February, 1, 1, 1, 1, 1, time.UTC)}
	assert.Equal(int64(2678400), tunneler.SecondsSinceSync())

	// Future Month. Should be -Month.
	tunneler.lastSync = time.Date(2015, time.February, 1, 1, 1, 1, 1, time.UTC).Unix()
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.January, 1, 1, 1, 1, 1, time.UTC)}
	assert.Equal(int64(-2678400), tunneler.SecondsSinceSync())
}

// TestRefreshTunnels verifies that the function errors when no addresses
// are associated with nodes
func TestRefreshTunnels(t *testing.T) {
	tunneler := &SSHTunneler{}
	tunneler.getAddresses = func() ([]string, error) { return []string{}, nil }
	assert := assert.New(t)

	// Fail case (no addresses associated with nodes)
	assert.Error(tunneler.refreshTunnels("test", "/tmp/undefined"))

	// TODO: pass case without needing actual connections?
}

// TestIsTunnelSyncHealthy verifies that the 600 second lag test
// is honored.
func TestIsTunnelSyncHealthy(t *testing.T) {
	tunneler := &SSHTunneler{}
	master, _, assert := setUp(t)
	master.tunneler = tunneler

	// Pass case: 540 second lag
	tunneler.lastSync = time.Date(2015, time.January, 1, 1, 1, 1, 1, time.UTC).Unix()
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.January, 1, 1, 9, 1, 1, time.UTC)}
	err := master.IsTunnelSyncHealthy(nil)
	assert.NoError(err, "IsTunnelSyncHealthy() should not have returned an error.")

	// Fail case: 720 second lag
	tunneler.clock = &util.FakeClock{Time: time.Date(2015, time.January, 1, 1, 12, 1, 1, time.UTC)}
	err = master.IsTunnelSyncHealthy(nil)
	assert.Error(err, "IsTunnelSyncHealthy() should have returned an error.")
}

// generateTempFile creates a temporary file path
func generateTempFilePath(prefix string) string {
	tmpPath, _ := filepath.Abs(fmt.Sprintf("%s/%s-%d", os.TempDir(), prefix, time.Now().Unix()))
	return tmpPath
}

// TestGenerateSSHKey verifies that SSH key generation does indeed
// generate keys even with keys already exist.
func TestGenerateSSHKey(t *testing.T) {
	tunneler := &SSHTunneler{}
	assert := assert.New(t)

	privateKey := generateTempFilePath("private")
	publicKey := generateTempFilePath("public")

	// Make sure we have no test keys laying around
	os.Remove(privateKey)
	os.Remove(publicKey)

	// Pass case: Sunny day case
	err := tunneler.generateSSHKey("unused", privateKey, publicKey)
	assert.NoError(err, "generateSSHKey should not have retuend an error: %s", err)

	// Pass case: PrivateKey exists test case
	os.Remove(publicKey)
	err = tunneler.generateSSHKey("unused", privateKey, publicKey)
	assert.NoError(err, "generateSSHKey should not have retuend an error: %s", err)

	// Pass case: PublicKey exists test case
	os.Remove(privateKey)
	err = tunneler.generateSSHKey("unused", privateKey, publicKey)
	assert.NoError(err, "generateSSHKey should not have retuend an error: %s", err)

	// Make sure we have no test keys laying around
	os.Remove(privateKey)
	os.Remove(publicKey)

	// TODO: testing error cases where the file can not be removed?
}
