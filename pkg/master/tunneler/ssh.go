/*
Copyright 2015 The Kubernetes Authors.

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

package tunneler

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/clock"
	"k8s.io/kubernetes/pkg/ssh"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

type InstallSSHKey func(user string, data []byte) error

type AddressFunc func() (addresses []string, err error)

type Tunneler interface {
	Run(AddressFunc)
	Stop()
	Dial(net, addr string) (net.Conn, error)
	SecondsSinceSync() int64
	SecondsSinceSSHKeySync() int64
}

// TunnelSyncHealthChecker returns a health func that indicates if a tunneler is healthy.
// It's compatible with healthz.NamedCheck
func TunnelSyncHealthChecker(tunneler Tunneler) func(req *http.Request) error {
	return func(req *http.Request) error {
		if tunneler == nil {
			return nil
		}
		lag := tunneler.SecondsSinceSync()
		if lag > 600 {
			return fmt.Errorf("Tunnel sync is taking to long: %d", lag)
		}
		sshKeyLag := tunneler.SecondsSinceSSHKeySync()
		if sshKeyLag > 600 {
			return fmt.Errorf("SSHKey sync is taking to long: %d", sshKeyLag)
		}
		return nil
	}
}

type SSHTunneler struct {
	// Important: Since these two int64 fields are using sync/atomic, they have to be at the top of the struct due to a bug on 32-bit platforms
	// See: https://golang.org/pkg/sync/atomic/ for more information
	lastSync       int64 // Seconds since Epoch
	lastSSHKeySync int64 // Seconds since Epoch

	SSHUser        string
	SSHKeyfile     string
	InstallSSHKey  InstallSSHKey
	HealthCheckURL *url.URL

	tunnels        *ssh.SSHTunnelList
	lastSyncMetric prometheus.GaugeFunc
	clock          clock.Clock

	getAddresses AddressFunc
	stopChan     chan struct{}
}

func New(sshUser, sshKeyfile string, healthCheckURL *url.URL, installSSHKey InstallSSHKey) Tunneler {
	return &SSHTunneler{
		SSHUser:        sshUser,
		SSHKeyfile:     sshKeyfile,
		InstallSSHKey:  installSSHKey,
		HealthCheckURL: healthCheckURL,
		clock:          clock.RealClock{},
	}
}

// Run establishes tunnel loops and returns
func (c *SSHTunneler) Run(getAddresses AddressFunc) {
	if c.stopChan != nil {
		return
	}
	c.stopChan = make(chan struct{})

	// Save the address getter
	if getAddresses != nil {
		c.getAddresses = getAddresses
	}

	// Usernames are capped @ 32
	if len(c.SSHUser) > 32 {
		glog.Warning("SSH User is too long, truncating to 32 chars")
		c.SSHUser = c.SSHUser[0:32]
	}
	glog.Infof("Setting up proxy: %s %s", c.SSHUser, c.SSHKeyfile)

	// public keyfile is written last, so check for that.
	publicKeyFile := c.SSHKeyfile + ".pub"
	exists, err := util.FileExists(publicKeyFile)
	if err != nil {
		glog.Errorf("Error detecting if key exists: %v", err)
	} else if !exists {
		glog.Infof("Key doesn't exist, attempting to create")
		if err := generateSSHKey(c.SSHKeyfile, publicKeyFile); err != nil {
			glog.Errorf("Failed to create key pair: %v", err)
		}
	}

	c.tunnels = ssh.NewSSHTunnelList(c.SSHUser, c.SSHKeyfile, c.HealthCheckURL, c.stopChan)
	// Sync loop to ensure that the SSH key has been installed.
	c.lastSSHKeySync = c.clock.Now().Unix()
	c.installSSHKeySyncLoop(c.SSHUser, publicKeyFile)
	// Sync tunnelList w/ nodes.
	c.lastSync = c.clock.Now().Unix()
	c.nodesSyncLoop()
}

// Stop gracefully shuts down the tunneler
func (c *SSHTunneler) Stop() {
	if c.stopChan != nil {
		close(c.stopChan)
		c.stopChan = nil
	}
}

func (c *SSHTunneler) Dial(net, addr string) (net.Conn, error) {
	return c.tunnels.Dial(net, addr)
}

func (c *SSHTunneler) SecondsSinceSync() int64 {
	now := c.clock.Now().Unix()
	then := atomic.LoadInt64(&c.lastSync)
	return now - then
}

func (c *SSHTunneler) SecondsSinceSSHKeySync() int64 {
	now := c.clock.Now().Unix()
	then := atomic.LoadInt64(&c.lastSSHKeySync)
	return now - then
}

func (c *SSHTunneler) installSSHKeySyncLoop(user, publicKeyfile string) {
	go wait.Until(func() {
		if c.InstallSSHKey == nil {
			glog.Error("Won't attempt to install ssh key: InstallSSHKey function is nil")
			return
		}
		key, err := ssh.ParsePublicKeyFromFile(publicKeyfile)
		if err != nil {
			glog.Errorf("Failed to load public key: %v", err)
			return
		}
		keyData, err := ssh.EncodeSSHKey(key)
		if err != nil {
			glog.Errorf("Failed to encode public key: %v", err)
			return
		}
		if err := c.InstallSSHKey(user, keyData); err != nil {
			glog.Errorf("Failed to install ssh key: %v", err)
			return
		}
		atomic.StoreInt64(&c.lastSSHKeySync, c.clock.Now().Unix())
	}, 5*time.Minute, c.stopChan)
}

// nodesSyncLoop lists nodes every 15 seconds, calling Update() on the TunnelList
// each time (Update() is a noop if no changes are necessary).
func (c *SSHTunneler) nodesSyncLoop() {
	// TODO (cjcullen) make this watch.
	go wait.Until(func() {
		addrs, err := c.getAddresses()
		glog.V(4).Infof("Calling update w/ addrs: %v", addrs)
		if err != nil {
			glog.Errorf("Failed to getAddresses: %v", err)
		}
		c.tunnels.Update(addrs)
		atomic.StoreInt64(&c.lastSync, c.clock.Now().Unix())
	}, 15*time.Second, c.stopChan)
}

func generateSSHKey(privateKeyfile, publicKeyfile string) error {
	private, public, err := ssh.GenerateKey(2048)
	if err != nil {
		return err
	}
	// If private keyfile already exists, we must have only made it halfway
	// through last time, so delete it.
	exists, err := util.FileExists(privateKeyfile)
	if err != nil {
		glog.Errorf("Error detecting if private key exists: %v", err)
	} else if exists {
		glog.Infof("Private key exists, but public key does not")
		if err := os.Remove(privateKeyfile); err != nil {
			glog.Errorf("Failed to remove stale private key: %v", err)
		}
	}
	if err := ioutil.WriteFile(privateKeyfile, ssh.EncodePrivateKey(private), 0600); err != nil {
		return err
	}
	publicKeyBytes, err := ssh.EncodePublicKey(public)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(publicKeyfile+".tmp", publicKeyBytes, 0600); err != nil {
		return err
	}
	return os.Rename(publicKeyfile+".tmp", publicKeyfile)
}
