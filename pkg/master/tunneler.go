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
	"io/ioutil"
	"math/rand"
	"net"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

type AddressFunc func() (addresses []string, err error)

type Tunneler interface {
	Run(AddressFunc)
	Stop()
	Dial(net, addr string) (net.Conn, error)
	SecondsSinceSync() int64
}

type SSHTunneler struct {
	SSHUser       string
	SSHKeyfile    string
	InstallSSHKey InstallSSHKey

	tunnels        *util.SSHTunnelList
	tunnelsLock    sync.Mutex
	lastSync       int64 // Seconds since Epoch
	lastSyncMetric prometheus.GaugeFunc
	clock          util.Clock

	getAddresses AddressFunc
	stopChan     chan struct{}
}

func NewSSHTunneler(sshUser string, sshKeyfile string, installSSHKey InstallSSHKey) Tunneler {
	return &SSHTunneler{
		SSHUser:       sshUser,
		SSHKeyfile:    sshKeyfile,
		InstallSSHKey: installSSHKey,

		clock: util.RealClock{},
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
		err := c.generateSSHKey(c.SSHUser, c.SSHKeyfile, publicKeyFile)
		if err != nil {
			glog.Errorf("Failed to create key pair: %v", err)
		}
	}
	c.tunnels = &util.SSHTunnelList{}
	c.setupSecureProxy(c.SSHUser, c.SSHKeyfile, publicKeyFile)
	c.lastSync = c.clock.Now().Unix()
}

// Stop gracefully shuts down the tunneler
func (c *SSHTunneler) Stop() {
	if c.stopChan != nil {
		close(c.stopChan)
		c.stopChan = nil
	}
}

func (c *SSHTunneler) Dial(net, addr string) (net.Conn, error) {
	// Only lock while picking a tunnel.
	tunnel, err := func() (util.SSHTunnelEntry, error) {
		c.tunnelsLock.Lock()
		defer c.tunnelsLock.Unlock()
		return c.tunnels.PickRandomTunnel()
	}()
	if err != nil {
		return nil, err
	}

	start := time.Now()
	id := rand.Int63() // So you can match begins/ends in the log.
	glog.V(3).Infof("[%x: %v] Dialing...", id, tunnel.Address)
	defer func() {
		glog.V(3).Infof("[%x: %v] Dialed in %v.", id, tunnel.Address, time.Now().Sub(start))
	}()
	return tunnel.Tunnel.Dial(net, addr)
}

func (c *SSHTunneler) SecondsSinceSync() int64 {
	now := c.clock.Now().Unix()
	then := atomic.LoadInt64(&c.lastSync)
	return now - then
}

func (c *SSHTunneler) needToReplaceTunnels(addrs []string) bool {
	c.tunnelsLock.Lock()
	defer c.tunnelsLock.Unlock()
	if c.tunnels == nil || c.tunnels.Len() != len(addrs) {
		return true
	}
	// TODO (cjcullen): This doesn't need to be n^2
	for ix := range addrs {
		if !c.tunnels.Has(addrs[ix]) {
			return true
		}
	}
	return false
}

func (c *SSHTunneler) replaceTunnels(user, keyfile string, newAddrs []string) error {
	glog.Infof("replacing tunnels. New addrs: %v", newAddrs)
	tunnels := util.MakeSSHTunnels(user, keyfile, newAddrs)
	if err := tunnels.Open(); err != nil {
		return err
	}
	c.tunnelsLock.Lock()
	defer c.tunnelsLock.Unlock()
	if c.tunnels != nil {
		c.tunnels.Close()
	}
	c.tunnels = tunnels
	atomic.StoreInt64(&c.lastSync, c.clock.Now().Unix())
	return nil
}

func (c *SSHTunneler) loadTunnels(user, keyfile string) error {
	addrs, err := c.getAddresses()
	if err != nil {
		return err
	}
	if !c.needToReplaceTunnels(addrs) {
		return nil
	}
	// TODO: This is going to unnecessarily close connections to unchanged nodes.
	// See comment about using Watch above.
	glog.Info("found different nodes. Need to replace tunnels")
	return c.replaceTunnels(user, keyfile, addrs)
}

func (c *SSHTunneler) refreshTunnels(user, keyfile string) error {
	addrs, err := c.getAddresses()
	if err != nil {
		return err
	}
	return c.replaceTunnels(user, keyfile, addrs)
}

func (c *SSHTunneler) setupSecureProxy(user, privateKeyfile, publicKeyfile string) {
	// Sync loop to ensure that the SSH key has been installed.
	go util.Until(func() {
		if c.InstallSSHKey == nil {
			glog.Error("Won't attempt to install ssh key: InstallSSHKey function is nil")
			return
		}
		key, err := util.ParsePublicKeyFromFile(publicKeyfile)
		if err != nil {
			glog.Errorf("Failed to load public key: %v", err)
			return
		}
		keyData, err := util.EncodeSSHKey(key)
		if err != nil {
			glog.Errorf("Failed to encode public key: %v", err)
			return
		}
		if err := c.InstallSSHKey(user, keyData); err != nil {
			glog.Errorf("Failed to install ssh key: %v", err)
		}
	}, 5*time.Minute, c.stopChan)
	// Sync loop for tunnels
	// TODO: switch this to watch.
	go util.Until(func() {
		if err := c.loadTunnels(user, privateKeyfile); err != nil {
			glog.Errorf("Failed to load SSH Tunnels: %v", err)
		}
		if c.tunnels != nil && c.tunnels.Len() != 0 {
			// Sleep for 10 seconds if we have some tunnels.
			// TODO (cjcullen): tunnels can lag behind actually existing nodes.
			time.Sleep(9 * time.Second)
		}
	}, 1*time.Second, c.stopChan)
	// Refresh loop for tunnels
	// TODO: could make this more controller-ish
	go util.Until(func() {
		time.Sleep(5 * time.Minute)
		if err := c.refreshTunnels(user, privateKeyfile); err != nil {
			glog.Errorf("Failed to refresh SSH Tunnels: %v", err)
		}
	}, 0*time.Second, c.stopChan)
}

func (c *SSHTunneler) generateSSHKey(user, privateKeyfile, publicKeyfile string) error {
	// TODO: user is not used. Consider removing it as an input to the function.
	private, public, err := util.GenerateKey(2048)
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
	if err := ioutil.WriteFile(privateKeyfile, util.EncodePrivateKey(private), 0600); err != nil {
		return err
	}
	publicKeyBytes, err := util.EncodePublicKey(public)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(publicKeyfile+".tmp", publicKeyBytes, 0600); err != nil {
		return err
	}
	return os.Rename(publicKeyfile+".tmp", publicKeyfile)
}
