/*
Copyright 2017 The Kubernetes Authors.

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

package digitalocean

import (
	"fmt"
	"time"

	"github.com/digitalocean/godo"
	"github.com/digitalocean/godo/context"
	"github.com/golang/glog"
	"golang.org/x/oauth2"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
)

// "k8s.io/kubernetes/pkg/api/v1"

const (
	volumeAttachmentStatusConsecutiveErrorLimit = 10
	volumeAttachmentStatusInitialDelay          = 10 * time.Second
	volumeAttachmentStatusFactor                = 1.2
	volumeAttachmentStatusSteps                 = 20
)

// DOManager communicates with the DO API
type doManager struct {
	config  *doManagerConfig
	client  *godo.Client
	context context.Context
}

// DOManagerConfig keeps Digital Ocean client configuration
type doManagerConfig struct {
	token  string
	region string
}

// TokenSource represents and oauth2 token source
type TokenSource struct {
	AccessToken string
}

// Token returns an oauth2 token
func (t *TokenSource) Token() (*oauth2.Token, error) {
	token := &oauth2.Token{
		AccessToken: t.AccessToken,
	}
	return token, nil
}

// NewDOManager returns a Digitial Ocean manager
func newDOManager(config *doManagerConfig) (*doManager, error) {
	do := &doManager{
		config: config,
	}
	// generate client and test retrieving account info
	_, err := do.getAccount()
	if err != nil {
		return nil, err
	}
	return do, nil
}

// refreshDOClient will update the Digital Ocean client if it is not
// already cached
func (m *doManager) refreshDOClient() error {
	if m.context != nil && m.client != nil {
		return nil
	}
	if m.config.token == "" {
		return fmt.Errorf("DOManager needs to be initialized with a token")
	}
	if m.config.token == "" {
		return fmt.Errorf("DOManager needs to be initialized with the cluster region")
	}

	tokenSource := &TokenSource{
		AccessToken: m.config.token,
	}
	m.context = context.Background()
	oauthClient := oauth2.NewClient(m.context, tokenSource)
	m.client = godo.NewClient(oauthClient)

	return nil
}

// removeDOClient will remove the cached Digital Ocean client
func (m *doManager) removeDOClient() {
	m.context = nil
	m.client = nil
}

// getAccount returns the token related account
func (m *doManager) getAccount() (*godo.Account, error) {
	m.refreshDOClient()
	account, _, err := m.client.Account.Get(m.context)
	if err != nil {
		m.removeDOClient()
		return nil, err
	}
	return account, nil
}

// GetDroplet retrieves the droplet by ID
func (m *doManager) GetDroplet(dropletID int) (*godo.Droplet, error) {
	m.refreshDOClient()
	droplet, _, err := m.client.Droplets.Get(m.context, dropletID)
	if err != nil {
		m.removeDOClient()
		return nil, err
	}
	return droplet, err
}

// DropletList return all droplets
func (m *doManager) DropletList() ([]godo.Droplet, error) {
	list := []godo.Droplet{}
	opt := &godo.ListOptions{}
	for {
		droplets, resp, err := m.client.Droplets.List(m.context, opt)
		if err != nil {
			m.removeDOClient()
			return nil, err
		}

		for _, d := range droplets {
			list = append(list, d)
		}
		if resp.Links == nil || resp.Links.IsLastPage() {
			break
		}
		page, err := resp.Links.CurrentPage()
		if err != nil {
			return nil, err
		}
		opt.Page = page + 1
	}
	return list, nil
}

func (m *doManager) GetVolume(volumeID string) (*godo.Volume, error) {
	vol, _, err := m.client.Storage.GetVolume(m.context, volumeID)
	if err != nil {
		m.removeDOClient()
		return nil, err
	}
	return vol, nil
}

// DeleteVolume deletes a Digital Ocean volume
func (m *doManager) DeleteVolume(volumeID string) error {
	m.refreshDOClient()
	_, err := m.client.Storage.DeleteVolume(m.context, volumeID)
	if err != nil {
		m.removeDOClient()
		return err
	}
	return nil
}

// CreateVolume creates a Digital Ocean volume from a provisioner and returns the ID
func (m *doManager) CreateVolume(name, description string, sizeGB int) (string, error) {
	m.refreshDOClient()

	req := &godo.VolumeCreateRequest{
		Region:        m.config.region,
		Name:          name,
		Description:   description,
		SizeGigaBytes: int64(sizeGB),
	}

	vol, _, err := m.client.Storage.CreateVolume(m.context, req)
	if err != nil {
		m.removeDOClient()
		return "", err
	}

	return vol.ID, nil
}

// AttachVolume attaches volume to given droplet
// returns the path the disk is being attached to
func (m *doManager) AttachVolume(volumeID string, dropletID int) (string, error) {
	vol, err := m.GetVolume(volumeID)
	if err != nil {
		m.removeDOClient()
		return "", err
	}

	needAttach := true
	for id := range vol.DropletIDs {
		if id == dropletID {
			needAttach = false
		}
	}

	if needAttach {
		action, _, err := m.client.StorageActions.Attach(m.context, volumeID, dropletID)
		if err != nil {
			return "", err
		}
		glog.V(2).Infof("AttachVolume volume=%q droplet=%q requested")
		err = m.WaitForVolumeAttach(volumeID, action.ID)
		if err != nil {
			return "", err
		}
	}
	return "/dev/disk/by-id/scsi-0DO_Volume_" + vol.Name, nil
}

// DetachDisk detaches a disk to given droplet
func (m *doManager) DetachVolume(volumeID string, dropletID int) error {
	_, _, err := m.client.StorageActions.DetachByDropletID(m.context, volumeID, dropletID)
	return err
}

// DetachDisk detaches a disk to given droplet
func (m *doManager) DetachVolumeByName(volumeName string, dropletID int) error {

	droplet, err := m.GetDroplet(dropletID)
	if err != nil {
		return err
	}

	for _, volumeID := range droplet.VolumeIDs {
		vol, err := m.GetVolume(volumeID)
		if err != nil {
			return err
		}
		if vol.Name == volumeName {
			return m.DetachVolume(volumeID, dropletID)
		}
	}
	return fmt.Errorf("Detach failed. Couldn't find Digital Ocean volume named %q", volumeName)
}

func (m *doManager) GetVolumeAction(volumeID string, actionID int) (*godo.Action, error) {
	action, _, err := m.client.StorageActions.Get(m.context, volumeID, actionID)
	if err != nil {
		m.removeDOClient()
		return nil, err
	}
	return action, nil
}

func (m *doManager) WaitForVolumeAttach(volumeID string, actionID int) error {
	backoff := wait.Backoff{
		Duration: volumeAttachmentStatusInitialDelay,
		Factor:   volumeAttachmentStatusFactor,
		Steps:    volumeAttachmentStatusSteps,
	}

	errorCount := 0
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		action, e := m.GetVolumeAction(volumeID, actionID)
		if e != nil {
			errorCount++
			if errorCount > volumeAttachmentStatusConsecutiveErrorLimit {
				return false, e
			}
			glog.Warningf("Ignoring error from get volume action; will retry: %q", e)
			return false, nil
		}
		errorCount = 0

		if action.Status != godo.ActionCompleted {
			glog.V(2).Infof("Waiting for volume %q state: actual=%s, desired=%s",
				volumeID, action.Status, godo.ActionCompleted)
			return false, nil
		}

		return true, nil
	})
	return err
}

// DisksAreAttached checks if a list of volumes are attached to the node with the specified NodeName
func (m *doManager) DisksAreAttached(volumeIDs []string, dropletID int) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, volumeID := range volumeIDs {
		attached[volumeID] = false
	}
	droplet, err := m.GetDroplet(dropletID)
	if err != nil {
		return attached, err
	}

	for _, attachedID := range droplet.VolumeIDs {
		for _, expectedID := range volumeIDs {
			if attachedID == expectedID {
				attached[attachedID] = true
			}
		}
	}

	return attached, nil
}

func (m *doManager) FindDropletForNode(node *v1.Node) (*godo.Droplet, error) {

	// try to find droplet with same name as the kubernetes node
	droplets, err := m.DropletList()
	if err != nil {
		return nil, err
	}

	for _, droplet := range droplets {
		if droplet.Name == node.Name {
			return &droplet, nil
		}
	}

	// if not found, look for other kubernetes properties
	// Internal IP seems to be our safest bet when names doesn't match
	for _, droplet := range droplets {
		for _, address := range node.Status.Addresses {
			if address.Type == v1.NodeInternalIP {
				ip, err := droplet.PrivateIPv4()
				if err != nil {
					return nil, err
				}
				if ip == address.Address {
					return &droplet, nil
				}
			}
		}
	}

	return nil, fmt.Errorf("Couldn't match droplet name to node name, nor droplet private ip to node internal ip")
}
