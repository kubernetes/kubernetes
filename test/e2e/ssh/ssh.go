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

package ssh

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
	"golang.org/x/crypto/ssh"
)

// NodeSSHHosts returns SSH-able host names for all nodes. It returns an error
// if it can't find an external IP for every node, though it still returns all
// hosts that it found in that case.
func NodeSSHHosts(c *client.Client) ([]string, error) {
	var hosts []string
	nodelist, err := c.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return hosts, fmt.Errorf("error getting nodes: %v", err)
	}
	for _, n := range nodelist.Items {
		for _, addr := range n.Status.Addresses {
			// Use the first external IP address we find on the node, and
			// use at most one per node.
			// TODO(mbforbes): Use the "preferred" address for the node, once
			// such a thing is defined (#2462).
			if addr.Type == api.NodeExternalIP {
				hosts = append(hosts, addr.Address+":22")
				break
			}
		}
	}

	// Error if any node didn't have an external IP.
	if len(hosts) != len(nodelist.Items) {
		return hosts, fmt.Errorf(
			"only found %d external IPs on nodes, but found %d nodes. Nodelist: %v",
			len(hosts), len(nodelist.Items), nodelist)
	}
	return hosts, nil
}

// SSH synchronously SSHs to a node running on provider and runs cmd. If there
// is no error performing the SSH, the stdout, stderr, and exit code are
// returned.
func SSH(cmd, host, provider string) (string, string, int, error) {
	// Get a signer for the provider.
	signer, err := GetSigner(provider)
	if err != nil {
		return "", "", 0, fmt.Errorf("error getting signer for provider %s: '%v'", provider, err)
	}

	user := os.Getenv("KUBE_SSH_USER")
	if user == "" {
		user = os.Getenv("USER")
	}
	remote := fmt.Sprintf("%s@%s", user, host)
	glog.Infof("Running `%s` on %s", cmd, remote)
	stdout, stderr, code, err := util.RunSSHCommand(cmd, user, host, signer)
	glog.Infof("[%s] stdout:    %q", remote, stdout)
	glog.Infof("[%s] stderr:    %q", remote, stderr)
	glog.Infof("[%s] exit code: %d", remote, code)
	glog.Infof("[%s] error:     %v", remote, err)
	return stdout, stderr, code, err
}

// getSigner returns an ssh.Signer for the provider ("gce", etc.) that can be
// used to SSH to their nodes.
// TODO(mbforbes): Make GetSigner unexported by doing the refactor proposed as a
//                 TODO in addon_update.go (it copy/pastes much of this code).
func GetSigner(provider string) (ssh.Signer, error) {
	// Get the directory in which SSH keys are located.
	keydir := filepath.Join(os.Getenv("HOME"), ".ssh")

	// Select the key itself to use. When implementing more providers here,
	// please also add them to any SSH tests that are disabled because of signer
	// support.
	keyfile := ""
	switch provider {
	case "gce", "gke":
		keyfile = "google_compute_engine"
	case "aws":
		keyfile = "kube_aws_rsa"
	default:
		return nil, fmt.Errorf("GetSigner(...) not implemented for %s", provider)
	}
	key := filepath.Join(keydir, keyfile)
	glog.Infof("Using SSH key: %s", key)

	return util.MakePrivateKeySignerFromFile(key)
}
