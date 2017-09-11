/*
Copyright 2016 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
)

type Suite string

const (
	E2E           Suite = "e2e"
	NodeE2E       Suite = "node e2e"
	FederationE2E Suite = "federation e2e"
)

var (
	mountImage   = imageutils.GetE2EImage(imageutils.Mounttest)
	busyboxImage = imageutils.GetBusyBoxImage()
)

var CurrentSuite Suite

// CommonImageWhiteList is the list of images used in common test. These images should be prepulled
// before a tests starts, so that the tests won't fail due image pulling flakes. Currently, this is
// only used by node e2e test.
// TODO(random-liu): Change the image puller pod to use similar mechanism.
var CommonImageWhiteList = sets.NewString(
	imageutils.GetBusyBoxImage(),
	imageutils.GetE2EImage(imageutils.EntrypointTester),
	imageutils.GetE2EImage(imageutils.Liveness),
	imageutils.GetE2EImage(imageutils.Mounttest),
	imageutils.GetE2EImage(imageutils.MounttestUser),
	imageutils.GetE2EImage(imageutils.Netexec),
	imageutils.GetE2EImage(imageutils.NginxSlim),
	imageutils.GetE2EImage(imageutils.ServeHostname),
	imageutils.GetE2EImage(imageutils.TestWebserver),
	imageutils.GetE2EImage(imageutils.Hostexec),
	"gcr.io/google_containers/volume-nfs:0.8",
	"gcr.io/google_containers/volume-gluster:0.2",
	"gcr.io/google_containers/e2e-net-amd64:1.0",
)

func svcByName(name string, port int) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeNodePort,
			Selector: map[string]string{
				"name": name,
			},
			Ports: []v1.ServicePort{{
				Port:       int32(port),
				TargetPort: intstr.FromInt(port),
			}},
		},
	}
}

func NewSVCByName(c clientset.Interface, ns, name string) error {
	const testPort = 9376
	_, err := c.Core().Services(ns).Create(svcByName(name, testPort))
	return err
}

// NewRCByName creates a replication controller with a selector by name of name.
func NewRCByName(c clientset.Interface, ns, name string, replicas int32, gracePeriod *int64) (*v1.ReplicationController, error) {
	By(fmt.Sprintf("creating replication controller %s", name))
	return c.Core().ReplicationControllers(ns).Create(framework.RcByNamePort(
		name, replicas, framework.ServeHostnameImage, 9376, v1.ProtocolTCP, map[string]string{}, gracePeriod))
}

func RestartNodes(c clientset.Interface, nodeNames []string) error {
	// List old boot IDs.
	oldBootIDs := make(map[string]string)
	for _, name := range nodeNames {
		node, err := c.Core().Nodes().Get(name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("error getting node info before reboot: %s", err)
		}
		oldBootIDs[name] = node.Status.NodeInfo.BootID
	}
	// Reboot the nodes.
	args := []string{
		"compute",
		fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
		"instances",
		"reset",
	}
	args = append(args, nodeNames...)
	args = append(args, fmt.Sprintf("--zone=%s", framework.TestContext.CloudConfig.Zone))
	stdout, stderr, err := framework.RunCmd("gcloud", args...)
	if err != nil {
		return fmt.Errorf("error restarting nodes: %s\nstdout: %s\nstderr: %s", err, stdout, stderr)
	}
	// Wait for their boot IDs to change.
	for _, name := range nodeNames {
		if err := wait.Poll(30*time.Second, 5*time.Minute, func() (bool, error) {
			node, err := c.Core().Nodes().Get(name, metav1.GetOptions{})
			if err != nil {
				return false, fmt.Errorf("error getting node info after reboot: %s", err)
			}
			return node.Status.NodeInfo.BootID != oldBootIDs[name], nil
		}); err != nil {
			return fmt.Errorf("error waiting for node %s boot ID to change: %s", name, err)
		}
	}
	return nil
}
