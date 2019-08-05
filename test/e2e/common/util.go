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
	"bytes"
	"fmt"
	"text/template"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

// Suite represents test suite.
type Suite string

const (
	// E2E represents a test suite for e2e.
	E2E Suite = "e2e"
	// NodeE2E represents a test suite for node e2e.
	NodeE2E Suite = "node e2e"
)

// CurrentSuite represents current test suite.
var CurrentSuite Suite

// CommonImageWhiteList is the list of images used in common test. These images should be prepulled
// before a tests starts, so that the tests won't fail due image pulling flakes. Currently, this is
// only used by node e2e test.
// TODO(random-liu): Change the image puller pod to use similar mechanism.
var CommonImageWhiteList = sets.NewString(
	imageutils.GetE2EImage(imageutils.Agnhost),
	imageutils.GetE2EImage(imageutils.BusyBox),
	imageutils.GetE2EImage(imageutils.IpcUtils),
	imageutils.GetE2EImage(imageutils.Mounttest),
	imageutils.GetE2EImage(imageutils.MounttestUser),
	imageutils.GetE2EImage(imageutils.Nginx),
	imageutils.GetE2EImage(imageutils.Httpd),
	imageutils.GetE2EImage(imageutils.TestWebserver),
	imageutils.GetE2EImage(imageutils.VolumeNFSServer),
	imageutils.GetE2EImage(imageutils.VolumeGlusterServer),
)

type testImagesStruct struct {
	AgnhostImage    string
	BusyBoxImage    string
	GBFrontendImage string
	KittenImage     string
	MounttestImage  string
	NautilusImage   string
	NginxImage      string
	NginxNewImage   string
	HttpdImage      string
	HttpdNewImage   string
	PauseImage      string
	RedisImage      string
}

var testImages testImagesStruct

func init() {
	testImages = testImagesStruct{
		imageutils.GetE2EImage(imageutils.Agnhost),
		imageutils.GetE2EImage(imageutils.BusyBox),
		imageutils.GetE2EImage(imageutils.GBFrontend),
		imageutils.GetE2EImage(imageutils.Kitten),
		imageutils.GetE2EImage(imageutils.Mounttest),
		imageutils.GetE2EImage(imageutils.Nautilus),
		imageutils.GetE2EImage(imageutils.Nginx),
		imageutils.GetE2EImage(imageutils.NginxNew),
		imageutils.GetE2EImage(imageutils.Httpd),
		imageutils.GetE2EImage(imageutils.HttpdNew),
		imageutils.GetE2EImage(imageutils.Pause),
		imageutils.GetE2EImage(imageutils.Redis),
	}
}

// SubstituteImageName replaces image name in content.
func SubstituteImageName(content string) string {
	contentWithImageName := new(bytes.Buffer)
	tmpl, err := template.New("imagemanifest").Parse(content)
	if err != nil {
		e2elog.Failf("Failed Parse the template: %v", err)
	}
	err = tmpl.Execute(contentWithImageName, testImages)
	if err != nil {
		e2elog.Failf("Failed executing template: %v", err)
	}
	return contentWithImageName.String()
}

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

// NewSVCByName creates a service by name.
func NewSVCByName(c clientset.Interface, ns, name string) error {
	const testPort = 9376
	_, err := c.CoreV1().Services(ns).Create(svcByName(name, testPort))
	return err
}

// NewRCByName creates a replication controller with a selector by name of name.
func NewRCByName(c clientset.Interface, ns, name string, replicas int32, gracePeriod *int64, containerArgs []string) (*v1.ReplicationController, error) {
	ginkgo.By(fmt.Sprintf("creating replication controller %s", name))

	if containerArgs == nil {
		containerArgs = []string{"serve-hostname"}
	}

	return c.CoreV1().ReplicationControllers(ns).Create(framework.RcByNamePort(
		name, replicas, framework.ServeHostnameImage, containerArgs, 9376, v1.ProtocolTCP, map[string]string{}, gracePeriod))
}

// RestartNodes restarts specific nodes.
func RestartNodes(c clientset.Interface, nodes []v1.Node) error {
	// Build mapping from zone to nodes in that zone.
	nodeNamesByZone := make(map[string][]string)
	for i := range nodes {
		node := &nodes[i]
		zone := framework.TestContext.CloudConfig.Zone
		if z, ok := node.Labels[v1.LabelZoneFailureDomain]; ok {
			zone = z
		}
		nodeNamesByZone[zone] = append(nodeNamesByZone[zone], node.Name)
	}

	// Reboot the nodes.
	for zone, nodeNames := range nodeNamesByZone {
		args := []string{
			"compute",
			fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
			"instances",
			"reset",
		}
		args = append(args, nodeNames...)
		args = append(args, fmt.Sprintf("--zone=%s", zone))
		stdout, stderr, err := framework.RunCmd("gcloud", args...)
		if err != nil {
			return fmt.Errorf("error restarting nodes: %s\nstdout: %s\nstderr: %s", err, stdout, stderr)
		}
	}

	// Wait for their boot IDs to change.
	for i := range nodes {
		node := &nodes[i]
		if err := wait.Poll(30*time.Second, framework.RestartNodeReadyAgainTimeout, func() (bool, error) {
			newNode, err := c.CoreV1().Nodes().Get(node.Name, metav1.GetOptions{})
			if err != nil {
				return false, fmt.Errorf("error getting node info after reboot: %s", err)
			}
			return node.Status.NodeInfo.BootID != newNode.Status.NodeInfo.BootID, nil
		}); err != nil {
			return fmt.Errorf("error waiting for node %s boot ID to change: %s", node.Name, err)
		}
	}
	return nil
}
