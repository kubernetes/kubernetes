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
	"context"
	"fmt"
	"os"
	"text/template"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
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

var (
	// non-Administrator Windows user used in tests. This is the Windows equivalent of the Linux non-root UID usage.
	nonAdminTestUserName = "ContainerUser"
	// non-root UID used in tests.
	nonRootTestUserID = int64(1000)
)

// CurrentSuite represents current test suite.
var CurrentSuite Suite

// PrePulledImages are a list of images used in e2e/common tests. These images should be prepulled
// before tests starts, so that the tests won't fail due image pulling flakes.
// Currently, this is only used by node e2e test.
// See also updateImageAllowList() in ../../e2e_node/image_list.go
// TODO(random-liu): Change the image puller pod to use similar mechanism.
var PrePulledImages = sets.NewString(
	imageutils.GetE2EImage(imageutils.Agnhost),
	imageutils.GetE2EImage(imageutils.BusyBox),
	imageutils.GetE2EImage(imageutils.IpcUtils),
	imageutils.GetE2EImage(imageutils.Nginx),
	imageutils.GetE2EImage(imageutils.Httpd),
	imageutils.GetE2EImage(imageutils.VolumeNFSServer),
	imageutils.GetE2EImage(imageutils.VolumeGlusterServer),
	imageutils.GetE2EImage(imageutils.NonRoot),
)

type testImagesStruct struct {
	AgnhostImage  string
	BusyBoxImage  string
	KittenImage   string
	NautilusImage string
	NginxImage    string
	NginxNewImage string
	HttpdImage    string
	HttpdNewImage string
	PauseImage    string
	RedisImage    string
}

var testImages testImagesStruct

func init() {
	testImages = testImagesStruct{
		imageutils.GetE2EImage(imageutils.Agnhost),
		imageutils.GetE2EImage(imageutils.BusyBox),
		imageutils.GetE2EImage(imageutils.Kitten),
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
		framework.Failf("Failed Parse the template: %v", err)
	}
	err = tmpl.Execute(contentWithImageName, testImages)
	if err != nil {
		framework.Failf("Failed executing template: %v", err)
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
	_, err := c.CoreV1().Services(ns).Create(context.TODO(), svcByName(name, testPort), metav1.CreateOptions{})
	return err
}

// NewRCByName creates a replication controller with a selector by name of name.
func NewRCByName(c clientset.Interface, ns, name string, replicas int32, gracePeriod *int64, containerArgs []string) (*v1.ReplicationController, error) {
	ginkgo.By(fmt.Sprintf("creating replication controller %s", name))

	if containerArgs == nil {
		containerArgs = []string{"serve-hostname"}
	}

	return c.CoreV1().ReplicationControllers(ns).Create(context.TODO(), rcByNamePort(
		name, replicas, framework.ServeHostnameImage, containerArgs, 9376, v1.ProtocolTCP, map[string]string{}, gracePeriod), metav1.CreateOptions{})
}

// RestartNodes restarts specific nodes.
func RestartNodes(c clientset.Interface, nodes []v1.Node) error {
	// Build mapping from zone to nodes in that zone.
	nodeNamesByZone := make(map[string][]string)
	for i := range nodes {
		node := &nodes[i]
		zone := framework.TestContext.CloudConfig.Zone
		if z, ok := node.Labels[v1.LabelFailureDomainBetaZone]; ok {
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
			newNode, err := c.CoreV1().Nodes().Get(context.TODO(), node.Name, metav1.GetOptions{})
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

// rcByNamePort returns a ReplicationController with specified name and port
func rcByNamePort(name string, replicas int32, image string, containerArgs []string, port int, protocol v1.Protocol,
	labels map[string]string, gracePeriod *int64) *v1.ReplicationController {

	return e2erc.ByNameContainer(name, replicas, labels, v1.Container{
		Name:  name,
		Image: image,
		Args:  containerArgs,
		Ports: []v1.ContainerPort{{ContainerPort: int32(port), Protocol: protocol}},
	}, gracePeriod)
}

// setPodNonRootUser configures the Pod to run as a non-root user.
// For Windows, it sets the RunAsUserName field to ContainerUser, and for Linux, it sets the RunAsUser field to 1000.
func setPodNonRootUser(pod *v1.Pod) {
	if framework.NodeOSDistroIs("windows") {
		pod.Spec.SecurityContext.WindowsOptions = &v1.WindowsSecurityContextOptions{RunAsUserName: &nonAdminTestUserName}
	} else {
		pod.Spec.SecurityContext.RunAsUser = &nonRootTestUserID
	}
}

// getFileModeRegex returns a file mode related regex which should be matched by the mounttest pods' output.
// If the given mask is nil, then the regex will contain the default OS file modes, which are 0644 for Linux and 0775 for Windows.
func getFileModeRegex(filePath string, mask *int32) string {
	var (
		linuxMask   int32
		windowsMask int32
	)
	if mask == nil {
		linuxMask = int32(0644)
		windowsMask = int32(0775)
	} else {
		linuxMask = *mask
		windowsMask = *mask
	}

	linuxOutput := fmt.Sprintf("mode of file \"%s\": %v", filePath, os.FileMode(linuxMask))
	windowsOutput := fmt.Sprintf("mode of Windows file \"%v\": %s", filePath, os.FileMode(windowsMask))

	return fmt.Sprintf("(%s|%s)", linuxOutput, windowsOutput)
}

// createMounts creates a v1.VolumeMount list with a single element.
func createMounts(volumeName, volumeMountPath string, readOnly bool) []v1.VolumeMount {
	return []v1.VolumeMount{
		{
			Name:      volumeName,
			MountPath: volumeMountPath,
			ReadOnly:  readOnly,
		},
	}
}
