/*
Copyright 2020 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"io/ioutil"
	testutils "k8s.io/kubernetes/test/utils"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	// SRIOVDevicePluginCMYAML is the path of the config map to configure the sriov device plugin.
	SRIOVDevicePluginCMYAML = "test/e2e_node/testing-manifests/sriovdp-cm.yaml"
	// SRIOVDevicePluginDSYAML is the path of the daemonset template of the sriov device plugin. // TODO: Parametrize it by making it a feature in TestFramework.
	SRIOVDevicePluginDSYAML = "test/e2e_node/testing-manifests/sriovdp-ds.yaml"
	// SRIOVDevicePluginSAYAML is the path of the service account needed by the sriov device plugin to run.
	SRIOVDevicePluginSAYAML = "test/e2e_node/testing-manifests/sriovdp-sa.yaml"
	// SRIOVDevicePluginName is the name of the device plugin pod
	SRIOVDevicePluginName = "sriov-device-plugin"
)

const (
	minSriovResource = 7 // This is the min number of SRIOV VFs needed on the system under test.
)

func countSRIOVDevices() (int, error) {
	outData, err := exec.Command("/bin/sh", "-c", "ls /sys/bus/pci/devices/*/physfn | wc -w").Output()
	if err != nil {
		return -1, err
	}
	return strconv.Atoi(strings.TrimSpace(string(outData)))
}

func detectSRIOVDevices() int {
	devCount, err := countSRIOVDevices()
	framework.ExpectNoError(err)
	return devCount
}

// getSRIOVDevicePluginPod returns the Device Plugin pod for sriov resources in e2e tests.
func getSRIOVDevicePluginPod() *v1.Pod {
	data, err := e2etestfiles.Read(SRIOVDevicePluginDSYAML)
	if err != nil {
		framework.Fail(err.Error())
	}

	ds := readDaemonSetV1OrDie(data)
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      SRIOVDevicePluginName,
			Namespace: metav1.NamespaceSystem,
		},

		Spec: ds.Spec.Template.Spec,
	}

	return p
}

func readConfigMapV1OrDie(objBytes []byte) *v1.ConfigMap {
	v1.AddToScheme(appsScheme)
	requiredObj, err := runtime.Decode(appsCodecs.UniversalDecoder(v1.SchemeGroupVersion), objBytes)
	if err != nil {
		panic(err)
	}
	return requiredObj.(*v1.ConfigMap)
}

func readServiceAccountV1OrDie(objBytes []byte) *v1.ServiceAccount {
	v1.AddToScheme(appsScheme)
	requiredObj, err := runtime.Decode(appsCodecs.UniversalDecoder(v1.SchemeGroupVersion), objBytes)
	if err != nil {
		panic(err)
	}
	return requiredObj.(*v1.ServiceAccount)
}

func findSRIOVResource(node *v1.Node) (string, int64) {
	framework.Logf("Node status allocatable: %v", node.Status.Allocatable)
	re := regexp.MustCompile(`^intel.com/.*sriov.*`)
	for key, val := range node.Status.Allocatable {
		resource := string(key)
		if re.MatchString(resource) {
			v := val.Value()
			if v > 0 {
				return resource, v
			}
		}
	}
	return "", 0
}

func getSRIOVDevicePluginConfigMap(cmFile string) *v1.ConfigMap {
	data, err := e2etestfiles.Read(SRIOVDevicePluginCMYAML)
	if err != nil {
		framework.Fail(err.Error())
	}

	// the SRIOVDP configuration is hw-dependent, so we allow per-test-host customization.
	framework.Logf("host-local SRIOV Device Plugin Config Map %q", cmFile)
	if cmFile != "" {
		data, err = ioutil.ReadFile(cmFile)
		if err != nil {
			framework.Failf("unable to load the SRIOV Device Plugin ConfigMap: %v", err)
		}
	} else {
		framework.Logf("Using built-in SRIOV Device Plugin Config Map")
	}

	return readConfigMapV1OrDie(data)
}

type sriovData struct {
	configMap      *v1.ConfigMap
	serviceAccount *v1.ServiceAccount
	pod            *v1.Pod

	resourceName   string
	resourceAmount int64
}

func setupSRIOVConfigOrFail(f *framework.Framework, configMap *v1.ConfigMap) *sriovData {
	var err error

	ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", metav1.NamespaceSystem, configMap.Name))
	if _, err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	data, err := e2etestfiles.Read(SRIOVDevicePluginSAYAML)
	if err != nil {
		framework.Fail(err.Error())
	}
	serviceAccount := readServiceAccountV1OrDie(data)
	ginkgo.By(fmt.Sprintf("Creating serviceAccount %v/%v", metav1.NamespaceSystem, serviceAccount.Name))
	if _, err = f.ClientSet.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(context.TODO(), serviceAccount, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test serviceAccount %s: %v", serviceAccount.Name, err)
	}

	e2enode.WaitForNodeToBeReady(f.ClientSet, framework.TestContext.NodeName, 5*time.Minute)

	dp := getSRIOVDevicePluginPod()
	dp.Spec.NodeName = framework.TestContext.NodeName

	ginkgo.By("Create SRIOV device plugin pod")
	dpPod, err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Create(context.TODO(), dp, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	if err = e2epod.WaitForPodCondition(f.ClientSet, metav1.NamespaceSystem, dp.Name, "Ready", 120*time.Second, testutils.PodRunningReady); err != nil {
		framework.Logf("SRIOV Pod %v took too long to enter running/ready: %v", dp.Name, err)
	}
	framework.ExpectNoError(err)

	sriovResourceName := ""
	var sriovResourceAmount int64
	ginkgo.By("Waiting for devices to become available on the local node")
	gomega.Eventually(func() bool {
		node := getLocalNode(f)
		sriovResourceName, sriovResourceAmount = findSRIOVResource(node)
		return sriovResourceAmount > minSriovResource
	}, 2*time.Minute, framework.Poll).Should(gomega.BeTrue())
	framework.Logf("Successfully created device plugin pod, detected %d SRIOV allocatable devices %q", sriovResourceAmount, sriovResourceName)

	return &sriovData{
		configMap:      configMap,
		serviceAccount: serviceAccount,
		pod:            dpPod,
		resourceName:   sriovResourceName,
		resourceAmount: sriovResourceAmount,
	}
}

func teardownSRIOVConfigOrFail(f *framework.Framework, sd *sriovData) {
	var err error
	gp := int64(0)
	deleteOptions := metav1.DeleteOptions{
		GracePeriodSeconds: &gp,
	}

	ginkgo.By("Delete SRIOV device plugin pod %s/%s")
	err = f.ClientSet.CoreV1().Pods(sd.pod.Namespace).Delete(context.TODO(), sd.pod.Name, deleteOptions)
	framework.ExpectNoError(err)
	waitForContainerRemoval(sd.pod.Spec.Containers[0].Name, sd.pod.Name, sd.pod.Namespace)

	ginkgo.By(fmt.Sprintf("Deleting configMap %v/%v", metav1.NamespaceSystem, sd.configMap.Name))
	err = f.ClientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Delete(context.TODO(), sd.configMap.Name, deleteOptions)
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Deleting serviceAccount %v/%v", metav1.NamespaceSystem, sd.serviceAccount.Name))
	err = f.ClientSet.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Delete(context.TODO(), sd.serviceAccount.Name, deleteOptions)
	framework.ExpectNoError(err)
}
