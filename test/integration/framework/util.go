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

// TODO: This file can potentially be moved to a common place used by both e2e and integration tests.

package framework

import (
	"io/ioutil"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api/testapi"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubectl"
)

const (
	// When these values are updated, also update cmd/kubelet/app/options/options.go
	// A copy of these values exist in e2e/framework/util.go.
	currentPodInfraContainerImageName    = "gcr.io/google_containers/pause"
	currentPodInfraContainerImageVersion = "3.0"
)

// GetServerArchitecture fetches the architecture of the cluster's apiserver.
func GetServerArchitecture(c clientset.Interface) string {
	arch := ""
	sVer, err := c.Discovery().ServerVersion()
	if err != nil || sVer.Platform == "" {
		// If we failed to get the server version for some reason, default to amd64.
		arch = "amd64"
	} else {
		// Split the platform string into OS and Arch separately.
		// The platform string may for example be "linux/amd64", "linux/arm" or "windows/amd64".
		osArchArray := strings.Split(sVer.Platform, "/")
		arch = osArchArray[1]
	}
	return arch
}

// GetPauseImageName fetches the pause image name for the same architecture as the apiserver.
func GetPauseImageName(c clientset.Interface) string {
	return currentPodInfraContainerImageName + "-" + GetServerArchitecture(c) + ":" + currentPodInfraContainerImageVersion
}

func CreateTestingNamespace(baseName string, apiserver *httptest.Server, t *testing.T) *v1.Namespace {
	// TODO: Create a namespace with a given basename.
	// Currently we neither create the namespace nor delete all of its contents at the end.
	// But as long as tests are not using the same namespaces, this should work fine.
	return &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			// TODO: Once we start creating namespaces, switch to GenerateName.
			Name: baseName,
		},
	}
}

func DeleteTestingNamespace(ns *v1.Namespace, apiserver *httptest.Server, t *testing.T) {
	// TODO: Remove all resources from a given namespace once we implement CreateTestingNamespace.
}

// RCFromManifest reads a .json file and returns the rc in it.
func RCFromManifest(fileName string) *v1.ReplicationController {
	data, err := ioutil.ReadFile(fileName)
	if err != nil {
		glog.Fatalf("Unexpected error reading rc manifest %v", err)
	}
	var controller v1.ReplicationController
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &controller); err != nil {
		glog.Fatalf("Unexpected error reading rc manifest %v", err)
	}
	return &controller
}

// StopRC stops the rc via kubectl's stop library
func StopRC(rc *v1.ReplicationController, clientset internalclientset.Interface) error {
	reaper, err := kubectl.ReaperFor(api.Kind("ReplicationController"), clientset)
	if err != nil || reaper == nil {
		return err
	}
	err = reaper.Stop(rc.Namespace, rc.Name, 0, nil)
	if err != nil {
		return err
	}
	return nil
}

// ScaleRC scales the given rc to the given replicas.
func ScaleRC(name, ns string, replicas int32, clientset internalclientset.Interface) (*api.ReplicationController, error) {
	scaler, err := kubectl.ScalerFor(api.Kind("ReplicationController"), clientset)
	if err != nil {
		return nil, err
	}
	retry := &kubectl.RetryParams{Interval: 50 * time.Millisecond, Timeout: DefaultTimeout}
	waitForReplicas := &kubectl.RetryParams{Interval: 50 * time.Millisecond, Timeout: DefaultTimeout}
	err = scaler.Scale(ns, name, uint(replicas), nil, retry, waitForReplicas)
	if err != nil {
		return nil, err
	}
	scaled, err := clientset.Core().ReplicationControllers(ns).Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return scaled, nil
}
