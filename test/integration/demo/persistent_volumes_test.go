/*
Copyright 2024 The Kubernetes Authors.

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

package demo

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"
)

func TestPersistentVolumeProvisionMultiPVCsWIP(t *testing.T) {
	klog.Info(t.Name())
	// out := filepath.Join(os.Getenv("ARTIFACTS"), "test.txt")
	// os.WriteFile(out, []byte("test"), 0644)
	// t.Fatal(t.Name())
	t.Error("WIP")
}

func dumpAll(t *testing.T, client clientset.Interface, namespace string) {
	out := os.Getenv("ARTIFACTS")
	os.MkdirAll(filepath.Dir(out), 0755)
	pvs, _ := client.CoreV1().PersistentVolumes().List(context.TODO(), metav1.ListOptions{})
	data, _ := yaml.Marshal(pvs)
	os.WriteFile(filepath.Join(out, t.Name()+"pvs.yaml"), data, 0644)
	pvcs, _ := client.CoreV1().PersistentVolumeClaims(namespace).List(context.TODO(), metav1.ListOptions{})
	data, _ = yaml.Marshal(pvcs)
	os.WriteFile(filepath.Join(out, t.Name()+"pvcs.yaml"), data, 0644)
	events, _ := client.EventsV1().Events(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{})
	data, _ = yaml.Marshal(events)
	os.WriteFile(filepath.Join(out, t.Name()+"events.yaml"), data, 0644)
	klog.Info("dump")
}
