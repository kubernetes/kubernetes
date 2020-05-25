// +build linux

/*
Copyright 2014 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"
	"io"
	"io/ioutil"
	"k8s.io/apimachinery/pkg/runtime"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/securitycontext"
	"os"
	"path/filepath"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/kubelet/config"
)

func Test_makePodSourceConfig_StaticPodPath(t *testing.T) {
	nodeName := "makepodsourceconfig"
	args := []struct {
		name                    string
		kubeCfg                 *kubeletconfiginternal.KubeletConfiguration
		kubeDeps                *Dependencies
		staticPods              []*v1.Pod
		nodeName                string
		bootstrapCheckpointPath string
		want                    *config.PodConfig
	}{
		{
			name: "test_makePodSourceConfig_StaticPodPath_multi_pod",
			kubeCfg: &kubeletconfiginternal.KubeletConfiguration{
				StaticPodPath:      "test",
				FileCheckFrequency: metav1.Duration{Duration: 15 * time.Second},
			},
			kubeDeps: &Dependencies{
				KubeClient: nil,
			},
			staticPods: []*v1.Pod{
				{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Pod",
						APIVersion: "",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test1",
						UID:       "12345",
						Namespace: "mynamespace1",
					},
					Spec: v1.PodSpec{
						Containers:      []v1.Container{{Name: "image", Image: "test/image", SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults()}},
						SecurityContext: &v1.PodSecurityContext{},
						SchedulerName:   api.DefaultSchedulerName,
					},
				}, {
					TypeMeta: metav1.TypeMeta{
						Kind:       "Pod",
						APIVersion: "",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test2",
						UID:       "54321",
						Namespace: "mynamespace2",
					},
					Spec: v1.PodSpec{
						Containers:      []v1.Container{{Name: "image", Image: "test/image", SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults()}},
						SecurityContext: &v1.PodSecurityContext{},
						SchedulerName:   api.DefaultSchedulerName,
					},
				},
			},
			nodeName: nodeName,
		},
		{
			name: "test_makePodSourceConfig_StaticPodPath_one_pod",
			kubeCfg: &kubeletconfiginternal.KubeletConfiguration{
				StaticPodPath:      "test",
				FileCheckFrequency: metav1.Duration{Duration: 15 * time.Second},
			},
			kubeDeps: &Dependencies{
				KubeClient: nil,
			},
			staticPods: []*v1.Pod{
				{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Pod",
						APIVersion: "",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test3",
						UID:       "67890",
						Namespace: "mynamespace3",
					},
					Spec: v1.PodSpec{
						Containers:      []v1.Container{{Name: "image", Image: "test/image", ImagePullPolicy: v1.PullAlways, TerminationMessagePolicy: v1.TerminationMessageReadFile}},
						SecurityContext: &v1.PodSecurityContext{},
						SchedulerName:   api.DefaultSchedulerName,
					},
				},
			},
			nodeName: nodeName,
		},
		{
			name: "test_makePodSourceConfig_StaticPodPath_without_pod",
			kubeCfg: &kubeletconfiginternal.KubeletConfiguration{
				StaticPodPath:      "test",
				FileCheckFrequency: metav1.Duration{Duration: 15 * time.Second},
			},
			kubeDeps: &Dependencies{
				KubeClient: nil,
			},
			staticPods: []*v1.Pod{},
			nodeName:   nodeName,
		},
	}

	for _, tt := range args {
		t.Run(tt.name, func(t *testing.T) {
			dirName, err := mkTempDir(tt.kubeCfg.StaticPodPath)
			if err != nil {
				t.Fatalf("unable to create temp dir: %v", err)
			}
			defer removeAll(dirName, t)
			tt.kubeCfg.StaticPodPath = dirName
			for _, pod := range tt.staticPods {
				_ = writeToFile(dirName, pod, t)
			}

			updates, err := makePodSourceConfig(tt.kubeCfg, tt.kubeDeps, types.NodeName(tt.nodeName), tt.bootstrapCheckpointPath)
			if err != nil {
				t.Fatalf("makePodSourceConfig() error = %v", err)
			}

			select {
			case got := <-updates.Updates():
				actualPods := got.Pods
				expectPods := tt.staticPods
				actualCount := len(actualPods)
				expectCount := len(expectPods)
				if actualCount != expectCount {
					t.Fatalf("makePodSourceConfig() fail, actual update Pods count: %d expect count: %d", actualCount, expectCount)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("%s: Expected update, timeout instead", tt.name)
			}
		})
	}
}

func writeToFile(dir string, pod *v1.Pod, t *testing.T) string {
	fileContents, err := runtime.Encode(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), pod)
	if err != nil {
		t.Fatalf("error in encoding the pod: %v", err)
	}

	fileName := filepath.Join(dir, fmt.Sprintf("%s.json", pod.Name))
	if err := writeFile(fileName, []byte(fileContents)); err != nil {
		t.Fatalf("unable to write test file %#v", err)
	}
	return fileName
}

func writeFile(filename string, data []byte) error {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	n, err := f.Write(data)
	if err == nil && n < len(data) {
		err = io.ErrShortWrite
	}
	if err1 := f.Close(); err == nil {
		err = err1
	}
	return err
}

func mkTempDir(prefix string) (string, error) {
	return ioutil.TempDir(os.TempDir(), prefix)
}

func removeAll(dir string, t *testing.T) {
	if err := os.RemoveAll(dir); err != nil {
		t.Fatalf("unable to remove dir %s: %v", dir, err)
	}
}
