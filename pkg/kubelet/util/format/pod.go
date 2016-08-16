/*
Copyright 2015 The Kubernetes Authors.

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

package format

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"
	"time"

	//"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

type podHandler func(*api.Pod) string

// PodContext wraps the Pod object and ensures only a subset of attributes
// can be read via the
type PodContext struct {
	pod *api.Pod
}

//Accessors for PodContext

func (pc PodContext) PodIP() string {
	return pc.pod.Status.PodIP
}

func (pc PodContext) Name() string {
	return pc.pod.Name
}

func (pc PodContext) Labels() string {
	labels := make([]string, 0, len(pc.pod.Labels))
	for label, value := range pc.pod.Labels {
		labels = append(labels, label+"="+value)
	}
	return strings.Join(labels, ",")
}

// MapContext is an object that contains all the data needed to render
// the templates in ConfigMap
type MapContext struct {
	pod    *api.Pod
	client clientset.Interface
}

func NewMapContext(pod *api.Pod, client clientset.Interface) MapContext {
	return MapContext{pod: pod, client: client}
}

func (mc MapContext) Pod() PodContext {
	return PodContext{pod: mc.pod}
}

// GetConfigMapData finds config maps by name and returns its Data

func (mc MapContext) ConfigMap(name string) (map[string]string, error) {
	configmap, err := mc.client.Core().ConfigMaps(mc.pod.Namespace).Get(name)
	if err != nil {
		return nil, err
	}

	return configmap.Data, nil
}

// func (mc *MapContext) ConfigMap () ()

// Pod returns a string reprenetating a pod in a human readable format,
// with pod UID as part of the string.
func Pod(pod *api.Pod) string {
	// Use underscore as the delimiter because it is not allowed in pod name
	// (DNS subdomain format), while allowed in the container name format.
	return fmt.Sprintf("%s_%s(%s)", pod.Name, pod.Namespace, pod.UID)
}

// PodWithDeletionTimestamp is the same as Pod. In addition, it prints the
// deletion timestamp of the pod if it's not nil.
func PodWithDeletionTimestamp(pod *api.Pod) string {
	var deletionTimestamp string
	if pod.DeletionTimestamp != nil {
		deletionTimestamp = ":DeletionTimestamp=" + pod.DeletionTimestamp.UTC().Format(time.RFC3339)
	}
	return Pod(pod) + deletionTimestamp
}

// Pods returns a string representating a list of pods in a human
// readable format.
func Pods(pods []*api.Pod) string {
	return aggregatePods(pods, Pod)
}

// PodsWithDeletiontimestamps is the same as Pods. In addition, it prints the
// deletion timestamps of the pods if they are not nil.
func PodsWithDeletiontimestamps(pods []*api.Pod) string {
	return aggregatePods(pods, PodWithDeletionTimestamp)
}

func aggregatePods(pods []*api.Pod, handler podHandler) string {
	podStrings := make([]string, 0, len(pods))
	for _, pod := range pods {
		podStrings = append(podStrings, handler(pod))
	}
	return fmt.Sprintf(strings.Join(podStrings, ", "))
}

func ExpandConfigMap(configMap *api.ConfigMap, pod *api.Pod, client clientset.Interface) error {
	for k := range configMap.Data {
		tmpl, err := template.New("").Parse(configMap.Data[k])
		if err != nil {
			return err
		}
		ctx := NewMapContext(pod, client)
		buf := new(bytes.Buffer)
		err = tmpl.Execute(buf, ctx)
		if err != nil {
			return err
		}
		configMap.Data[k] = buf.String()
	}
	return nil
}

func RenderConfigMap(configMap *api.ConfigMap, pod *api.Pod, client clientset.Interface, tmplStr string) ([]byte, error) {
	err := ExpandConfigMap(configMap, pod, client)
	if err != nil {
		return nil, err
	}
	tmpl, err := template.New("").Parse(tmplStr)
	if err != nil {
		return nil, err
	}
	buf := new(bytes.Buffer)
	err = tmpl.Execute(buf, configMap.Data)
	return []byte(buf.String()), nil
}
