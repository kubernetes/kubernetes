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

package util

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/user"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	FlagMatchBinaryVersion = "match-server-version"
)

// Factory provides abstractions that allow the Kubectl command to be extended across multiple types
// of resources and different API sets.
// TODO: make the functions interfaces
// TODO: pass the various interfaces on the factory directly into the command constructors (so the
// commands are decoupled from the factory).
type Factory interface {
	Ring0Factory
	Ring1Factory
	Ring2Factory
}

func getGroupVersionKinds(gvks []schema.GroupVersionKind, group string) []schema.GroupVersionKind {
	result := []schema.GroupVersionKind{}
	for ix := range gvks {
		if gvks[ix].Group == group {
			result = append(result, gvks[ix])
		}
	}
	return result
}

func makeInterfacesFor(versionList []schema.GroupVersion) func(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
	accessor := meta.NewAccessor()
	return func(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
		for ix := range versionList {
			if versionList[ix].String() == version.String() {
				return &meta.VersionInterfaces{
					ObjectConvertor:  thirdpartyresourcedata.NewThirdPartyObjectConverter(api.Scheme),
					MetadataAccessor: accessor,
				}, nil
			}
		}
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %v)", version, versionList)
	}
}

type factory struct {
	Ring0Factory
	Ring1Factory
	Ring2Factory
}

// NewFactory creates a factory with the default Kubernetes resources defined
// if optionalClientConfig is nil, then flags will be bound to a new clientcmd.ClientConfig.
// if optionalClientConfig is not nil, then this factory will make use of it.
func NewFactory(optionalClientConfig clientcmd.ClientConfig) Factory {
	// this looks weird, but there's a reason.  In order for composers to be able to provide alternative factory implementations
	// they need to provide low level pieces of *certain* functions so that when the factory calls back into itself
	// it uses the custom version of the function.  Rather than try to enumerate everything that someone would wnat to override
	// we split the factory into rings, where each ring can depend on methods  an earlier ring, but cannot depend
	// upon peer methods in its own ring.
	// TODO pull the polymorphic command helpers out of the rings and run them separately.
	ring2Factory :=
		NewRing2Factory( // bits that depend on client mapper functions
			NewRing1Factory( // Object typing and most other things everything else.
				NewRing0Factory(optionalClientConfig))) // discovery, negotiation, and no-dep calls

	f := &factory{
		Ring0Factory: ring2Factory.ring1Factory().ring0Factory(),
		Ring1Factory: ring2Factory.ring1Factory(),
		Ring2Factory: ring2Factory,
	}

	return f
}

// GetFirstPod returns a pod matching the namespace and label selector
// and the number of all pods that match the label selector.
func GetFirstPod(client coreclient.PodsGetter, namespace string, selector labels.Selector, timeout time.Duration, sortBy func([]*v1.Pod) sort.Interface) (*api.Pod, int, error) {
	options := api.ListOptions{LabelSelector: selector}

	podList, err := client.Pods(namespace).List(options)
	if err != nil {
		return nil, 0, err
	}
	pods := []*v1.Pod{}
	for i := range podList.Items {
		pod := podList.Items[i]
		externalPod := &v1.Pod{}
		v1.Convert_api_Pod_To_v1_Pod(&pod, externalPod, nil)
		pods = append(pods, externalPod)
	}
	if len(pods) > 0 {
		sort.Sort(sortBy(pods))
		internalPod := &api.Pod{}
		v1.Convert_v1_Pod_To_api_Pod(pods[0], internalPod, nil)
		return internalPod, len(podList.Items), nil
	}

	// Watch until we observe a pod
	options.ResourceVersion = podList.ResourceVersion
	w, err := client.Pods(namespace).Watch(options)
	if err != nil {
		return nil, 0, err
	}
	defer w.Stop()

	condition := func(event watch.Event) (bool, error) {
		return event.Type == watch.Added || event.Type == watch.Modified, nil
	}
	event, err := watch.Until(timeout, w, condition)
	if err != nil {
		return nil, 0, err
	}
	pod, ok := event.Object.(*api.Pod)
	if !ok {
		return nil, 0, fmt.Errorf("%#v is not a pod event", event)
	}
	return pod, 1, nil
}

func makePortsString(ports []api.ServicePort, useNodePort bool) string {
	pieces := make([]string, len(ports))
	for ix := range ports {
		var port int32
		if useNodePort {
			port = ports[ix].NodePort
		} else {
			port = ports[ix].Port
		}
		pieces[ix] = fmt.Sprintf("%s:%d", strings.ToLower(string(ports[ix].Protocol)), port)
	}
	return strings.Join(pieces, ",")
}

func getPorts(spec api.PodSpec) []string {
	result := []string{}
	for _, container := range spec.Containers {
		for _, port := range container.Ports {
			result = append(result, strconv.Itoa(int(port.ContainerPort)))
		}
	}
	return result
}

func getProtocols(spec api.PodSpec) map[string]string {
	result := make(map[string]string)
	for _, container := range spec.Containers {
		for _, port := range container.Ports {
			result[strconv.Itoa(int(port.ContainerPort))] = string(port.Protocol)
		}
	}
	return result
}

// Extracts the ports exposed by a service from the given service spec.
func getServicePorts(spec api.ServiceSpec) []string {
	result := []string{}
	for _, servicePort := range spec.Ports {
		result = append(result, strconv.Itoa(int(servicePort.Port)))
	}
	return result
}

// Extracts the protocols exposed by a service from the given service spec.
func getServiceProtocols(spec api.ServiceSpec) map[string]string {
	result := make(map[string]string)
	for _, servicePort := range spec.Ports {
		result[strconv.Itoa(int(servicePort.Port))] = string(servicePort.Protocol)
	}
	return result
}

type clientSwaggerSchema struct {
	c        restclient.Interface
	cacheDir string
}

const schemaFileName = "schema.json"

type schemaClient interface {
	Get() *restclient.Request
}

func recursiveSplit(dir string) []string {
	parent, file := path.Split(dir)
	if len(parent) == 0 {
		return []string{file}
	}
	return append(recursiveSplit(parent[:len(parent)-1]), file)
}

func substituteUserHome(dir string) (string, error) {
	if len(dir) == 0 || dir[0] != '~' {
		return dir, nil
	}
	parts := recursiveSplit(dir)
	if len(parts[0]) == 1 {
		parts[0] = os.Getenv("HOME")
	} else {
		usr, err := user.Lookup(parts[0][1:])
		if err != nil {
			return "", err
		}
		parts[0] = usr.HomeDir
	}
	return path.Join(parts...), nil
}

func writeSchemaFile(schemaData []byte, cacheDir, cacheFile, prefix, groupVersion string) error {
	if err := os.MkdirAll(path.Join(cacheDir, prefix, groupVersion), 0755); err != nil {
		return err
	}
	tmpFile, err := ioutil.TempFile(cacheDir, "schema")
	if err != nil {
		// If we can't write, keep going.
		if os.IsPermission(err) {
			return nil
		}
		return err
	}
	if _, err := io.Copy(tmpFile, bytes.NewBuffer(schemaData)); err != nil {
		return err
	}
	if err := os.Link(tmpFile.Name(), cacheFile); err != nil {
		// If we can't write due to file existing, or permission problems, keep going.
		if os.IsExist(err) || os.IsPermission(err) {
			return nil
		}
		return err
	}
	return nil
}

func getSchemaAndValidate(c schemaClient, data []byte, prefix, groupVersion, cacheDir string, delegate validation.Schema) (err error) {
	var schemaData []byte
	var firstSeen bool
	fullDir, err := substituteUserHome(cacheDir)
	if err != nil {
		return err
	}
	cacheFile := path.Join(fullDir, prefix, groupVersion, schemaFileName)

	if len(cacheDir) != 0 {
		if schemaData, err = ioutil.ReadFile(cacheFile); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	if schemaData == nil {
		firstSeen = true
		schemaData, err = downloadSchemaAndStore(c, cacheDir, fullDir, cacheFile, prefix, groupVersion)
		if err != nil {
			return err
		}
	}
	schema, err := validation.NewSwaggerSchemaFromBytes(schemaData, delegate)
	if err != nil {
		return err
	}
	err = schema.ValidateBytes(data)
	if _, ok := err.(validation.TypeNotFoundError); ok && !firstSeen {
		// As a temporary hack, kubectl would re-get the schema if validation
		// fails for type not found reason.
		// TODO: runtime-config settings needs to make into the file's name
		schemaData, err = downloadSchemaAndStore(c, cacheDir, fullDir, cacheFile, prefix, groupVersion)
		if err != nil {
			return err
		}
		schema, err := validation.NewSwaggerSchemaFromBytes(schemaData, delegate)
		if err != nil {
			return err
		}
		return schema.ValidateBytes(data)
	}

	return err
}

// Download swagger schema from apiserver and store it to file.
func downloadSchemaAndStore(c schemaClient, cacheDir, fullDir, cacheFile, prefix, groupVersion string) (schemaData []byte, err error) {
	schemaData, err = c.Get().
		AbsPath("/swaggerapi", prefix, groupVersion).
		Do().
		Raw()
	if err != nil {
		return
	}
	if len(cacheDir) != 0 {
		if err = writeSchemaFile(schemaData, fullDir, cacheFile, prefix, groupVersion); err != nil {
			return
		}
	}
	return
}

func (c *clientSwaggerSchema) ValidateBytes(data []byte) error {
	gvk, err := json.DefaultMetaFactory.Interpret(data)
	if err != nil {
		return err
	}
	if ok := registered.IsEnabledVersion(gvk.GroupVersion()); !ok {
		// if we don't have this in our scheme, just skip validation because its an object we don't recognize
		return nil
	}

	switch gvk.Group {
	case api.GroupName:
		return getSchemaAndValidate(c.c, data, "api", gvk.GroupVersion().String(), c.cacheDir, c)

	default:
		return getSchemaAndValidate(c.c, data, "apis/", gvk.GroupVersion().String(), c.cacheDir, c)
	}
}
