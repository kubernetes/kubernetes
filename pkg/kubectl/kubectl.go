/*
Copyright 2014 Google Inc. All rights reserved.

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

// A set of common functions needed by cmd/kubectl and pkg/kubectl packages.
package kubectl

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"gopkg.in/v1/yaml"
)

var apiVersionToUse = "v1beta1"

func GetKubeClient(config *client.Config, matchVersion bool) (*client.Client, error) {
	// TODO: get the namespace context when kubectl ns is completed
	c, err := client.New(config)
	if err != nil {
		return nil, err
	}

	if matchVersion {
		clientVersion := version.Get()
		serverVersion, err := c.ServerVersion()
		if err != nil {
			return nil, fmt.Errorf("Couldn't read version from server: %v\n", err)
		}
		if s := *serverVersion; !reflect.DeepEqual(clientVersion, s) {
			return nil, fmt.Errorf("Server version (%#v) differs from client version (%#v)!\n", s, clientVersion)
		}
	}

	return c, nil
}

type AuthInfo struct {
	User        string
	Password    string
	CAFile      string
	CertFile    string
	KeyFile     string
	BearerToken string
	Insecure    *bool
}

type NamespaceInfo struct {
	Namespace string
}

// LoadNamespaceInfo parses a NamespaceInfo object from a file path. It creates a file at the specified path if it doesn't exist with the default namespace.
func LoadNamespaceInfo(path string) (*NamespaceInfo, error) {
	var ns NamespaceInfo
	if _, err := os.Stat(path); os.IsNotExist(err) {
		ns.Namespace = api.NamespaceDefault
		err = SaveNamespaceInfo(path, &ns)
		return &ns, err
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &ns)
	if err != nil {
		return nil, err
	}
	return &ns, err
}

// SaveNamespaceInfo saves a NamespaceInfo object at the specified file path.
func SaveNamespaceInfo(path string, ns *NamespaceInfo) error {
	if !util.IsDNSLabel(ns.Namespace) {
		return fmt.Errorf("Namespace %s is not a valid DNS Label", ns.Namespace)
	}
	data, err := json.Marshal(ns)
	err = ioutil.WriteFile(path, data, 0600)
	return err
}

// LoadAuthInfo parses an AuthInfo object from a file path. It prompts user and creates file if it doesn't exist.
func LoadAuthInfo(path string, r io.Reader) (*AuthInfo, error) {
	var auth AuthInfo
	if _, err := os.Stat(path); os.IsNotExist(err) {
		auth.User = promptForString("Username", r)
		auth.Password = promptForString("Password", r)
		data, err := json.Marshal(auth)
		if err != nil {
			return &auth, err
		}
		err = ioutil.WriteFile(path, data, 0600)
		return &auth, err
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &auth)
	if err != nil {
		return nil, err
	}
	return &auth, err
}

func promptForString(field string, r io.Reader) string {
	fmt.Printf("Please enter %s: ", field)
	var result string
	fmt.Fscan(r, &result)
	return result
}

func CreateResource(resource, id string) ([]byte, error) {
	kind, err := resolveResource(resolveToKind, resource)
	if err != nil {
		return nil, err
	}

	s := fmt.Sprintf(`{"kind": "%s", "apiVersion": "%s", "id": "%s"}`, kind, apiVersionToUse, id)
	return []byte(s), nil
}

// TODO Move to labels package.
func formatLabels(labelMap map[string]string) string {
	l := labels.Set(labelMap).String()
	if l == "" {
		l = "<none>"
	}
	return l
}

func makeImageList(manifest api.ContainerManifest) string {
	var images []string
	for _, container := range manifest.Containers {
		images = append(images, container.Image)
	}
	return strings.Join(images, ",")
}

// Takes input 'data' as either json or yaml and attemps to decode it into the
// supplied object.
func dataToObject(data []byte) (runtime.Object, error) {
	// This seems hacky but we can't get the codec from kubeClient.
	versionInterfaces, err := latest.InterfacesFor(apiVersionToUse)
	if err != nil {
		return nil, err
	}
	obj, err := versionInterfaces.Codec.Decode(data)
	if err != nil {
		return nil, err
	}
	return obj, nil
}

const (
	resolveToPath = "path"
	resolveToKind = "kind"
)

// Takes a human-friendly reference to a resource and converts it to either a
// resource path for an API call or to a Kind to construct a JSON definition.
// See usages of the function for more context.
//
// target is one of the above constants ("path" or "kind") to determine what to
// resolve the resource to.
//
// resource is the human-friendly reference to the resource you want to
// convert.
func resolveResource(target, resource string) (string, error) {
	if target != resolveToPath && target != resolveToKind {
		return "", fmt.Errorf("Unrecognized target to convert to: %s", target)
	}

	var resolved string
	var err error
	// Caseless comparison.
	resource = strings.ToLower(resource)
	switch resource {
	case "pods", "pod", "po":
		if target == resolveToPath {
			resolved = "pods"
		} else {
			resolved = "Pod"
		}
	case "replicationcontrollers", "replicationcontroller", "rc":
		if target == resolveToPath {
			resolved = "replicationControllers"
		} else {
			resolved = "ReplicationController"
		}
	case "services", "service", "se":
		if target == resolveToPath {
			resolved = "services"
		} else {
			resolved = "Service"
		}
	case "minions", "minion", "mi":
		if target == resolveToPath {
			resolved = "minions"
		} else {
			resolved = "Minion"
		}
	default:
		// It might be a GUID, but we don't know how to handle those for now.
		err = fmt.Errorf("Resource %s not recognized; need pods, replicationControllers, services or minions.", resource)
	}
	return resolved, err
}

func resolveKindToResource(kind string) (resource string, err error) {
	// Determine the REST resource according to the type in data.
	switch kind {
	case "Pod":
		resource = "pods"
	case "ReplicationController":
		resource = "replicationControllers"
	case "Service":
		resource = "services"
	default:
		err = fmt.Errorf("Object %s not recognized", kind)
	}
	return
}

// versionAndKind will return the APIVersion and Kind of the given wire-format
// enconding of an APIObject, or an error. This is hacked in until the
// migration to v1beta3.
func versionAndKind(data []byte) (version, kind string, err error) {
	findKind := struct {
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	err = yaml.Unmarshal(data, &findKind)
	if err != nil {
		return "", "", fmt.Errorf("couldn't get version/kind: %v", err)
	}
	return findKind.APIVersion, findKind.Kind, nil
}
