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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
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
			return nil, fmt.Errorf("couldn't read version from server: %v\n", err)
		}
		if s := *serverVersion; !reflect.DeepEqual(clientVersion, s) {
			return nil, fmt.Errorf("server version (%#v) differs from client version (%#v)!\n", s, clientVersion)
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
		return fmt.Errorf("namespace %s is not a valid DNS Label", ns.Namespace)
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

// ExpandResourceShortcut will return the expanded version of resource
// (something that a pkg/api/meta.RESTMapper can understand), if it is
// indeed a shortcut. Otherwise, will return resource unmodified.
// TODO: Combine with RESTMapper stuff to provide a general solution
// to this problem.
func ExpandResourceShortcut(resource string) string {
	shortForms := map[string]string{
		"po": "pods",
		"rc": "replicationcontrollers",
		"se": "services",
		"mi": "minions",
		"ev": "events",
	}
	if expanded, ok := shortForms[resource]; ok {
		return expanded
	}
	return resource
}
