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

// podex is a command line tool to bootstrap kubernetes container
// manifest from docker image metadata.
//
// Manifests can then be edited by a human to match deployment needs.
//
// Example usage:
//
// $ docker pull google/nodejs-hello
// $ podex -yaml google/nodejs-hello > google/nodejs-hello/pod.yaml
// $ podex -json google/nodejs-hello > google/nodejs-hello/pod.json

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/ghodss/yaml"
)

const usage = "usage: podex [-json|-yaml] [-id PODNAME] IMAGES"

var generateJSON = flag.Bool("json", false, "generate json manifest")
var generateYAML = flag.Bool("yaml", false, "generate yaml manifest")
var podName = flag.String("id", "", "set pod name")

type image struct {
	Host      string
	Namespace string
	Image     string
	Tag       string
}

func main() {
	flag.Parse()

	if flag.NArg() < 1 {
		log.Fatal(usage)
	}
	if *podName == "" {
		if flag.NArg() > 1 {
			log.Print(usage)
			log.Fatal("podex: -id arg is required when passing more than one image")
		}
		_, _, *podName, _ = splitDockerImageName(flag.Arg(0))
	}

	if (!*generateJSON && !*generateYAML) || (*generateJSON && *generateYAML) {
		log.Fatal(usage)
	}

	podContainers := []v1beta1.Container{}

	for _, imageName := range flag.Args() {
		host, namespace, repo, tag := splitDockerImageName(imageName)

		container := v1beta1.Container{
			Name:  repo,
			Image: imageName,
		}

		img, err := getImageMetadata(host, namespace, repo, tag)

		if err != nil {
			log.Fatalf("failed to get image metadata %q: %v", imageName, err)
		}
		for p := range img.ContainerConfig.ExposedPorts {
			port, err := strconv.Atoi(p.Port())
			if err != nil {
				log.Fatalf("failed to parse port %q: %v", p.Port(), err)
			}
			container.Ports = append(container.Ports, v1beta1.Port{
				Name:          strings.Join([]string{repo, p.Proto(), p.Port()}, "-"),
				ContainerPort: port,
				Protocol:      v1beta1.Protocol(strings.ToUpper(p.Proto())),
			})
		}
		podContainers = append(podContainers, container)
	}

	// TODO(proppy): add flag to handle multiple version
	manifest := v1beta1.ContainerManifest{
		Version:    "v1beta1",
		ID:         *podName + "-pod",
		Containers: podContainers,
		RestartPolicy: v1beta1.RestartPolicy{
			Always: &v1beta1.RestartPolicyAlways{},
		},
	}

	if *generateJSON {
		bs, err := json.MarshalIndent(manifest, "", "  ")
		if err != nil {
			log.Fatalf("failed to render JSON container manifest: %v", err)
		}
		os.Stdout.Write(bs)
	}
	if *generateYAML {
		bs, err := yaml.Marshal(manifest)
		if err != nil {
			log.Fatalf("failed to render YAML container manifest: %v", err)
		}
		os.Stdout.Write(bs)
	}
}

// splitDockerImageName split a docker image name of the form [HOST/][NAMESPACE/]REPOSITORY[:TAG]
func splitDockerImageName(imageName string) (host, namespace, repo, tag string) {
	hostNamespaceImage := strings.Split(imageName, "/")
	last := len(hostNamespaceImage) - 1
	repoTag := strings.Split(hostNamespaceImage[last], ":")
	repo = repoTag[0]
	if len(repoTag) > 1 {
		tag = repoTag[1]
	}
	switch len(hostNamespaceImage) {
	case 2:
		host = ""
		namespace = hostNamespaceImage[0]
	case 3:
		host = hostNamespaceImage[0]
		namespace = hostNamespaceImage[1]
	}
	return
}

type Port string

func (p Port) Port() string {
	parts := strings.Split(string(p), "/")
	return parts[0]
}

func (p Port) Proto() string {
	parts := strings.Split(string(p), "/")
	if len(parts) == 1 {
		return "tcp"
	}
	return parts[1]
}

type imageMetadata struct {
	ID              string `json:"id"`
	ContainerConfig struct {
		ExposedPorts map[Port]struct{}
	} `json:"container_config"`
}

func getImageMetadata(host, namespace, repo, tag string) (*imageMetadata, error) {
	if host == "" {
		host = "index.docker.io"
	}
	if namespace == "" {
		namespace = "library"
	}
	if tag == "" {
		tag = "latest"
	}
	req, err := http.NewRequest("GET", fmt.Sprintf("https://%s/v1/repositories/%s/%s/images", host, namespace, repo), nil)

	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}
	req.Header.Add("X-Docker-Token", "true")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error getting X-Docker-Token from index.docker.io: %v", err)
	}
	endpoints := resp.Header.Get("X-Docker-Endpoints")
	token := resp.Header.Get("X-Docker-Token")
	req, err = http.NewRequest("GET", fmt.Sprintf("https://%s/v1/repositories/%s/%s/tags/%s", endpoints, namespace, repo, tag), nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}
	req.Header.Add("Authorization", "Token "+token)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error getting image id for %s/%s:%s %v", namespace, repo, tag, err)
	}
	var imageID string
	if err = json.NewDecoder(resp.Body).Decode(&imageID); err != nil {
		return nil, fmt.Errorf("error decoding image id: %v", err)
	}
	req, err = http.NewRequest("GET", fmt.Sprintf("https://%s/v1/images/%s/json", endpoints, imageID), nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}
	req.Header.Add("Authorization", "Token "+token)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error getting json for image %q: %v", imageID, err)
	}
	var image imageMetadata
	if err := json.NewDecoder(resp.Body).Decode(&image); err != nil {
		return nil, fmt.Errorf("error decoding image %q metadata: %v", imageID, err)
	}
	return &image, nil
}
