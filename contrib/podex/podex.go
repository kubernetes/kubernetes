/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/ghodss/yaml"
	goyaml "gopkg.in/yaml.v2"
)

const usage = "podex [-format=yaml|json] [-type=pod|container] [-id NAME] IMAGES..."

var flManifestFormat = flag.String("format", "yaml", "manifest format to output, `yaml` or `json`")
var flManifestType = flag.String("type", "pod", "manifest type to output, `pod` or `container`")
var flManifestName = flag.String("name", "", "manifest name, default to image base name")
var flDaemon = flag.Bool("daemon", false, "daemon mode")

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s\n", usage)
		flag.PrintDefaults()
	}
}

type image struct {
	Host      string
	Namespace string
	Image     string
	Tag       string
}

func main() {
	flag.Parse()

	if *flDaemon {
		http.HandleFunc("/pods/", func(w http.ResponseWriter, r *http.Request) {
			image := strings.TrimPrefix(r.URL.Path, "/pods/")
			_, _, manifestName, _ := splitDockerImageName(image)
			manifest, err := getManifest(manifestName, "pod", "json", image)
			if err != nil {
				errMessage := fmt.Sprintf("failed to generate pod manifest for image %q: %v", image, err)
				log.Print(errMessage)
				http.Error(w, errMessage, http.StatusInternalServerError)
				return
			}
			io.Copy(w, manifest)
		})
		log.Fatal(http.ListenAndServe(":8080", nil))
	}

	if flag.NArg() < 1 {
		flag.Usage()
		log.Fatal("pod: missing image argument")
	}
	if *flManifestName == "" {
		if flag.NArg() > 1 {
			flag.Usage()
			log.Fatal("podex: -id arg is required when passing more than one image")
		}
		_, _, *flManifestName, _ = splitDockerImageName(flag.Arg(0))
	}
	if *flManifestType != "pod" && *flManifestType != "container" {
		flag.Usage()
		log.Fatalf("unsupported manifest type %q", *flManifestType)
	}
	if *flManifestFormat != "yaml" && *flManifestFormat != "json" {
		flag.Usage()
		log.Fatalf("unsupported manifest format %q", *flManifestFormat)
	}

	manifest, err := getManifest(*flManifestName, *flManifestType, *flManifestFormat, flag.Args()...)
	if err != nil {
		log.Fatalf("failed to generate %q manifest for %v: %v", *flManifestType, flag.Args(), err)
	}
	io.Copy(os.Stdout, manifest)
}

// getManifest infers a pod (or container) manifest for a list of docker images.
func getManifest(manifestName, manifestType, manifestFormat string, images ...string) (io.Reader, error) {
	podContainers := []goyaml.MapSlice{}

	for _, imageName := range images {
		host, namespace, repo, tag := splitDockerImageName(imageName)

		container := goyaml.MapSlice{
			{Key: "name", Value: repo},
			{Key: "image", Value: imageName},
		}

		img, err := getImageMetadata(host, namespace, repo, tag)

		if err != nil {
			return nil, fmt.Errorf("failed to get image metadata %q: %v", imageName, err)
		}
		portSlice := []goyaml.MapSlice{}
		for p := range img.ContainerConfig.ExposedPorts {
			port, err := strconv.Atoi(p.Port())
			if err != nil {
				return nil, fmt.Errorf("failed to parse port %q: %v", p.Port(), err)
			}
			portEntry := goyaml.MapSlice{{
				Key:   "name",
				Value: strings.Join([]string{repo, p.Proto(), p.Port()}, "-"),
			}, {
				Key:   "containerPort",
				Value: port,
			}}
			portSlice = append(portSlice, portEntry)
			if p.Proto() != "tcp" {
				portEntry = append(portEntry, goyaml.MapItem{Key: "protocol", Value: strings.ToUpper(p.Proto())})
			}
		}
		if len(img.ContainerConfig.ExposedPorts) > 0 {
			container = append(container, goyaml.MapItem{Key: "ports", Value: portSlice})
		}
		podContainers = append(podContainers, container)
	}

	// TODO(proppy): add flag to handle multiple version
	containerManifest := goyaml.MapSlice{
		{Key: "version", Value: "v1beta2"},
		{Key: "containers", Value: podContainers},
	}

	var data interface{}

	switch manifestType {
	case "container":
		containerManifest = append(goyaml.MapSlice{
			{Key: "id", Value: manifestName},
		}, containerManifest...)
		data = containerManifest
	case "pod":
		data = goyaml.MapSlice{
			{Key: "id", Value: manifestName},
			{Key: "kind", Value: "Pod"},
			{Key: "apiVersion", Value: "v1beta1"},
			{Key: "desiredState", Value: goyaml.MapSlice{
				{Key: "manifest", Value: containerManifest},
			}},
		}
	default:
		return nil, fmt.Errorf("unsupported manifest type %q", manifestFormat)
	}

	yamlBytes, err := goyaml.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal container manifest: %v", err)
	}

	switch manifestFormat {
	case "yaml":
		return bytes.NewBuffer(yamlBytes), nil
	case "json":
		jsonBytes, err := yaml.YAMLToJSON(yamlBytes)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal container manifest into JSON: %v", err)
		}
		var jsonPretty bytes.Buffer
		if err := json.Indent(&jsonPretty, jsonBytes, "", "  "); err != nil {
			return nil, fmt.Errorf("failed to indent json %q: %v", string(jsonBytes), err)
		}
		return &jsonPretty, nil
	default:
		return nil, fmt.Errorf("unsupported manifest format %q", manifestFormat)
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
		return nil, fmt.Errorf("error making request to %q: %v", host, err)
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("error getting X-Docker-Token from %s: %q", host, resp.Status)
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
