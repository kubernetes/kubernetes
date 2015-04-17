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

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
)

const usage = "podex [--labels=LABELS] NODES..."

var inlabels = flag.String("labels", "", "labels to apply to nodes")

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s\n", usage)
		flag.PrintDefaults()
	}
}

func main() {
	flag.Parse()

	if flag.NArg() < 1 {
		flag.Usage()
		log.Fatal("missing node names")
	}

	var labels []string

	if len(*inlabels) > 0 {
		labels = strings.Split(*inlabels, ",")
	}

	labelmap := make(map[string]string)

	for _, label := range labels {
		kv := strings.Split(label, "=")
		if len(kv) != 2 {
			log.Fatalf("invalid label %v\n", label)
		}
		labelmap[kv[0]] = kv[1]
	}

	manifest, err := getManifest(labelmap, flag.Args()...)
	if err != nil {
		log.Fatalf("failed to generate node manifest for %v: %v", flag.Args(), err)
	}
	io.Copy(os.Stdout, manifest)
}

func getManifest(labelmap map[string]string, nodenames ...string) (io.Reader, error) {
	typeMeta := v1beta3.TypeMeta{
		APIVersion: "v1beta3",
		Kind:       "List",
	}
	list := &v1beta3.NodeList{
		TypeMeta: typeMeta,
		Items:    []v1beta3.Node{},
	}

	typeMeta.Kind = "Node"
	for _, nodename := range nodenames {
		metadata := v1beta3.ObjectMeta{
			Name:   nodename,
			Labels: labelmap,
		}
		node := v1beta3.Node{
			TypeMeta:   typeMeta,
			ObjectMeta: metadata,
		}
		list.Items = append(list.Items, node)
	}

	jsonBytes, err := json.Marshal(list)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal container manifest into JSON: %v", err)
	}

	var jsonPretty bytes.Buffer
	if err := json.Indent(&jsonPretty, jsonBytes, "", "  "); err != nil {
		return nil, fmt.Errorf("failed to indent json %q: %v", string(jsonBytes), err)
	}
	return &jsonPretty, nil
}
