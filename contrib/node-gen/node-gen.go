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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
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
	versions, err := latest.InterfacesFor("v1beta3")
	if err != nil {
		return nil, err
	}
	codec := versions.Codec

	list := &api.NodeList{
		Items: []api.Node{},
	}

	for _, nodename := range nodenames {
		node := &api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   nodename,
				Labels: labelmap,
			},
		}
		// The Encode/Decode obviously isn't required, but it will cause the
		// Spec.ExternalID to get set == Name. But why not, right?
		b, err := codec.Encode(node)
		if err != nil {
			return nil, err
		}
		err = codec.DecodeInto(b, node)
		if err != nil {
			return nil, err
		}

		list.Items = append(list.Items, *node)
	}

	jsonBytes, err := codec.Encode(list)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal container manifest into JSON: %v", err)
	}

	var jsonPretty bytes.Buffer
	if err := json.Indent(&jsonPretty, jsonBytes, "", "  "); err != nil {
		return nil, fmt.Errorf("failed to indent json %q: %v", string(jsonBytes), err)
	}
	return &jsonPretty, nil
}
