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

// A tiny script to help conver a given kubeconfig into a secret.
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
)

// TODO:
// Add a -o flag that writes to the specified destination file.
var (
	kubeconfig = flag.String("kubeconfig", "", "path to kubeconfig file.")
	name       = flag.String("name", "kubeconfig", "name to use in the metadata of the secret.")
	ns         = flag.String("ns", "default", "namespace of the secret.")
)

func read(file string) []byte {
	b, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("Cannot read file %v, %v", file, err)
	}
	return b
}

func main() {
	flag.Parse()
	if *kubeconfig == "" {
		log.Fatalf("Need to specify --kubeconfig")
	}
	cfg := read(*kubeconfig)
	secret := &api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      *name,
			Namespace: *ns,
		},
		Data: map[string][]byte{
			"config": cfg,
		},
	}
	fmt.Printf(runtime.EncodeOrDie(api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), secret))
}
