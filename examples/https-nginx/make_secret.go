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

// A small script that converts the given open ssl public/private keys to
// a secret that it writes to stdout as json. Most common use case is to
// create a secret from self signed certificates used to authenticate with
// a devserver. Usage: go run make_secret.go -crt ca.crt -key priv.key > secret.json
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"

	// This installs the legacy v1 API
	_ "k8s.io/kubernetes/pkg/api/install"
)

// TODO:
// Add a -o flag that writes to the specified destination file.
// Teach the script to create crt and key if -crt and -key aren't specified.
var (
	crt = flag.String("crt", "", "path to nginx certificates.")
	key = flag.String("key", "", "path to nginx private key.")
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
	if *crt == "" || *key == "" {
		log.Fatalf("Need to specify -crt -key and -template")
	}
	nginxCrt := read(*crt)
	nginxKey := read(*key)
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name: "nginxsecret",
		},
		Data: map[string][]byte{
			"nginx.crt": nginxCrt,
			"nginx.key": nginxKey,
		},
	}
	fmt.Printf(runtime.EncodeOrDie(api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), secret))
}
