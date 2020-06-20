/*
Copyright 2020 The Kubernetes Authors.

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
	"log"
	"os"
)

const (
	dataDirEnv         = "DATA_DIRECTORY"
	defaultEtcdDataDir = "/var/etcd/data/"
)

type backupOpts struct {
	dataDir string
}

func (opts *backupOpts) validateAndDefault() error {
	if opts.dataDir == "" {
		// fallback to env
		val, ok := os.LookupEnv(dataDirEnv)
		if ok && len(val) != 0 {
			log.Printf("--data-dir unset, falling back to %s variable", dataDirEnv)
			opts.dataDir = val
		} else {
			// fallback to default
			log.Printf("--data-dir unset, defaulting to %s", defaultEtcdDataDir)
			opts.dataDir = defaultEtcdDataDir

		}
	}
	return nil
}
