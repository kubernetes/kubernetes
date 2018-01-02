// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import "github.com/spf13/cobra"

var (
	cmdImage = &cobra.Command{
		Use:   "image [command]",
		Short: "Operate on image(s) in the local store",
		Long: `This subcommand operates on image(s) in the local store.

The "cat-manifest", "export", "extract", "render", and "rm" subcommands
take the ID or image name to reference images in the local store.

The ID can be specified using the long or short version, i.e. "sha512-78c08a541997",
or "sha512-78c08a5419979fff71e615f27aad75b84362a3cd9a13703b9d47ec27d1cfd029".

The image name can be specified including the version tag as stored in the local store,
i.e. "quay.io/coreos/etcd:latest", or "quay.io/coreos/etcd:v3.0.13".

The version tag may be left out, i.e. "quay.io/coreos/etcd".
In case of ambiguity, the least recently fetched image with this name will be chosen.`,
	}
)

func init() {
	cmdRkt.AddCommand(cmdImage)
}
