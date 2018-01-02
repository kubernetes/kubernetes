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

import (
	"encoding/json"

	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/spf13/cobra"
)

var (
	cmdCatManifest = &cobra.Command{
		Use:   "cat-manifest --uuid-file=FILE | UUID ...",
		Short: "Inspect and print the pod manifest",
		Long:  `UUID should be the UUID of a pod`,
		Run:   runWrapper(runCatManifest),
	}
	flagPMPrettyPrint bool
)

func init() {
	cmdRkt.AddCommand(cmdCatManifest)
	cmdCatManifest.Flags().BoolVar(&flagPMPrettyPrint, "pretty-print", true, "apply indent to format the output")
	cmdCatManifest.Flags().StringVar(&flagUUIDFile, "uuid-file", "", "read pod UUID from file instead of argument")
}

func runCatManifest(cmd *cobra.Command, args []string) (exit int) {
	var podUUID string

	if flagUUIDFile != "" {
		uuid, err := pkgPod.ReadUUIDFromFile(flagUUIDFile)
		if err != nil {
			stderr.PrintE("unable to resolve UUID from file", err)
			return 254
		}
		podUUID = uuid
	} else {
		if len(args) != 1 {
			cmd.Usage()
			return 254
		}
		podUUID = args[0]
	}

	pod, err := pkgPod.PodFromUUIDString(getDataDir(), podUUID)
	if err != nil {
		stderr.PrintE("problem retrieving pod", err)
		return 254
	}

	defer pod.Close()

	_, manifest, err := pod.PodManifest()
	if err != nil {
		return 254
	}

	var b []byte
	if flagPMPrettyPrint {
		b, err = json.MarshalIndent(manifest, "", "\t")
	} else {
		b, err = json.Marshal(manifest)
	}
	if err != nil {
		stderr.PrintE("cannot read the pod manifest", err)
		return 254
	}

	stdout.Print(string(b))
	return 0
}
