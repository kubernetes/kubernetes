// Copyright 2016 The rkt Authors
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

	"github.com/spf13/cobra"
)

var (
	cmdConfig = &cobra.Command{
		Use:   "config",
		Short: "Print configuration for each stage in JSON format",
		Long: `The output will be parsable JSON with "stage0" and stage1" as keys and rkt configuration entries as values.
The generated configuration entries resemble the original rkt configuration format.`,
		Run: runWrapper(runConfig),
	}
	flagConfigPrettyPrint bool
)

func init() {
	cmdRkt.AddCommand(cmdConfig)
	cmdConfig.Flags().BoolVar(&flagConfigPrettyPrint, "pretty-print", true, "apply indent to format the output")
}

func runConfig(cmd *cobra.Command, args []string) int {
	config, err := getConfig()
	if err != nil {
		stderr.PrintE("cannot get configuration", err)
		return 254
	}

	var b []byte
	if flagConfigPrettyPrint {
		b, err = json.MarshalIndent(config, "", "\t")
	} else {
		b, err = json.Marshal(config)
	}
	if err != nil {
		stderr.PanicE("error marshaling configuration", err)
	}

	stdout.Print(string(b))
	return 0
}
