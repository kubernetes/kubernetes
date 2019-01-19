/*
Copyright 2019 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"k8s.io/component-base/experimental/cli/kflag"
	"k8s.io/component-base/experimental/cli/kflag/example/config"
	"k8s.io/component-base/experimental/cli/kflag/example/options"
)

// go run example.go --num=3 --other-num=6 --map=enableFoo=false,enableBaz=true
func main() {
	fs := kflag.NewFlagSet("component")
	applyFlags := options.AddFlags(fs)
	applyConfigFlags := options.AddConfigFlags(fs)

	// could also be args from Cobra
	if err := fs.Parse(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, "Fatal:", err)
		os.Exit(1)
	}

	// Note how we can construct these at an arbitrary point *after* parsing
	flags := options.NewFlags()
	fmt.Println("Default Flags:", flags)
	applyFlags(flags)
	fmt.Println("Parsed Flags:", flags)

	config := options.NewConfig()
	fmt.Println("Default Config:", config)
	applyConfigFlags(config)
	fmt.Println("Parsed Config:", config)

	// MAKE ARBITRARY DECISIONS

	// Load a config file, don't have to re-parse to enforce flag precedence
	config, err := loadConfig("config.json")
	if err != nil {
		fmt.Fprintln(os.Stderr, "Fatal:", err)
		os.Exit(1)
	}
	applyConfigFlags(config)

	// MAKE MORE ARBITRARY DECISIONS
}

func loadConfig(path string) (*config.Config, error) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	c := options.NewConfig()
	if err := json.Unmarshal(b, c); err != nil {
		return nil, err
	}
	return c, nil
}
