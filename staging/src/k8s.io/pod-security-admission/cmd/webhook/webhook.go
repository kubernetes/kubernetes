/*
Copyright 2021 The Kubernetes Authors.

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
	"flag"
	"math/rand"
	"os"
	"time"

	"github.com/spf13/pflag"

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
	_ "k8s.io/component-base/logs/json/register" // for JSON log format registration
	"k8s.io/klog/v2"
	"k8s.io/pod-security-admission/cmd/webhook/server"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	pflag.CommandLine.SetNormalizeFunc(cliflag.WordSepNormalizeFunc)

	command := server.NewServerCommand()

	logs.InitLogs()
	defer logs.FlushLogs()
	logFlags := flag.NewFlagSet("klog", flag.ContinueOnError)
	klog.InitFlags(logFlags)
	command.Flags().AddGoFlagSet(logFlags)

	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
