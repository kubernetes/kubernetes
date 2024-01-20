/*
Copyright 2018 The Kubernetes Authors.

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
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/component-base/cli"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/logs/example"
	"k8s.io/klog/v2"

	_ "k8s.io/component-base/logs/json/register"
)

var featureGate = featuregate.NewFeatureGate()

func main() {
	runtime.Must(logsapi.AddFeatureGates(featureGate))
	command := NewLoggerCommand()
	code := cli.Run(command)
	os.Exit(code)
}

func NewLoggerCommand() *cobra.Command {
	c := logsapi.NewLoggingConfiguration()
	cmd := &cobra.Command{
		Run: func(cmd *cobra.Command, args []string) {
			logs.InitLogs()
			if err := logsapi.ValidateAndApply(c, featureGate); err != nil {
				fmt.Fprintf(os.Stderr, "%v\n", err)
				os.Exit(1)
			}
			if len(args) > 0 {
				fmt.Fprintf(os.Stderr, "Unexpected additional command line arguments:\n    %s\n", strings.Join(args, "\n    "))
				os.Exit(1)
			}

			// Initialize contextual logging.
			logger := klog.LoggerWithValues(klog.LoggerWithName(klog.Background(), "example"), "foo", "bar")
			ctx := klog.NewContext(context.Background(), logger)

			// Produce some output.
			example.Run(ctx)
		},
	}
	logsapi.AddFeatureGates(featureGate)
	featureGate.AddFlag(cmd.Flags(), "")
	logsapi.AddFlags(c, cmd.Flags())
	return cmd
}
