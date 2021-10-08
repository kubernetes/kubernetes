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
	"errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"k8s.io/component-base/cli"
	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"

	_ "k8s.io/component-base/logs/json/register"
)

func main() {
	command := NewLoggerCommand()
	code := cli.Run(command)
	os.Exit(code)
}

func NewLoggerCommand() *cobra.Command {
	o := logs.NewOptions()
	cmd := &cobra.Command{
		Run: func(cmd *cobra.Command, args []string) {
			errs := o.Validate()
			if len(errs) != 0 {
				fmt.Fprintf(os.Stderr, "%v\n", errs)
				os.Exit(1)
			}
			o.Apply()
			runLogger()
		},
	}
	o.AddFlags(cmd.Flags())
	return cmd
}

func runLogger() {
	klog.Infof("Log using Infof, key: %s", "value")
	klog.InfoS("Log using InfoS", "key", "value")
	err := errors.New("fail")
	klog.Errorf("Log using Errorf, err: %v", err)
	klog.ErrorS(err, "Log using ErrorS")
	data := SensitiveData{Key: "secret"}
	klog.Infof("Log with sensitive key, data: %q", data)
	klog.V(1).Info("Log less important message")
}

type SensitiveData struct {
	Key string `json:"key" datapolicy:"secret-key"`
}
