//go:build go1.21
// +build go1.21

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
	"log/slog"
	"os"
	"strings"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/component-base/cli"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
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
			// This configures the global logger in klog *and* slog, if compiled
			// with Go >= 1.21.
			logs.InitLogs()
			if err := logsapi.ValidateAndApply(c, featureGate); err != nil {
				fmt.Fprintf(os.Stderr, "%v\n", err)
				os.Exit(1)
			}
			if len(args) > 0 {
				fmt.Fprintf(os.Stderr, "Unexpected additional command line arguments:\n    %s\n", strings.Join(args, "\n    "))
				os.Exit(1)
			}

			// Produce some output. Special types used by Kubernetes work.
			podRef := klog.KObj(&metav1.ObjectMeta{Name: "some-pod", Namespace: "some-namespace"})
			podRefs := klog.KObjSlice([]interface{}{
				&metav1.ObjectMeta{Name: "some-pod", Namespace: "some-namespace"},
				nil,
				&metav1.ObjectMeta{Name: "other-pod"},
			})
			slog.Info("slog.Info", "pod", podRef, "pods", podRefs)
			klog.InfoS("klog.InfoS", "pod", podRef, "pods", podRefs)
			klog.Background().Info("klog.Background+logr.Logger.Info")
			klog.FromContext(context.Background()).Info("klog.FromContext+logr.Logger.Info")
			slogLogger := slog.Default()
			slogLogger.Info("slog.Default+slog.Logger.Info")
		},
	}
	if err := logsapi.AddFeatureGates(featureGate); err != nil {
		// Shouldn't happen.
		panic(err)
	}
	featureGate.AddFlag(cmd.Flags(), "")
	logsapi.AddFlags(c, cmd.Flags())
	return cmd
}
