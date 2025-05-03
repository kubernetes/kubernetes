/*
Copyright 2016 The Kubernetes Authors.

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

package app

import (
	"flag"
	"os"

	"github.com/spf13/pflag"

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd"
)

// Run creates and executes new kubeadm command
func Run() error {
	var allFlags flag.FlagSet
	klog.InitFlags(&allFlags)
	// Only add the flags that are currently supported for kubeadm. This
	// preprevents new klog flags from being accidentally exposed.
	allFlags.VisitAll(func(f *flag.Flag) {
		switch f.Name {
		case
			"v",
			"add_dir_header",
			"skip_headers",
			// Below are flags we support but don't expose in the help.
			"alsologtostderr",
			"log_backtrace_at",
			"log_dir",
			"logtostderr",
			"log_file",
			"log_file_max_size",
			"one_output",
			"skip_log_headers",
			"stderrthreshold",
			"vmodule":
			flag.CommandLine.Var(f.Value, f.Name, f.Usage)
		}
	})

	pflag.CommandLine.SetNormalizeFunc(cliflag.WordSepNormalizeFunc)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)

	// We hide the klog flags that most users will not care about to reduce the
	// clutter from the output. Note that these MarkHidden calls must be after
	// the lines above.
	pflag.CommandLine.MarkHidden("alsologtostderr")   //nolint:errcheck
	pflag.CommandLine.MarkHidden("log-backtrace-at")  //nolint:errcheck
	pflag.CommandLine.MarkHidden("log-dir")           //nolint:errcheck
	pflag.CommandLine.MarkHidden("logtostderr")       //nolint:errcheck
	pflag.CommandLine.MarkHidden("log-file")          //nolint:errcheck
	pflag.CommandLine.MarkHidden("log-file-max-size") //nolint:errcheck
	pflag.CommandLine.MarkHidden("one-output")        //nolint:errcheck
	pflag.CommandLine.MarkHidden("skip-log-headers")  //nolint:errcheck
	pflag.CommandLine.MarkHidden("stderrthreshold")   //nolint:errcheck
	pflag.CommandLine.MarkHidden("vmodule")           //nolint:errcheck

	cmd := cmd.NewKubeadmCommand(os.Stdin, os.Stdout, os.Stderr)
	return cmd.Execute()
}
