/*
Copyright 2026 The Kubernetes Authors.

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

package apidefinitions

import (
	"github.com/spf13/pflag"
)

// ReportArgs configures where a generator writes its API rule violation report.
type ReportArgs struct {
	// ReportFilename is the file path to write the API rule violation report to.
	// If unset, reporting is disabled.
	ReportFilename string
}

// AddReportFlags registers the --report-filename flag, parallel to AddFlags for
// LintArgs.
func AddReportFlags(args *ReportArgs, fs *pflag.FlagSet) {
	fs.StringVar(&args.ReportFilename, "report-filename", args.ReportFilename,
		"The file path to write the API rule violation report to. If unset reporting is disabled.")
}
