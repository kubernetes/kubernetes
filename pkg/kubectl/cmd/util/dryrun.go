/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func SetupDryRun(cmd *cobra.Command) {
	AddDryRunFlag(cmd)
	AddServerDryRunFlag(cmd)
}

type DryRun struct {
	Client bool
	Server bool
}

// NewDryRunFromCmd crates a new DryRun structure by reading the values
// of the flags in the provided cmd.
func NewDryRunFromCmd(cmd *cobra.Command) *DryRun {
	return &DryRun{
		Client: GetDryRunFlag(cmd),
		Server: GetServerDryRunFlag(cmd),
	}
}

// IsDryRun returns true either if it's client-side dry-run or server-side dry-run
func (d *DryRun) IsDryRun() bool {
	return d.Server || d.Client
}

func (d *DryRun) dryRunFlag() []string {
	if d.Server {
		return []string{metav1.DryRunAll}
	}
	return []string{}
}

// CreateOptions adds the dry-run information in the CreateOptions
// passed. If nil is received a new UpdateOptions is created, the new
// value is returned.
func (d *DryRun) CreateOptions(opts *metav1.CreateOptions) *metav1.CreateOptions {
	if opts == nil {
		opts = &metav1.CreateOptions{}
	}
	opts.DryRun = d.dryRunFlag()

	return opts
}

// UpdateOptions adds the dry-run information in the UpdateOptions
// arg. If nil is received a new UpdateOptions is created. The value
// is returned.
func (d *DryRun) UpdateOptions(opts *metav1.UpdateOptions) *metav1.UpdateOptions {
	if opts == nil {
		opts = &metav1.UpdateOptions{}
	}
	opts.DryRun = d.dryRunFlag()

	return opts
}

// DeleteOptions adds the dry-run information in the DeleteOptions
// passed. If nil is received a new UpdateOptions is created, the new
// value is returned.
func (d *DryRun) DeleteOptions(opts *metav1.DeleteOptions) *metav1.DeleteOptions {
	if opts == nil {
		opts = &metav1.DeleteOptions{}
	}
	opts.DryRun = d.dryRunFlag()

	return opts
}
