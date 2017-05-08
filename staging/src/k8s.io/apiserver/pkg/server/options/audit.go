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

package options

import (
	"github.com/spf13/pflag"
	"k8s.io/apiserver/pkg/auditor"
	"k8s.io/apiserver/pkg/server"
)

type AuditLogOptions struct {
	Path               string
	MaxAge             int
	MaxBackups         int
	MaxSize            int
	GroupedByUser      bool
	GroupedByNamespace bool
}

func NewAuditLogOptions() *AuditLogOptions {
	return &AuditLogOptions{}
}

func (o *AuditLogOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&o.Path, "audit-log-path", o.Path,
		"If set, all requests coming to the apiserver will be logged to this file.  '-' means standard out.")
	fs.IntVar(&o.MaxAge, "audit-log-maxage", o.MaxBackups,
		"The maximum number of days to retain old audit log files based on the timestamp encoded in their filename.")
	fs.IntVar(&o.MaxBackups, "audit-log-maxbackup", o.MaxBackups,
		"The maximum number of old audit log files to retain.")
	fs.IntVar(&o.MaxSize, "audit-log-maxsize", o.MaxSize,
		"The maximum size in megabytes of the audit log file before it gets rotated.")
	fs.BoolVar(&o.GroupedByUser, "audit-grouped-by-user", o.GroupedByUser,
		"Whether to split audit into multi log files by user. It will not work if audit-log-path is set to standard out. It conflicts with --audit-grouped-by-namespace")
	fs.BoolVar(&o.GroupedByNamespace, "audit-grouped-by-namespace", o.GroupedByNamespace,
		"Whether to split audit into multi log flies by namespace. It will not work if audit-log-path is set to standard out. It conflicts with --audit-grouped-by-user=true")
}

func (o *AuditLogOptions) ApplyTo(c *server.Config) error {
	if len(o.Path) == 0 {
		return nil
	}

	c.AuditWriter = auditor.NewGroupedAuditor(o.Path, o.MaxAge, o.MaxBackups, o.MaxSize, o.GroupedByUser, o.GroupedByNamespace)
	return nil
}
