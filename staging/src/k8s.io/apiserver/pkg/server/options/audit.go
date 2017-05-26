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
	"fmt"
	"io"
	"os"

	"github.com/spf13/pflag"
	"gopkg.in/natefinch/lumberjack.v2"

	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	pluginlog "k8s.io/apiserver/plugin/pkg/audit/log"
)

type AuditLogOptions struct {
	Path       string
	MaxAge     int
	MaxBackups int
	MaxSize    int

	PolicyFile string
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

	fs.StringVar(&o.PolicyFile, "audit-policy-file", o.PolicyFile,
		"Path to the file that defines the audit policy configuration. Requires the 'AdvancedAuditing' feature gate."+
			" With AdvancedAuditing, a profile is required to enable auditing.")
}

func (o *AuditLogOptions) ApplyTo(c *server.Config) error {
	if utilfeature.DefaultFeatureGate.Enabled(features.AdvancedAuditing) {
		if o.PolicyFile != "" {
			p, err := policy.LoadPolicyFromFile(o.PolicyFile)
			if err != nil {
				return err
			}
			c.AuditPolicyChecker = policy.NewChecker(p)
		}
	} else {
		if o.PolicyFile != "" {
			return fmt.Errorf("feature '%s' must be enabled to set an audit policy", features.AdvancedAuditing)
		}
	}

	// TODO: Generalize for alternative audit backends.
	if len(o.Path) == 0 {
		return nil
	}

	var w io.Writer = os.Stdout
	if o.Path != "-" {
		w = &lumberjack.Logger{
			Filename:   o.Path,
			MaxAge:     o.MaxAge,
			MaxBackups: o.MaxBackups,
			MaxSize:    o.MaxSize,
		}
	}
	c.LegacyAuditWriter = w
	c.AuditBackend = pluginlog.NewBackend(w)
	return nil
}
