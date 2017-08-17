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
	"strings"

	"github.com/spf13/pflag"
	"gopkg.in/natefinch/lumberjack.v2"

	"k8s.io/apimachinery/pkg/runtime/schema"
	auditv1alpha1 "k8s.io/apiserver/pkg/apis/audit/v1alpha1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	pluginlog "k8s.io/apiserver/plugin/pkg/audit/log"
	pluginwebhook "k8s.io/apiserver/plugin/pkg/audit/webhook"
)

func appendBackend(existing, newBackend audit.Backend) audit.Backend {
	if existing == nil {
		return newBackend
	}
	return audit.Union(existing, newBackend)
}

func advancedAuditingEnabled() bool {
	return utilfeature.DefaultFeatureGate.Enabled(features.AdvancedAuditing)
}

type AuditOptions struct {
	// Policy configuration file for filtering audit events that are captured.
	// If unspecified, a default is provided.
	PolicyFile string

	// Plugin options

	LogOptions     AuditLogOptions
	WebhookOptions AuditWebhookOptions
}

// AuditLogOptions holds the legacy audit log writer. If the AdvancedAuditing feature
// is enabled, these options determine the output of the structured audit log.
type AuditLogOptions struct {
	Path       string
	MaxAge     int
	MaxBackups int
	MaxSize    int
	Format     string
}

// AuditWebhookOptions control the webhook configuration for audit events.
type AuditWebhookOptions struct {
	ConfigFile string
	// Should the webhook asynchronous batch events to the webhook backend or
	// should the webhook block responses?
	//
	// Defaults to asynchronous batch events.
	Mode string
}

func NewAuditOptions() *AuditOptions {
	return &AuditOptions{
		WebhookOptions: AuditWebhookOptions{Mode: pluginwebhook.ModeBatch},
		LogOptions:     AuditLogOptions{Format: pluginlog.FormatLegacy},
	}
}

// Validate checks invalid config combination
func (o *AuditOptions) Validate() []error {
	allErrors := []error{}

	if !advancedAuditingEnabled() {
		if len(o.PolicyFile) > 0 {
			allErrors = append(allErrors, fmt.Errorf("feature '%s' must be enabled to set option --audit-policy-file", features.AdvancedAuditing))
		}
		if len(o.WebhookOptions.ConfigFile) > 0 {
			allErrors = append(allErrors, fmt.Errorf("feature '%s' must be enabled to set option --audit-webhook-config-file", features.AdvancedAuditing))
		}
	} else {
		// check webhook mode
		validMode := false
		for _, m := range pluginwebhook.AllowedModes {
			if m == o.WebhookOptions.Mode {
				validMode = true
				break
			}
		}
		if !validMode {
			allErrors = append(allErrors, fmt.Errorf("invalid audit webhook mode %s, allowed modes are %q", o.WebhookOptions.Mode, strings.Join(pluginwebhook.AllowedModes, ",")))
		}

		// check log format
		validFormat := false
		for _, f := range pluginlog.AllowedFormats {
			if f == o.LogOptions.Format {
				validFormat = true
				break
			}
		}
		if !validFormat {
			allErrors = append(allErrors, fmt.Errorf("invalid audit log format %s, allowed formats are %q", o.LogOptions.Format, strings.Join(pluginlog.AllowedFormats, ",")))
		}
	}

	// Check validities of MaxAge, MaxBackups and MaxSize of log options
	if o.LogOptions.MaxAge < 0 {
		allErrors = append(allErrors, fmt.Errorf("--audit-log-maxage %v can't be a negative number", o.LogOptions.MaxAge))
	}
	if o.LogOptions.MaxBackups < 0 {
		allErrors = append(allErrors, fmt.Errorf("--audit-log-maxbackup %v can't be a negative number", o.LogOptions.MaxBackups))
	}
	if o.LogOptions.MaxSize < 0 {
		allErrors = append(allErrors, fmt.Errorf("--audit-log-maxsize %v can't be a negative number", o.LogOptions.MaxSize))
	}

	return allErrors
}

func (o *AuditOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&o.PolicyFile, "audit-policy-file", o.PolicyFile,
		"Path to the file that defines the audit policy configuration. Requires the 'AdvancedAuditing' feature gate."+
			" With AdvancedAuditing, a profile is required to enable auditing.")

	o.LogOptions.AddFlags(fs)
	o.WebhookOptions.AddFlags(fs)
}

func (o *AuditOptions) ApplyTo(c *server.Config) error {
	// Apply legacy audit options if advanced audit is not enabled.
	if !advancedAuditingEnabled() {
		return o.LogOptions.legacyApplyTo(c)
	}

	// Apply advanced options if advanced audit is enabled.
	// 1. Apply generic options.
	if err := o.applyTo(c); err != nil {
		return err
	}

	// 2. Apply plugin options.
	if err := o.LogOptions.advancedApplyTo(c); err != nil {
		return err
	}
	if err := o.WebhookOptions.applyTo(c); err != nil {
		return err
	}
	return nil
}

func (o *AuditOptions) applyTo(c *server.Config) error {
	if o.PolicyFile == "" {
		return nil
	}

	p, err := policy.LoadPolicyFromFile(o.PolicyFile)
	if err != nil {
		return fmt.Errorf("loading audit policy file: %v", err)
	}
	c.AuditPolicyChecker = policy.NewChecker(p)
	return nil
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
	fs.StringVar(&o.Format, "audit-log-format", o.Format,
		"Format of saved audits. \"legacy\" indicates 1-line text format for each event."+
			" \"json\" indicates structured json format. Requires the 'AdvancedAuditing' feature"+
			" gate. Known formats are "+strings.Join(pluginlog.AllowedFormats, ",")+".")
}

func (o *AuditLogOptions) getWriter() io.Writer {
	if o.Path == "" {
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
	return w
}

func (o *AuditLogOptions) advancedApplyTo(c *server.Config) error {
	if w := o.getWriter(); w != nil {
		c.AuditBackend = appendBackend(c.AuditBackend, pluginlog.NewBackend(w, o.Format))
	}
	return nil
}

func (o *AuditLogOptions) legacyApplyTo(c *server.Config) error {
	c.LegacyAuditWriter = o.getWriter()
	return nil
}

func (o *AuditWebhookOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&o.ConfigFile, "audit-webhook-config-file", o.ConfigFile,
		"Path to a kubeconfig formatted file that defines the audit webhook configuration."+
			" Requires the 'AdvancedAuditing' feature gate.")
	fs.StringVar(&o.Mode, "audit-webhook-mode", o.Mode,
		"Strategy for sending audit events. Blocking indicates sending events should block"+
			" server responses. Batch causes the webhook to buffer and send events"+
			" asynchronously. Known modes are "+strings.Join(pluginwebhook.AllowedModes, ",")+".")
}

func (o *AuditWebhookOptions) applyTo(c *server.Config) error {
	if o.ConfigFile == "" {
		return nil
	}

	// TODO: switch to beta
	webhook, err := pluginwebhook.NewBackend(o.ConfigFile, o.Mode, []schema.GroupVersion{auditv1alpha1.SchemeGroupVersion})
	if err != nil {
		return fmt.Errorf("initializing audit webhook: %v", err)
	}
	c.AuditBackend = appendBackend(c.AuditBackend, webhook)
	return nil
}
