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
	"time"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"gopkg.in/natefinch/lumberjack.v2"

	auditv1beta1 "k8s.io/apiserver/pkg/apis/audit/v1beta1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	pluginbuffered "k8s.io/apiserver/plugin/pkg/audit/buffered"
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

const (
	// ModeBatch indicates that the audit backend should buffer audit events
	// internally, sending batch updates either once a certain number of
	// events have been received or a certain amount of time has passed.
	ModeBatch = "batch"
	// ModeBlocking causes the audit backend to block on every attempt to process
	// a set of events. This causes requests to the API server to wait for the
	// flush before sending a response.
	ModeBlocking = "blocking"
)

// AllowedModes is the modes known for audit backends.
var AllowedModes = []string{
	ModeBatch,
	ModeBlocking,
}

type AuditBatchOptions struct {
	// Should the backend asynchronous batch events to the webhook backend or
	// should the backend block responses?
	//
	// Defaults to asynchronous batch events.
	Mode string
	// Configuration for batching backend. Only used in batch mode.
	BatchConfig pluginbuffered.BatchConfig
}

// AuditLogOptions determines the output of the structured audit log by default.
// If the AdvancedAuditing feature is set to false, AuditLogOptions holds the legacy
// audit log writer.
type AuditLogOptions struct {
	Path       string
	MaxAge     int
	MaxBackups int
	MaxSize    int
	Format     string

	BatchOptions AuditBatchOptions
}

// AuditWebhookOptions control the webhook configuration for audit events.
type AuditWebhookOptions struct {
	ConfigFile     string
	InitialBackoff time.Duration

	BatchOptions AuditBatchOptions
}

func NewAuditOptions() *AuditOptions {
	defaultLogBatchConfig := pluginbuffered.NewDefaultBatchConfig()
	defaultLogBatchConfig.ThrottleEnable = false

	return &AuditOptions{
		WebhookOptions: AuditWebhookOptions{
			BatchOptions: AuditBatchOptions{
				Mode:        ModeBatch,
				BatchConfig: pluginbuffered.NewDefaultBatchConfig(),
			},
			InitialBackoff: pluginwebhook.DefaultInitialBackoff,
		},
		LogOptions: AuditLogOptions{
			Format: pluginlog.FormatJson,
			BatchOptions: AuditBatchOptions{
				Mode:        ModeBlocking,
				BatchConfig: defaultLogBatchConfig,
			},
		},
	}
}

// Validate checks invalid config combination
func (o *AuditOptions) Validate() []error {
	if o == nil {
		return nil
	}

	allErrors := []error{}

	if !advancedAuditingEnabled() {
		if len(o.PolicyFile) > 0 {
			allErrors = append(allErrors, fmt.Errorf("feature '%s' must be enabled to set option --audit-policy-file", features.AdvancedAuditing))
		}
		if len(o.WebhookOptions.ConfigFile) > 0 {
			allErrors = append(allErrors, fmt.Errorf("feature '%s' must be enabled to set option --audit-webhook-config-file", features.AdvancedAuditing))
		}
	}

	allErrors = append(allErrors, o.LogOptions.Validate()...)
	allErrors = append(allErrors, o.WebhookOptions.Validate()...)

	return allErrors
}

func validateBackendMode(pluginName string, mode string) error {
	for _, m := range AllowedModes {
		if m == mode {
			return nil
		}
	}
	return fmt.Errorf("invalid audit %s mode %s, allowed modes are %q", pluginName, mode, strings.Join(AllowedModes, ","))
}

func validateBackendBatchOptions(pluginName string, options AuditBatchOptions) error {
	if err := validateBackendMode(pluginName, options.Mode); err != nil {
		return err
	}
	if options.Mode != ModeBatch {
		// Don't validate the unused options.
		return nil
	}
	config := options.BatchConfig
	if config.BufferSize <= 0 {
		return fmt.Errorf("invalid audit batch %s buffer size %v, must be a positive number", pluginName, config.BufferSize)
	}
	if config.MaxBatchSize <= 0 {
		return fmt.Errorf("invalid audit batch %s max batch size %v, must be a positive number", pluginName, config.MaxBatchSize)
	}
	if config.ThrottleQPS <= 0 {
		return fmt.Errorf("invalid audit batch %s throttle QPS %v, must be a positive number", pluginName, config.ThrottleQPS)
	}
	if config.ThrottleBurst <= 0 {
		return fmt.Errorf("invalid audit batch %s throttle burst %v, must be a positive number", pluginName, config.ThrottleBurst)
	}
	return nil
}

func (o *AuditOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.PolicyFile, "audit-policy-file", o.PolicyFile,
		"Path to the file that defines the audit policy configuration. Requires the 'AdvancedAuditing' feature gate."+
			" With AdvancedAuditing, a profile is required to enable auditing.")

	o.LogOptions.AddFlags(fs)
	o.LogOptions.BatchOptions.AddFlags(pluginlog.PluginName, fs)
	o.WebhookOptions.AddFlags(fs)
	o.WebhookOptions.BatchOptions.AddFlags(pluginwebhook.PluginName, fs)
}

func (o *AuditOptions) ApplyTo(c *server.Config) error {
	if o == nil {
		return nil
	}

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

	if c.AuditBackend != nil && c.AuditPolicyChecker == nil {
		glog.V(2).Info("No audit policy file provided for AdvancedAuditing, no events will be recorded.")
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

func (o *AuditBatchOptions) AddFlags(pluginName string, fs *pflag.FlagSet) {
	fs.StringVar(&o.Mode, fmt.Sprintf("audit-%s-mode", pluginName), o.Mode,
		"Strategy for sending audit events. Blocking indicates sending events should block"+
			" server responses. Batch causes the backend to buffer and write events"+
			" asynchronously. Known modes are "+strings.Join(AllowedModes, ",")+".")
	fs.IntVar(&o.BatchConfig.BufferSize, fmt.Sprintf("audit-%s-batch-buffer-size", pluginName),
		o.BatchConfig.BufferSize, "The size of the buffer to store events before "+
			"batching and writing. Only used in batch mode.")
	fs.IntVar(&o.BatchConfig.MaxBatchSize, fmt.Sprintf("audit-%s-batch-max-size", pluginName),
		o.BatchConfig.MaxBatchSize, "The maximum size of a batch. Only used in batch mode.")
	fs.DurationVar(&o.BatchConfig.MaxBatchWait, fmt.Sprintf("audit-%s-batch-max-wait", pluginName),
		o.BatchConfig.MaxBatchWait, "The amount of time to wait before force writing the "+
			"batch that hadn't reached the max size. Only used in batch mode.")
	fs.BoolVar(&o.BatchConfig.ThrottleEnable, fmt.Sprintf("audit-%s-batch-throttle-enable", pluginName),
		o.BatchConfig.ThrottleEnable, "Whether batching throttling is enabled. Only used in batch mode.")
	fs.Float32Var(&o.BatchConfig.ThrottleQPS, fmt.Sprintf("audit-%s-batch-throttle-qps", pluginName),
		o.BatchConfig.ThrottleQPS, "Maximum average number of batches per second. "+
			"Only used in batch mode.")
	fs.IntVar(&o.BatchConfig.ThrottleBurst, fmt.Sprintf("audit-%s-batch-throttle-burst", pluginName),
		o.BatchConfig.ThrottleBurst, "Maximum number of requests sent at the same "+
			"moment if ThrottleQPS was not utilized before. Only used in batch mode.")
}

func (o *AuditBatchOptions) wrapBackend(delegate audit.Backend) audit.Backend {
	if o.Mode == ModeBlocking {
		return delegate
	}
	return pluginbuffered.NewBackend(delegate, o.BatchConfig)
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

func (o *AuditLogOptions) Validate() []error {
	// Check whether the log backend is enabled based on the options.
	if !o.enabled() {
		return nil
	}

	var allErrors []error
	if advancedAuditingEnabled() {
		if err := validateBackendBatchOptions(pluginlog.PluginName, o.BatchOptions); err != nil {
			allErrors = append(allErrors, err)
		}

		// Check log format
		validFormat := false
		for _, f := range pluginlog.AllowedFormats {
			if f == o.Format {
				validFormat = true
				break
			}
		}
		if !validFormat {
			allErrors = append(allErrors, fmt.Errorf("invalid audit log format %s, allowed formats are %q", o.Format, strings.Join(pluginlog.AllowedFormats, ",")))
		}
	}

	// Check validities of MaxAge, MaxBackups and MaxSize of log options, if file log backend is enabled.
	if o.MaxAge < 0 {
		allErrors = append(allErrors, fmt.Errorf("--audit-log-maxage %v can't be a negative number", o.MaxAge))
	}
	if o.MaxBackups < 0 {
		allErrors = append(allErrors, fmt.Errorf("--audit-log-maxbackup %v can't be a negative number", o.MaxBackups))
	}
	if o.MaxSize < 0 {
		allErrors = append(allErrors, fmt.Errorf("--audit-log-maxsize %v can't be a negative number", o.MaxSize))
	}

	return allErrors
}

// Check whether the log backend is enabled based on the options.
func (o *AuditLogOptions) enabled() bool {
	return o != nil && o.Path != ""
}

func (o *AuditLogOptions) getWriter() io.Writer {
	if !o.enabled() {
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
		log := pluginlog.NewBackend(w, o.Format, auditv1beta1.SchemeGroupVersion)
		c.AuditBackend = appendBackend(c.AuditBackend, o.BatchOptions.wrapBackend(log))
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
	fs.DurationVar(&o.InitialBackoff, "audit-webhook-initial-backoff",
		o.InitialBackoff, "The amount of time to wait before retrying the first failed request.")
	fs.DurationVar(&o.InitialBackoff, "audit-webhook-batch-initial-backoff",
		o.InitialBackoff, "The amount of time to wait before retrying the first failed request.")
	fs.MarkDeprecated("audit-webhook-batch-initial-backoff",
		"Deprecated, use --audit-webhook-initial-backoff instead.")
}

func (o *AuditWebhookOptions) Validate() []error {
	if !o.enabled() {
		return nil
	}

	var allErrors []error
	if advancedAuditingEnabled() {
		if err := validateBackendBatchOptions(pluginwebhook.PluginName, o.BatchOptions); err != nil {
			allErrors = append(allErrors, err)
		}
	}
	return allErrors
}

func (o *AuditWebhookOptions) enabled() bool {
	return o != nil && o.ConfigFile != ""
}

func (o *AuditWebhookOptions) applyTo(c *server.Config) error {
	if !o.enabled() {
		return nil
	}

	webhook, err := pluginwebhook.NewBackend(o.ConfigFile, auditv1beta1.SchemeGroupVersion, o.InitialBackoff)
	if err != nil {
		return fmt.Errorf("initializing audit webhook: %v", err)
	}
	c.AuditBackend = appendBackend(c.AuditBackend, o.BatchOptions.wrapBackend(webhook))
	return nil
}
