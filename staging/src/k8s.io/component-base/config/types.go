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

package config

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ClientConnectionConfiguration contains details for constructing a client.
type ClientConnectionConfiguration struct {
	// kubeconfig is the path to a KubeConfig file.
	Kubeconfig string
	// acceptContentTypes defines the Accept header sent by clients when connecting to a server, overriding the
	// default value of 'application/json'. This field will control all connections to the server used by a particular
	// client.
	AcceptContentTypes string
	// contentType is the content type used when sending data to the server from this client.
	ContentType string
	// qps controls the number of queries per second allowed for this connection.
	QPS float32
	// burst allows extra queries to accumulate when a client is exceeding its rate.
	Burst int32
}

// LeaderElectionConfiguration defines the configuration of leader election
// clients for components that can run with leader election enabled.
type LeaderElectionConfiguration struct {
	// leaderElect enables a leader election client to gain leadership
	// before executing the main loop. Enable this when running replicated
	// components for high availability.
	LeaderElect bool
	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	LeaseDuration metav1.Duration
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	RenewDeadline metav1.Duration
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	RetryPeriod metav1.Duration
	// resourceLock indicates the resource object type that will be used to lock
	// during leader election cycles.
	ResourceLock string
	// resourceName indicates the name of resource object that will be used to lock
	// during leader election cycles.
	ResourceName string
	// resourceNamespace indicates the namespace of resource object that will be used to lock
	// during leader election cycles.
	ResourceNamespace string
}

// DebuggingConfiguration holds configuration for Debugging related features.
type DebuggingConfiguration struct {
	// enableProfiling enables profiling via web interface host:port/debug/pprof/
	EnableProfiling bool
	// enableContentionProfiling enables lock contention profiling, if
	// enableProfiling is true.
	EnableContentionProfiling bool
}

// LoggingConfiguration contains logging options
// Refer [Logs Options](https://github.com/kubernetes/component-base/blob/master/logs/options.go) for more information.
type LoggingConfiguration struct {
	// Format Flag specifies the structure of log messages.
	// default value of format is `text`
	Format string
	// Maximum number of seconds between log flushes. Ignored if the
	// selected logging backend writes log messages without buffering.
	FlushFrequency time.Duration
	// Verbosity is the threshold that determines which log messages are
	// logged. Default is zero which logs only the most important
	// messages. Higher values enable additional messages. Error messages
	// are always logged.
	Verbosity VerbosityLevel
	// VModule overrides the verbosity threshold for individual files.
	// Only supported for "text" log format.
	VModule VModuleConfiguration
	// [Experimental] When enabled prevents logging of fields tagged as sensitive (passwords, keys, tokens).
	// Runtime log sanitization may introduce significant computation overhead and therefore should not be enabled in production.`)
	Sanitization bool
	// [Experimental] Options holds additional parameters that are specific
	// to the different logging formats. Only the options for the selected
	// format get used, but all of them get validated.
	Options FormatOptions
}

// FormatOptions contains options for the different logging formats.
type FormatOptions struct {
	// [Experimental] JSON contains options for logging format "json".
	JSON JSONOptions
}

// JSONOptions contains options for logging format "json".
type JSONOptions struct {
	// [Experimental] SplitStream redirects error messages to stderr while
	// info messages go to stdout, with buffering. The default is to write
	// both to stdout, without buffering.
	SplitStream bool
	// [Experimental] InfoBufferSize sets the size of the info stream when
	// using split streams. The default is zero, which disables buffering.
	InfoBufferSize resource.QuantityValue
}

// VModuleConfiguration is a collection of individual file names or patterns
// and the corresponding verbosity threshold.
type VModuleConfiguration []VModuleItem

var _ pflag.Value = &VModuleConfiguration{}

// VModuleItem defines verbosity for one or more files which match a certain
// glob pattern.
type VModuleItem struct {
	// FilePattern is a base file name (i.e. minus the ".go" suffix and
	// directory) or a "glob" pattern for such a name. It must not contain
	// comma and equal signs because those are separators for the
	// corresponding klog command line argument.
	FilePattern string
	// Verbosity is the threshold for log messages emitted inside files
	// that match the pattern.
	Verbosity VerbosityLevel
}

// String returns the -vmodule parameter (comma-separated list of pattern=N).
func (vmodule *VModuleConfiguration) String() string {
	var patterns []string
	for _, item := range *vmodule {
		patterns = append(patterns, fmt.Sprintf("%s=%d", item.FilePattern, item.Verbosity))
	}
	return strings.Join(patterns, ",")
}

// Set parses the -vmodule parameter (comma-separated list of pattern=N).
func (vmodule *VModuleConfiguration) Set(value string) error {
	// This code mirrors https://github.com/kubernetes/klog/blob/9ad246211af1ed84621ee94a26fcce0038b69cd1/klog.go#L287-L313

	for _, pat := range strings.Split(value, ",") {
		if len(pat) == 0 {
			// Empty strings such as from a trailing comma can be ignored.
			continue
		}
		patLev := strings.Split(pat, "=")
		if len(patLev) != 2 || len(patLev[0]) == 0 || len(patLev[1]) == 0 {
			return fmt.Errorf("%q does not have the pattern=N format", pat)
		}
		pattern := patLev[0]
		// 31 instead of 32 to ensure that it also fits into int32.
		v, err := strconv.ParseUint(patLev[1], 10, 31)
		if err != nil {
			return fmt.Errorf("parsing verbosity in %q: %v", pat, err)
		}
		*vmodule = append(*vmodule, VModuleItem{FilePattern: pattern, Verbosity: VerbosityLevel(v)})
	}
	return nil
}

func (vmodule *VModuleConfiguration) Type() string {
	return "pattern=N,..."
}

// VerbosityLevel represents a klog or logr verbosity threshold.
type VerbosityLevel uint32

var _ pflag.Value = new(VerbosityLevel)

func (l *VerbosityLevel) String() string {
	return strconv.FormatInt(int64(*l), 10)
}

func (l *VerbosityLevel) Get() interface{} {
	return *l
}

func (l *VerbosityLevel) Set(value string) error {
	// Limited to int32 for compatibility with klog.
	v, err := strconv.ParseUint(value, 10, 31)
	if err != nil {
		return err
	}
	*l = VerbosityLevel(v)
	return nil
}

func (l *VerbosityLevel) Type() string {
	return "Level"
}
