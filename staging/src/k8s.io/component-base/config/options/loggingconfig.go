package options

import (
	"flag"
	"fmt"
	"github.com/spf13/pflag"
	"k8s.io/component-base/config"
	"k8s.io/component-base/logs"
	logs2 "k8s.io/component-base/logs/json"
	"k8s.io/klog/v2"
	"strings"
)

const (
	DefaultLogFormat = "text"
	JsonLogFormat    = "json"
)

var LogRegistry = logs.NewLogFormatRegistry()

func init() {
	// Text format is default klog format
	LogRegistry.Register(DefaultLogFormat, nil)
	LogRegistry.Register(JsonLogFormat, logs2.JSONLogger)
}

// List of logs (k8s.io/klog + k8s.io/component-base/logs) flags supported by all logging formats
var supportedLogsFlags = map[string]struct{}{
	"v":       {},
	"vmodule": {},
}

// BindLoggingFlags binds the LoggingConfiguration struct fields to a flagset
func BindLoggingFlags(c *config.LoggingConfiguration, fs *pflag.FlagSet) {
	unsupportedFlags := fmt.Sprintf("--%s", strings.Join(UnsupportedLoggingFlags(), ", --"))
	formats := fmt.Sprintf(`"%s"`, strings.Join(LogRegistry.List(), `", "`))
	fs.StringVar(&c.Format, "logging-format", c.Format, fmt.Sprintf("Sets the log format. Permitted formats: %s.\nNon-default formats don't honor these flags: %s.\nNon-default choices are currently alpha and subject to change without warning.", formats, unsupportedFlags))
	// No new log formats should be added after generation is of flag options
	LogRegistry.Freeze()
}

func UnsupportedLoggingFlags() []string {
	allFlags := []string{}
	// k8s.io/klog flags
	fs := &flag.FlagSet{}
	klog.InitFlags(fs)
	fs.VisitAll(func(flag *flag.Flag) {
		if _, found := supportedLogsFlags[flag.Name]; !found {
			allFlags = append(allFlags, flag.Name)
		}
	})

	// k8s.io/component-base/logs flags
	pfs := &pflag.FlagSet{}
	logs.AddFlags(pfs)
	pfs.VisitAll(func(flag *pflag.Flag) {
		if _, found := supportedLogsFlags[flag.Name]; !found {
			allFlags = append(allFlags, flag.Name)
		}
	})
	return allFlags
}
