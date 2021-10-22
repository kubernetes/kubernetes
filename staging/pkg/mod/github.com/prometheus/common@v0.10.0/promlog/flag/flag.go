// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package flag

import (
	"github.com/prometheus/common/promlog"
	kingpin "gopkg.in/alecthomas/kingpin.v2"
)

// LevelFlagName is the canonical flag name to configure the allowed log level
// within Prometheus projects.
const LevelFlagName = "log.level"

// LevelFlagHelp is the help description for the log.level flag.
const LevelFlagHelp = "Only log messages with the given severity or above. One of: [debug, info, warn, error]"

// FormatFlagName is the canonical flag name to configure the log format
// within Prometheus projects.
const FormatFlagName = "log.format"

// FormatFlagHelp is the help description for the log.format flag.
const FormatFlagHelp = "Output format of log messages. One of: [logfmt, json]"

// AddFlags adds the flags used by this package to the Kingpin application.
// To use the default Kingpin application, call AddFlags(kingpin.CommandLine)
func AddFlags(a *kingpin.Application, config *promlog.Config) {
	config.Level = &promlog.AllowedLevel{}
	a.Flag(LevelFlagName, LevelFlagHelp).
		Default("info").SetValue(config.Level)

	config.Format = &promlog.AllowedFormat{}
	a.Flag(FormatFlagName, FormatFlagHelp).
		Default("logfmt").SetValue(config.Format)
}
