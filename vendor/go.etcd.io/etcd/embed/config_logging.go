// Copyright 2018 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package embed

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"sync"

	"go.etcd.io/etcd/pkg/logutil"

	"github.com/coreos/pkg/capnslog"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
)

// GetLogger returns the logger.
func (cfg Config) GetLogger() *zap.Logger {
	cfg.loggerMu.RLock()
	l := cfg.logger
	cfg.loggerMu.RUnlock()
	return l
}

// for testing
var grpcLogOnce = new(sync.Once)

// setupLogging initializes etcd logging.
// Must be called after flag parsing or finishing configuring embed.Config.
func (cfg *Config) setupLogging() error {
	// handle "DeprecatedLogOutput" in v3.4
	// TODO: remove "DeprecatedLogOutput" in v3.5
	len1 := len(cfg.DeprecatedLogOutput)
	len2 := len(cfg.LogOutputs)
	if len1 != len2 {
		switch {
		case len1 > len2: // deprecate "log-output" flag is used
			fmt.Fprintln(os.Stderr, "'--log-output' flag has been deprecated! Please use '--log-outputs'!")
			cfg.LogOutputs = cfg.DeprecatedLogOutput
		case len1 < len2: // "--log-outputs" flag has been set with multiple writers
			cfg.DeprecatedLogOutput = []string{}
		}
	} else {
		if len1 > 1 {
			return errors.New("both '--log-output' and '--log-outputs' are set; only set '--log-outputs'")
		}
		if len1 < 1 {
			return errors.New("either '--log-output' or '--log-outputs' flag must be set")
		}
		if reflect.DeepEqual(cfg.DeprecatedLogOutput, cfg.LogOutputs) && cfg.DeprecatedLogOutput[0] != DefaultLogOutput {
			return fmt.Errorf("'--log-output=%q' and '--log-outputs=%q' are incompatible; only set --log-outputs", cfg.DeprecatedLogOutput, cfg.LogOutputs)
		}
		if !reflect.DeepEqual(cfg.DeprecatedLogOutput, []string{DefaultLogOutput}) {
			fmt.Fprintf(os.Stderr, "[WARNING] Deprecated '--log-output' flag is set to %q\n", cfg.DeprecatedLogOutput)
			fmt.Fprintln(os.Stderr, "Please use '--log-outputs' flag")
		}
	}

	// TODO: remove after deprecating log related flags in v3.5
	if cfg.Debug {
		fmt.Fprintf(os.Stderr, "[WARNING] Deprecated '--debug' flag is set to %v (use '--log-level=debug' instead\n", cfg.Debug)
	}
	if cfg.Debug && cfg.LogLevel != "debug" {
		fmt.Fprintf(os.Stderr, "[WARNING] Deprecated '--debug' flag is set to %v with inconsistent '--log-level=%s' flag\n", cfg.Debug, cfg.LogLevel)
	}
	if cfg.Logger == "capnslog" {
		fmt.Fprintf(os.Stderr, "[WARNING] Deprecated '--logger=%s' flag is set; use '--logger=zap' flag instead\n", cfg.Logger)
	}
	if cfg.LogPkgLevels != "" {
		fmt.Fprintf(os.Stderr, "[WARNING] Deprecated '--log-package-levels=%s' flag is set; use '--logger=zap' flag instead\n", cfg.LogPkgLevels)
	}

	switch cfg.Logger {
	case "capnslog": // TODO: deprecate this in v3.5
		cfg.ClientTLSInfo.HandshakeFailure = logTLSHandshakeFailure
		cfg.PeerTLSInfo.HandshakeFailure = logTLSHandshakeFailure

		if cfg.Debug {
			capnslog.SetGlobalLogLevel(capnslog.DEBUG)
			grpc.EnableTracing = true
			// enable info, warning, error
			grpclog.SetLoggerV2(grpclog.NewLoggerV2(os.Stderr, os.Stderr, os.Stderr))
		} else {
			capnslog.SetGlobalLogLevel(logutil.ConvertToCapnslogLogLevel(cfg.LogLevel))
			// only discard info
			grpclog.SetLoggerV2(grpclog.NewLoggerV2(ioutil.Discard, os.Stderr, os.Stderr))
		}

		// TODO: deprecate with "capnslog"
		if cfg.LogPkgLevels != "" {
			repoLog := capnslog.MustRepoLogger("go.etcd.io/etcd")
			settings, err := repoLog.ParseLogLevelConfig(cfg.LogPkgLevels)
			if err != nil {
				plog.Warningf("couldn't parse log level string: %s, continuing with default levels", err.Error())
				return nil
			}
			repoLog.SetLogLevel(settings)
		}

		if len(cfg.LogOutputs) != 1 {
			return fmt.Errorf("--logger=capnslog supports only 1 value in '--log-outputs', got %q", cfg.LogOutputs)
		}
		// capnslog initially SetFormatter(NewDefaultFormatter(os.Stderr))
		// where NewDefaultFormatter returns NewJournaldFormatter when syscall.Getppid() == 1
		// specify 'stdout' or 'stderr' to skip journald logging even when running under systemd
		output := cfg.LogOutputs[0]
		switch output {
		case StdErrLogOutput:
			capnslog.SetFormatter(capnslog.NewPrettyFormatter(os.Stderr, cfg.Debug))
		case StdOutLogOutput:
			capnslog.SetFormatter(capnslog.NewPrettyFormatter(os.Stdout, cfg.Debug))
		case DefaultLogOutput:
		default:
			return fmt.Errorf("unknown log-output %q (only supports %q, %q, %q)", output, DefaultLogOutput, StdErrLogOutput, StdOutLogOutput)
		}

	case "zap":
		if len(cfg.LogOutputs) == 0 {
			cfg.LogOutputs = []string{DefaultLogOutput}
		}
		if len(cfg.LogOutputs) > 1 {
			for _, v := range cfg.LogOutputs {
				if v == DefaultLogOutput {
					return fmt.Errorf("multi logoutput for %q is not supported yet", DefaultLogOutput)
				}
			}
		}

		outputPaths, errOutputPaths := make([]string, 0), make([]string, 0)
		isJournal := false
		for _, v := range cfg.LogOutputs {
			switch v {
			case DefaultLogOutput:
				outputPaths = append(outputPaths, StdErrLogOutput)
				errOutputPaths = append(errOutputPaths, StdErrLogOutput)

			case JournalLogOutput:
				isJournal = true

			case StdErrLogOutput:
				outputPaths = append(outputPaths, StdErrLogOutput)
				errOutputPaths = append(errOutputPaths, StdErrLogOutput)

			case StdOutLogOutput:
				outputPaths = append(outputPaths, StdOutLogOutput)
				errOutputPaths = append(errOutputPaths, StdOutLogOutput)

			default:
				outputPaths = append(outputPaths, v)
				errOutputPaths = append(errOutputPaths, v)
			}
		}

		if !isJournal {
			copied := logutil.DefaultZapLoggerConfig
			copied.OutputPaths = outputPaths
			copied.ErrorOutputPaths = errOutputPaths
			copied = logutil.MergeOutputPaths(copied)
			copied.Level = zap.NewAtomicLevelAt(logutil.ConvertToZapLevel(cfg.LogLevel))
			if cfg.Debug || cfg.LogLevel == "debug" {
				// enable tracing even when "--debug --log-level info"
				// in order to keep backward compatibility with <= v3.3
				// TODO: remove "Debug" check in v3.5
				grpc.EnableTracing = true
			}
			if cfg.ZapLoggerBuilder == nil {
				cfg.ZapLoggerBuilder = func(c *Config) error {
					var err error
					c.logger, err = copied.Build()
					if err != nil {
						return err
					}
					c.loggerMu.Lock()
					defer c.loggerMu.Unlock()
					c.loggerConfig = &copied
					c.loggerCore = nil
					c.loggerWriteSyncer = nil
					grpcLogOnce.Do(func() {
						// debug true, enable info, warning, error
						// debug false, only discard info
						var gl grpclog.LoggerV2
						gl, err = logutil.NewGRPCLoggerV2(copied)
						if err == nil {
							grpclog.SetLoggerV2(gl)
						}
					})
					return nil
				}
			}
		} else {
			if len(cfg.LogOutputs) > 1 {
				for _, v := range cfg.LogOutputs {
					if v != DefaultLogOutput {
						return fmt.Errorf("running with systemd/journal but other '--log-outputs' values (%q) are configured with 'default'; override 'default' value with something else", cfg.LogOutputs)
					}
				}
			}

			// use stderr as fallback
			syncer, lerr := getJournalWriteSyncer()
			if lerr != nil {
				return lerr
			}

			lvl := zap.NewAtomicLevelAt(logutil.ConvertToZapLevel(cfg.LogLevel))
			if cfg.Debug || cfg.LogLevel == "debug" {
				// enable tracing even when "--debug --log-level info"
				// in order to keep backward compatibility with <= v3.3
				// TODO: remove "Debug" check in v3.5
				grpc.EnableTracing = true
			}

			// WARN: do not change field names in encoder config
			// journald logging writer assumes field names of "level" and "caller"
			cr := zapcore.NewCore(
				zapcore.NewJSONEncoder(logutil.DefaultZapLoggerConfig.EncoderConfig),
				syncer,
				lvl,
			)
			if cfg.ZapLoggerBuilder == nil {
				cfg.ZapLoggerBuilder = func(c *Config) error {
					c.logger = zap.New(cr, zap.AddCaller(), zap.ErrorOutput(syncer))
					c.loggerMu.Lock()
					defer c.loggerMu.Unlock()
					c.loggerConfig = nil
					c.loggerCore = cr
					c.loggerWriteSyncer = syncer

					grpcLogOnce.Do(func() {
						grpclog.SetLoggerV2(logutil.NewGRPCLoggerV2FromZapCore(cr, syncer))
					})
					return nil
				}
			}
		}

		err := cfg.ZapLoggerBuilder(cfg)
		if err != nil {
			return err
		}

		logTLSHandshakeFailure := func(conn *tls.Conn, err error) {
			state := conn.ConnectionState()
			remoteAddr := conn.RemoteAddr().String()
			serverName := state.ServerName
			if len(state.PeerCertificates) > 0 {
				cert := state.PeerCertificates[0]
				ips := make([]string, len(cert.IPAddresses))
				for i := range cert.IPAddresses {
					ips[i] = cert.IPAddresses[i].String()
				}
				cfg.logger.Warn(
					"rejected connection",
					zap.String("remote-addr", remoteAddr),
					zap.String("server-name", serverName),
					zap.Strings("ip-addresses", ips),
					zap.Strings("dns-names", cert.DNSNames),
					zap.Error(err),
				)
			} else {
				cfg.logger.Warn(
					"rejected connection",
					zap.String("remote-addr", remoteAddr),
					zap.String("server-name", serverName),
					zap.Error(err),
				)
			}
		}
		cfg.ClientTLSInfo.HandshakeFailure = logTLSHandshakeFailure
		cfg.PeerTLSInfo.HandshakeFailure = logTLSHandshakeFailure

	default:
		return fmt.Errorf("unknown logger option %q", cfg.Logger)
	}

	return nil
}

// NewZapCoreLoggerBuilder generates a zap core logger builder.
func NewZapCoreLoggerBuilder(lg *zap.Logger, cr zapcore.Core, syncer zapcore.WriteSyncer) func(*Config) error {
	return func(cfg *Config) error {
		cfg.loggerMu.Lock()
		defer cfg.loggerMu.Unlock()
		cfg.logger = lg
		cfg.loggerConfig = nil
		cfg.loggerCore = cr
		cfg.loggerWriteSyncer = syncer

		grpcLogOnce.Do(func() {
			grpclog.SetLoggerV2(logutil.NewGRPCLoggerV2FromZapCore(cr, syncer))
		})
		return nil
	}
}
