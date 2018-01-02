// Copyright 2015 The etcd Authors
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

package command

import (
	"crypto/tls"
	"errors"
	"io"
	"io/ioutil"
	"strings"
	"time"

	"github.com/bgentry/speakeasy"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/flags"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/spf13/cobra"
)

// GlobalFlags are flags that defined globally
// and are inherited to all sub-commands.
type GlobalFlags struct {
	Insecure           bool
	InsecureSkipVerify bool
	Endpoints          []string
	DialTimeout        time.Duration
	CommandTimeOut     time.Duration

	TLS transport.TLSInfo

	OutputFormat string
	IsHex        bool

	User string
}

type secureCfg struct {
	cert   string
	key    string
	cacert string

	insecureTransport  bool
	insecureSkipVerify bool
}

type authCfg struct {
	username string
	password string
}

var display printer = &simplePrinter{}

func initDisplayFromCmd(cmd *cobra.Command) {
	isHex, err := cmd.Flags().GetBool("hex")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	outputType, err := cmd.Flags().GetString("write-out")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	if display = NewPrinter(outputType, isHex); display == nil {
		ExitWithError(ExitBadFeature, errors.New("unsupported output format"))
	}
}

func mustClientFromCmd(cmd *cobra.Command) *clientv3.Client {
	flags.SetPflagsFromEnv("ETCDCTL", cmd.InheritedFlags())

	endpoints, err := cmd.Flags().GetStringSlice("endpoints")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	dialTimeout := dialTimeoutFromCmd(cmd)
	sec := secureCfgFromCmd(cmd)
	auth := authCfgFromCmd(cmd)

	initDisplayFromCmd(cmd)

	return mustClient(endpoints, dialTimeout, sec, auth)
}

func mustClient(endpoints []string, dialTimeout time.Duration, scfg *secureCfg, acfg *authCfg) *clientv3.Client {
	cfg, err := newClientCfg(endpoints, dialTimeout, scfg, acfg)
	if err != nil {
		ExitWithError(ExitBadArgs, err)
	}

	client, err := clientv3.New(*cfg)
	if err != nil {
		ExitWithError(ExitBadConnection, err)
	}

	return client
}

func newClientCfg(endpoints []string, dialTimeout time.Duration, scfg *secureCfg, acfg *authCfg) (*clientv3.Config, error) {
	// set tls if any one tls option set
	var cfgtls *transport.TLSInfo
	tlsinfo := transport.TLSInfo{}
	if scfg.cert != "" {
		tlsinfo.CertFile = scfg.cert
		cfgtls = &tlsinfo
	}

	if scfg.key != "" {
		tlsinfo.KeyFile = scfg.key
		cfgtls = &tlsinfo
	}

	if scfg.cacert != "" {
		tlsinfo.CAFile = scfg.cacert
		cfgtls = &tlsinfo
	}

	cfg := &clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	}
	if cfgtls != nil {
		clientTLS, err := cfgtls.ClientConfig()
		if err != nil {
			return nil, err
		}
		cfg.TLS = clientTLS
	}
	// if key/cert is not given but user wants secure connection, we
	// should still setup an empty tls configuration for gRPC to setup
	// secure connection.
	if cfg.TLS == nil && !scfg.insecureTransport {
		cfg.TLS = &tls.Config{}
	}

	// If the user wants to skip TLS verification then we should set
	// the InsecureSkipVerify flag in tls configuration.
	if scfg.insecureSkipVerify && cfg.TLS != nil {
		cfg.TLS.InsecureSkipVerify = true
	}

	if acfg != nil {
		cfg.Username = acfg.username
		cfg.Password = acfg.password
	}

	return cfg, nil
}

func argOrStdin(args []string, stdin io.Reader, i int) (string, error) {
	if i < len(args) {
		return args[i], nil
	}
	bytes, err := ioutil.ReadAll(stdin)
	if string(bytes) == "" || err != nil {
		return "", errors.New("no available argument and stdin")
	}
	return string(bytes), nil
}

func dialTimeoutFromCmd(cmd *cobra.Command) time.Duration {
	dialTimeout, err := cmd.Flags().GetDuration("dial-timeout")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	return dialTimeout
}

func secureCfgFromCmd(cmd *cobra.Command) *secureCfg {
	cert, key, cacert := keyAndCertFromCmd(cmd)
	insecureTr := insecureTransportFromCmd(cmd)
	skipVerify := insecureSkipVerifyFromCmd(cmd)

	return &secureCfg{
		cert:   cert,
		key:    key,
		cacert: cacert,

		insecureTransport:  insecureTr,
		insecureSkipVerify: skipVerify,
	}
}

func insecureTransportFromCmd(cmd *cobra.Command) bool {
	insecureTr, err := cmd.Flags().GetBool("insecure-transport")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	return insecureTr
}

func insecureSkipVerifyFromCmd(cmd *cobra.Command) bool {
	skipVerify, err := cmd.Flags().GetBool("insecure-skip-tls-verify")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	return skipVerify
}

func keyAndCertFromCmd(cmd *cobra.Command) (cert, key, cacert string) {
	var err error
	if cert, err = cmd.Flags().GetString("cert"); err != nil {
		ExitWithError(ExitBadArgs, err)
	} else if cert == "" && cmd.Flags().Changed("cert") {
		ExitWithError(ExitBadArgs, errors.New("empty string is passed to --cert option"))
	}

	if key, err = cmd.Flags().GetString("key"); err != nil {
		ExitWithError(ExitBadArgs, err)
	} else if key == "" && cmd.Flags().Changed("key") {
		ExitWithError(ExitBadArgs, errors.New("empty string is passed to --key option"))
	}

	if cacert, err = cmd.Flags().GetString("cacert"); err != nil {
		ExitWithError(ExitBadArgs, err)
	} else if cacert == "" && cmd.Flags().Changed("cacert") {
		ExitWithError(ExitBadArgs, errors.New("empty string is passed to --cacert option"))
	}

	return cert, key, cacert
}

func authCfgFromCmd(cmd *cobra.Command) *authCfg {
	userFlag, err := cmd.Flags().GetString("user")
	if err != nil {
		ExitWithError(ExitBadArgs, err)
	}

	if userFlag == "" {
		return nil
	}

	var cfg authCfg

	splitted := strings.SplitN(userFlag, ":", 2)
	if len(splitted) < 2 {
		cfg.username = userFlag
		cfg.password, err = speakeasy.Ask("Password: ")
		if err != nil {
			ExitWithError(ExitError, err)
		}
	} else {
		cfg.username = splitted[0]
		cfg.password = splitted[1]
	}

	return &cfg
}
