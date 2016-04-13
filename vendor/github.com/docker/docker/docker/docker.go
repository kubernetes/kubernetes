package main

import (
	"crypto/tls"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/api/client"
	"github.com/docker/docker/autogen/dockerversion"
	"github.com/docker/docker/cliconfig"
	"github.com/docker/docker/opts"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/docker/pkg/term"
	"github.com/docker/docker/pkg/tlsconfig"
	"github.com/docker/docker/utils"
)

const (
	defaultTrustKeyFile = "key.json"
	defaultCaFile       = "ca.pem"
	defaultKeyFile      = "key.pem"
	defaultCertFile     = "cert.pem"
)

func main() {
	if reexec.Init() {
		return
	}

	// Set terminal emulation based on platform as required.
	stdin, stdout, stderr := term.StdStreams()

	initLogging(stderr)

	flag.Parse()
	// FIXME: validate daemon flags here

	if *flVersion {
		showVersion()
		return
	}

	if *flConfigDir != "" {
		cliconfig.SetConfigDir(*flConfigDir)
	}

	if *flLogLevel != "" {
		lvl, err := logrus.ParseLevel(*flLogLevel)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Unable to parse logging level: %s\n", *flLogLevel)
			os.Exit(1)
		}
		setLogLevel(lvl)
	} else {
		setLogLevel(logrus.InfoLevel)
	}

	if *flDebug {
		os.Setenv("DEBUG", "1")
		setLogLevel(logrus.DebugLevel)
	}

	if len(flHosts) == 0 {
		defaultHost := os.Getenv("DOCKER_HOST")
		if defaultHost == "" || *flDaemon {
			if runtime.GOOS != "windows" {
				// If we do not have a host, default to unix socket
				defaultHost = fmt.Sprintf("unix://%s", opts.DefaultUnixSocket)
			} else {
				// If we do not have a host, default to TCP socket on Windows
				defaultHost = fmt.Sprintf("tcp://%s:%d", opts.DefaultHTTPHost, opts.DefaultHTTPPort)
			}
		}
		defaultHost, err := opts.ValidateHost(defaultHost)
		if err != nil {
			if *flDaemon {
				logrus.Fatal(err)
			} else {
				fmt.Fprint(os.Stderr, err)
			}
			os.Exit(1)
		}
		flHosts = append(flHosts, defaultHost)
	}

	setDefaultConfFlag(flTrustKey, defaultTrustKeyFile)

	// Regardless of whether the user sets it to true or false, if they
	// specify --tlsverify at all then we need to turn on tls
	// *flTlsVerify can be true even if not set due to DOCKER_TLS_VERIFY env var, so we need to check that here as well
	if flag.IsSet("-tlsverify") || *flTLSVerify {
		*flTLS = true
	}

	if *flDaemon {
		if *flHelp {
			flag.Usage()
			return
		}
		mainDaemon()
		return
	}

	// From here on, we assume we're a client, not a server.

	if len(flHosts) > 1 {
		fmt.Fprintf(os.Stderr, "Please specify only one -H")
		os.Exit(0)
	}
	protoAddrParts := strings.SplitN(flHosts[0], "://", 2)

	var tlsConfig *tls.Config
	if *flTLS {
		tlsOptions.InsecureSkipVerify = !*flTLSVerify
		if !flag.IsSet("-tlscert") {
			if _, err := os.Stat(tlsOptions.CertFile); os.IsNotExist(err) {
				tlsOptions.CertFile = ""
			}
		}
		if !flag.IsSet("-tlskey") {
			if _, err := os.Stat(tlsOptions.KeyFile); os.IsNotExist(err) {
				tlsOptions.KeyFile = ""
			}
		}
		var err error
		tlsConfig, err = tlsconfig.Client(tlsOptions)
		if err != nil {
			fmt.Fprintln(stderr, err)
			os.Exit(1)
		}
	}
	cli := client.NewDockerCli(stdin, stdout, stderr, *flTrustKey, protoAddrParts[0], protoAddrParts[1], tlsConfig)

	if err := cli.Cmd(flag.Args()...); err != nil {
		if sterr, ok := err.(client.StatusError); ok {
			if sterr.Status != "" {
				fmt.Fprintln(cli.Err(), sterr.Status)
				os.Exit(1)
			}
			os.Exit(sterr.StatusCode)
		}
		fmt.Fprintln(cli.Err(), err)
		os.Exit(1)
	}
}

func showVersion() {
	if utils.ExperimentalBuild() {
		fmt.Printf("Docker version %s, build %s, experimental\n", dockerversion.VERSION, dockerversion.GITCOMMIT)
	} else {
		fmt.Printf("Docker version %s, build %s\n", dockerversion.VERSION, dockerversion.GITCOMMIT)
	}
}
