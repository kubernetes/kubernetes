package client

import (
	"encoding/json"
	"fmt"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/httputils"
	"github.com/docker/docker/pkg/ioutils"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/units"
)

// CmdInfo displays system-wide information.
//
// Usage: docker info
func (cli *DockerCli) CmdInfo(args ...string) error {
	cmd := cli.Subcmd("info", nil, "Display system-wide information", true)
	cmd.Require(flag.Exact, 0)

	cmd.ParseFlags(args, true)

	serverResp, err := cli.call("GET", "/info", nil, nil)
	if err != nil {
		return err
	}

	defer serverResp.body.Close()

	info := &types.Info{}
	if err := json.NewDecoder(serverResp.body).Decode(info); err != nil {
		return fmt.Errorf("Error reading remote info: %v", err)
	}

	fmt.Fprintf(cli.out, "Containers: %d\n", info.Containers)
	fmt.Fprintf(cli.out, "Images: %d\n", info.Images)
	ioutils.FprintfIfNotEmpty(cli.out, "Storage Driver: %s\n", info.Driver)
	if info.DriverStatus != nil {
		for _, pair := range info.DriverStatus {
			fmt.Fprintf(cli.out, " %s: %s\n", pair[0], pair[1])
		}
	}
	ioutils.FprintfIfNotEmpty(cli.out, "Execution Driver: %s\n", info.ExecutionDriver)
	ioutils.FprintfIfNotEmpty(cli.out, "Logging Driver: %s\n", info.LoggingDriver)
	ioutils.FprintfIfNotEmpty(cli.out, "Kernel Version: %s\n", info.KernelVersion)
	ioutils.FprintfIfNotEmpty(cli.out, "Operating System: %s\n", info.OperatingSystem)
	fmt.Fprintf(cli.out, "CPUs: %d\n", info.NCPU)
	fmt.Fprintf(cli.out, "Total Memory: %s\n", units.BytesSize(float64(info.MemTotal)))
	ioutils.FprintfIfNotEmpty(cli.out, "Name: %s\n", info.Name)
	ioutils.FprintfIfNotEmpty(cli.out, "ID: %s\n", info.ID)

	if info.Debug {
		fmt.Fprintf(cli.out, "Debug mode (server): %v\n", info.Debug)
		fmt.Fprintf(cli.out, "File Descriptors: %d\n", info.NFd)
		fmt.Fprintf(cli.out, "Goroutines: %d\n", info.NGoroutines)
		fmt.Fprintf(cli.out, "System Time: %s\n", info.SystemTime)
		fmt.Fprintf(cli.out, "EventsListeners: %d\n", info.NEventsListener)
		fmt.Fprintf(cli.out, "Init SHA1: %s\n", info.InitSha1)
		fmt.Fprintf(cli.out, "Init Path: %s\n", info.InitPath)
		fmt.Fprintf(cli.out, "Docker Root Dir: %s\n", info.DockerRootDir)
	}

	ioutils.FprintfIfNotEmpty(cli.out, "Http Proxy: %s\n", info.HttpProxy)
	ioutils.FprintfIfNotEmpty(cli.out, "Https Proxy: %s\n", info.HttpsProxy)
	ioutils.FprintfIfNotEmpty(cli.out, "No Proxy: %s\n", info.NoProxy)

	if info.IndexServerAddress != "" {
		u := cli.configFile.AuthConfigs[info.IndexServerAddress].Username
		if len(u) > 0 {
			fmt.Fprintf(cli.out, "Username: %v\n", u)
			fmt.Fprintf(cli.out, "Registry: %v\n", info.IndexServerAddress)
		}
	}
	// Only output these warnings if the server supports these features
	if h, err := httputils.ParseServerHeader(serverResp.header.Get("Server")); err == nil {
		if h.OS != "windows" {
			if !info.MemoryLimit {
				fmt.Fprintf(cli.err, "WARNING: No memory limit support\n")
			}
			if !info.SwapLimit {
				fmt.Fprintf(cli.err, "WARNING: No swap limit support\n")
			}
			if !info.IPv4Forwarding {
				fmt.Fprintf(cli.err, "WARNING: IPv4 forwarding is disabled.\n")
			}
			if !info.BridgeNfIptables {
				fmt.Fprintf(cli.err, "WARNING: bridge-nf-call-iptables is disabled\n")
			}
			if !info.BridgeNfIp6tables {
				fmt.Fprintf(cli.err, "WARNING: bridge-nf-call-ip6tables is disabled\n")
			}
		}
	}

	if info.Labels != nil {
		fmt.Fprintln(cli.out, "Labels:")
		for _, attribute := range info.Labels {
			fmt.Fprintf(cli.out, " %s\n", attribute)
		}
	}

	if info.ExperimentalBuild {
		fmt.Fprintf(cli.out, "Experimental: true\n")
	}

	return nil
}
