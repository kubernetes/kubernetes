// +build experimental

package client

import (
	"os"

	nwclient "github.com/docker/libnetwork/client"
)

func (cli *DockerCli) CmdNetwork(args ...string) error {
	nCli := nwclient.NewNetworkCli(cli.out, cli.err, nwclient.CallFunc(cli.callWrapper))
	args = append([]string{"network"}, args...)
	return nCli.Cmd(os.Args[0], args...)
}
