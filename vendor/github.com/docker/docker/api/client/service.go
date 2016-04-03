// +build experimental

package client

import (
	"os"

	nwclient "github.com/docker/libnetwork/client"
)

func (cli *DockerCli) CmdService(args ...string) error {
	nCli := nwclient.NewNetworkCli(cli.out, cli.err, nwclient.CallFunc(cli.callWrapper))
	args = append([]string{"service"}, args...)
	return nCli.Cmd(os.Args[0], args...)
}
