package main

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"

	"github.com/codegangsta/cli"
	"github.com/docker/docker/pkg/term"
	"github.com/docker/libnetwork/client"
)

var (
	containerCreateCommand = cli.Command{
		Name:   "create",
		Usage:  "Create a container",
		Action: runContainerCreate,
	}

	containerRmCommand = cli.Command{
		Name:   "rm",
		Usage:  "Remove a container",
		Action: runContainerRm,
	}

	containerCommands = []cli.Command{
		containerCreateCommand,
		containerRmCommand,
	}

	dnetCommands = []cli.Command{
		createDockerCommand("network"),
		createDockerCommand("service"),
		{
			Name:        "container",
			Usage:       "Container management commands",
			Subcommands: containerCommands,
		},
	}
)

func runContainerCreate(c *cli.Context) {
	if len(c.Args()) == 0 {
		fmt.Println("Please provide container id argument")
		os.Exit(1)
	}

	sc := client.SandboxCreate{ContainerID: c.Args()[0]}
	obj, _, err := readBody(epConn.httpCall("POST", "/sandboxes", sc, nil))
	if err != nil {
		fmt.Printf("POST failed during create container: %v\n", err)
		os.Exit(1)
	}

	var replyID string
	err = json.Unmarshal(obj, &replyID)
	if err != nil {
		fmt.Printf("Unmarshall of response failed during create container: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("%s\n", replyID)

}

func runContainerRm(c *cli.Context) {
	var sbList []*client.SandboxResource

	if len(c.Args()) == 0 {
		fmt.Println("Please provide container id argument")
		os.Exit(1)
	}

	obj, _, err := readBody(epConn.httpCall("GET", "/sandboxes?partial-container-id="+c.Args()[0], nil, nil))
	if err != nil {
		fmt.Printf("GET failed during container id lookup: %v\n", err)
		os.Exit(1)
	}

	err = json.Unmarshal(obj, &sbList)
	if err != nil {
		fmt.Printf("Unmarshall of container id lookup response failed: %v", err)
		os.Exit(1)
	}

	if len(sbList) == 0 {
		fmt.Printf("No sandbox for container %s found\n", c.Args()[0])
		os.Exit(1)
	}

	_, _, err = readBody(epConn.httpCall("DELETE", "/sandboxes/"+sbList[0].ID, nil, nil))
	if err != nil {
		fmt.Printf("DELETE of sandbox id %s failed: %v", sbList[0].ID, err)
		os.Exit(1)
	}
}

func runDockerCommand(c *cli.Context, cmd string) {
	_, stdout, stderr := term.StdStreams()
	oldcli := client.NewNetworkCli(stdout, stderr, epConn.httpCall)
	var args []string
	args = append(args, cmd)
	if c.Bool("h") {
		args = append(args, "--help")
	} else {
		args = append(args, c.Args()...)
	}
	if err := oldcli.Cmd("dnet", args...); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func createDockerCommand(cmd string) cli.Command {
	return cli.Command{
		Name:            cmd,
		Usage:           fmt.Sprintf("%s management commands", cmd),
		SkipFlagParsing: true,
		Action: func(c *cli.Context) {
			runDockerCommand(c, cmd)
		},
		Subcommands: []cli.Command{
			{
				Name:  "h, -help",
				Usage: fmt.Sprintf("%s help", cmd),
			},
		},
	}
}

func readBody(stream io.ReadCloser, hdr http.Header, statusCode int, err error) ([]byte, int, error) {
	if stream != nil {
		defer stream.Close()
	}
	if err != nil {
		return nil, statusCode, err
	}
	body, err := ioutil.ReadAll(stream)
	if err != nil {
		return nil, -1, err
	}
	return body, statusCode, nil
}
