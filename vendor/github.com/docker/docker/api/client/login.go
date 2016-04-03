package client

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/cliconfig"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/term"
	"github.com/docker/docker/registry"
)

// CmdLogin logs in or registers a user to a Docker registry service.
//
// If no server is specified, the user will be logged into or registered to the registry's index server.
//
// Usage: docker login SERVER
func (cli *DockerCli) CmdLogin(args ...string) error {
	cmd := cli.Subcmd("login", []string{"[SERVER]"}, "Register or log in to a Docker registry server, if no server is\nspecified \""+registry.INDEXSERVER+"\" is the default.", true)
	cmd.Require(flag.Max, 1)

	var username, password, email string

	cmd.StringVar(&username, []string{"u", "-username"}, "", "Username")
	cmd.StringVar(&password, []string{"p", "-password"}, "", "Password")
	cmd.StringVar(&email, []string{"e", "-email"}, "", "Email")

	cmd.ParseFlags(args, true)

	serverAddress := registry.INDEXSERVER
	if len(cmd.Args()) > 0 {
		serverAddress = cmd.Arg(0)
	}

	promptDefault := func(prompt string, configDefault string) {
		if configDefault == "" {
			fmt.Fprintf(cli.out, "%s: ", prompt)
		} else {
			fmt.Fprintf(cli.out, "%s (%s): ", prompt, configDefault)
		}
	}

	readInput := func(in io.Reader, out io.Writer) string {
		reader := bufio.NewReader(in)
		line, _, err := reader.ReadLine()
		if err != nil {
			fmt.Fprintln(out, err.Error())
			os.Exit(1)
		}
		return string(line)
	}

	authconfig, ok := cli.configFile.AuthConfigs[serverAddress]
	if !ok {
		authconfig = cliconfig.AuthConfig{}
	}

	if username == "" {
		promptDefault("Username", authconfig.Username)
		username = readInput(cli.in, cli.out)
		username = strings.Trim(username, " ")
		if username == "" {
			username = authconfig.Username
		}
	}
	// Assume that a different username means they may not want to use
	// the password or email from the config file, so prompt them
	if username != authconfig.Username {
		if password == "" {
			oldState, err := term.SaveState(cli.inFd)
			if err != nil {
				return err
			}
			fmt.Fprintf(cli.out, "Password: ")
			term.DisableEcho(cli.inFd, oldState)

			password = readInput(cli.in, cli.out)
			fmt.Fprint(cli.out, "\n")

			term.RestoreTerminal(cli.inFd, oldState)
			if password == "" {
				return fmt.Errorf("Error : Password Required")
			}
		}

		if email == "" {
			promptDefault("Email", authconfig.Email)
			email = readInput(cli.in, cli.out)
			if email == "" {
				email = authconfig.Email
			}
		}
	} else {
		// However, if they don't override the username use the
		// password or email from the cmd line if specified. IOW, allow
		// then to change/override them.  And if not specified, just
		// use what's in the config file
		if password == "" {
			password = authconfig.Password
		}
		if email == "" {
			email = authconfig.Email
		}
	}
	authconfig.Username = username
	authconfig.Password = password
	authconfig.Email = email
	authconfig.ServerAddress = serverAddress
	cli.configFile.AuthConfigs[serverAddress] = authconfig

	serverResp, err := cli.call("POST", "/auth", cli.configFile.AuthConfigs[serverAddress], nil)
	if serverResp.statusCode == 401 {
		delete(cli.configFile.AuthConfigs, serverAddress)
		if err2 := cli.configFile.Save(); err2 != nil {
			fmt.Fprintf(cli.out, "WARNING: could not save config file: %v\n", err2)
		}
		return err
	}
	if err != nil {
		return err
	}

	defer serverResp.body.Close()

	var response types.AuthResponse
	if err := json.NewDecoder(serverResp.body).Decode(&response); err != nil {
		// Upon error, remove entry
		delete(cli.configFile.AuthConfigs, serverAddress)
		return err
	}

	if err := cli.configFile.Save(); err != nil {
		return fmt.Errorf("Error saving config file: %v", err)
	}
	fmt.Fprintf(cli.out, "WARNING: login credentials saved in %s\n", cli.configFile.Filename())

	if response.Status != "" {
		fmt.Fprintf(cli.out, "%s\n", response.Status)
	}
	return nil
}
