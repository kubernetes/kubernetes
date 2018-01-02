package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"text/tabwriter"

	"github.com/docker/docker/pkg/stringid"
	flag "github.com/docker/libnetwork/client/mflag"
	"github.com/docker/libnetwork/netlabel"
)

type command struct {
	name        string
	description string
}

var (
	networkCommands = []command{
		{"create", "Create a network"},
		{"rm", "Remove a network"},
		{"ls", "List all networks"},
		{"info", "Display information of a network"},
	}
)

// CmdNetwork handles the root Network UI
func (cli *NetworkCli) CmdNetwork(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "network", "COMMAND [OPTIONS] [arg...]", networkUsage(chain), false)
	cmd.Require(flag.Min, 1)
	err := cmd.ParseFlags(args, true)
	if err == nil {
		cmd.Usage()
		return fmt.Errorf("invalid command : %v", args)
	}
	return err
}

// CmdNetworkCreate handles Network Create UI
func (cli *NetworkCli) CmdNetworkCreate(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "create", "NETWORK-NAME", "Creates a new network with a name specified by the user", false)
	flDriver := cmd.String([]string{"d", "-driver"}, "", "Driver to manage the Network")
	flID := cmd.String([]string{"-id"}, "", "Network ID string")
	flOpts := cmd.String([]string{"o", "-opt"}, "", "Network options")
	flInternal := cmd.Bool([]string{"-internal"}, false, "Config the network to be internal")
	flIPv6 := cmd.Bool([]string{"-ipv6"}, false, "Enable IPv6 on the network")
	flSubnet := cmd.String([]string{"-subnet"}, "", "Subnet option")
	flRange := cmd.String([]string{"-ip-range"}, "", "Range option")

	cmd.Require(flag.Exact, 1)
	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}
	networkOpts := make(map[string]string)
	if *flInternal {
		networkOpts[netlabel.Internal] = "true"
	}
	if *flIPv6 {
		networkOpts[netlabel.EnableIPv6] = "true"
	}

	driverOpts := make(map[string]string)
	if *flOpts != "" {
		opts := strings.Split(*flOpts, ",")
		for _, opt := range opts {
			driverOpts[netlabel.Key(opt)] = netlabel.Value(opt)
		}
	}

	var icList []ipamConf
	if *flSubnet != "" {
		ic := ipamConf{
			PreferredPool: *flSubnet,
		}

		if *flRange != "" {
			ic.SubPool = *flRange
		}

		icList = append(icList, ic)
	}

	// Construct network create request body
	nc := networkCreate{Name: cmd.Arg(0), NetworkType: *flDriver, ID: *flID, IPv4Conf: icList, DriverOpts: driverOpts, NetworkOpts: networkOpts}
	obj, _, err := readBody(cli.call("POST", "/networks", nc, nil))
	if err != nil {
		return err
	}
	var replyID string
	err = json.Unmarshal(obj, &replyID)
	if err != nil {
		return err
	}
	fmt.Fprintf(cli.out, "%s\n", replyID)
	return nil
}

// CmdNetworkRm handles Network Delete UI
func (cli *NetworkCli) CmdNetworkRm(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "rm", "NETWORK", "Deletes a network", false)
	cmd.Require(flag.Exact, 1)
	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}
	id, err := lookupNetworkID(cli, cmd.Arg(0))
	if err != nil {
		return err
	}
	_, _, err = readBody(cli.call("DELETE", "/networks/"+id, nil, nil))
	if err != nil {
		return err
	}
	return nil
}

// CmdNetworkLs handles Network List UI
func (cli *NetworkCli) CmdNetworkLs(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "ls", "", "Lists all the networks created by the user", false)
	quiet := cmd.Bool([]string{"q", "-quiet"}, false, "Only display numeric IDs")
	noTrunc := cmd.Bool([]string{"#notrunc", "-no-trunc"}, false, "Do not truncate the output")
	nLatest := cmd.Bool([]string{"l", "-latest"}, false, "Show the latest network created")
	last := cmd.Int([]string{"n"}, -1, "Show n last created networks")
	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}
	obj, _, err := readBody(cli.call("GET", "/networks", nil, nil))
	if err != nil {
		return err
	}
	if *last == -1 && *nLatest {
		*last = 1
	}

	var networkResources []networkResource
	err = json.Unmarshal(obj, &networkResources)
	if err != nil {
		return err
	}

	wr := tabwriter.NewWriter(cli.out, 20, 1, 3, ' ', 0)

	// unless quiet (-q) is specified, print field titles
	if !*quiet {
		fmt.Fprintln(wr, "NETWORK ID\tNAME\tTYPE")
	}

	for _, networkResource := range networkResources {
		ID := networkResource.ID
		netName := networkResource.Name
		if !*noTrunc {
			ID = stringid.TruncateID(ID)
		}
		if *quiet {
			fmt.Fprintln(wr, ID)
			continue
		}
		netType := networkResource.Type
		fmt.Fprintf(wr, "%s\t%s\t%s\t",
			ID,
			netName,
			netType)
		fmt.Fprint(wr, "\n")
	}
	wr.Flush()
	return nil
}

// CmdNetworkInfo handles Network Info UI
func (cli *NetworkCli) CmdNetworkInfo(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "info", "NETWORK", "Displays detailed information on a network", false)
	cmd.Require(flag.Exact, 1)
	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}

	id, err := lookupNetworkID(cli, cmd.Arg(0))
	if err != nil {
		return err
	}

	obj, _, err := readBody(cli.call("GET", "/networks/"+id, nil, nil))
	if err != nil {
		return err
	}
	networkResource := &networkResource{}
	if err := json.NewDecoder(bytes.NewReader(obj)).Decode(networkResource); err != nil {
		return err
	}
	fmt.Fprintf(cli.out, "Network Id: %s\n", networkResource.ID)
	fmt.Fprintf(cli.out, "Name: %s\n", networkResource.Name)
	fmt.Fprintf(cli.out, "Type: %s\n", networkResource.Type)
	if networkResource.Services != nil {
		for _, serviceResource := range networkResource.Services {
			fmt.Fprintf(cli.out, "  Service Id: %s\n", serviceResource.ID)
			fmt.Fprintf(cli.out, "\tName: %s\n", serviceResource.Name)
		}
	}

	return nil
}

// Helper function to predict if a string is a name or id or partial-id
// This provides a best-effort mechanism to identify an id with the help of GET Filter APIs
// Being a UI, its most likely that name will be used by the user, which is used to lookup
// the corresponding ID. If ID is not found, this function will assume that the passed string
// is an ID by itself.

func lookupNetworkID(cli *NetworkCli, nameID string) (string, error) {
	obj, statusCode, err := readBody(cli.call("GET", "/networks?name="+nameID, nil, nil))
	if err != nil {
		return "", err
	}

	if statusCode != http.StatusOK {
		return "", fmt.Errorf("name query failed for %s due to : statuscode(%d) %v", nameID, statusCode, string(obj))
	}

	var list []*networkResource
	err = json.Unmarshal(obj, &list)
	if err != nil {
		return "", err
	}
	if len(list) > 0 {
		// name query filter will always return a single-element collection
		return list[0].ID, nil
	}

	// Check for Partial-id
	obj, statusCode, err = readBody(cli.call("GET", "/networks?partial-id="+nameID, nil, nil))
	if err != nil {
		return "", err
	}

	if statusCode != http.StatusOK {
		return "", fmt.Errorf("partial-id match query failed for %s due to : statuscode(%d) %v", nameID, statusCode, string(obj))
	}

	err = json.Unmarshal(obj, &list)
	if err != nil {
		return "", err
	}
	if len(list) == 0 {
		return "", fmt.Errorf("resource not found %s", nameID)
	}
	if len(list) > 1 {
		return "", fmt.Errorf("multiple Networks matching the partial identifier (%s). Please use full identifier", nameID)
	}
	return list[0].ID, nil
}

func networkUsage(chain string) string {
	help := "Commands:\n"

	for _, cmd := range networkCommands {
		help += fmt.Sprintf("  %-25.25s%s\n", cmd.name, cmd.description)
	}

	help += fmt.Sprintf("\nRun '%s network COMMAND --help' for more information on a command.", chain)
	return help
}
