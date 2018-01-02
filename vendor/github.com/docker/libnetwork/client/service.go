package client

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"text/tabwriter"

	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/stringid"
	flag "github.com/docker/libnetwork/client/mflag"
	"github.com/docker/libnetwork/netutils"
)

var (
	serviceCommands = []command{
		{"publish", "Publish a service"},
		{"unpublish", "Remove a service"},
		{"attach", "Attach a backend (container) to the service"},
		{"detach", "Detach the backend from the service"},
		{"ls", "Lists all services"},
		{"info", "Display information about a service"},
	}
)

func lookupServiceID(cli *NetworkCli, nwName, svNameID string) (string, error) {
	// Sanity Check
	obj, _, err := readBody(cli.call("GET", fmt.Sprintf("/networks?name=%s", nwName), nil, nil))
	if err != nil {
		return "", err
	}
	var nwList []networkResource
	if err = json.Unmarshal(obj, &nwList); err != nil {
		return "", err
	}
	if len(nwList) == 0 {
		return "", fmt.Errorf("Network %s does not exist", nwName)
	}

	if nwName == "" {
		obj, _, err := readBody(cli.call("GET", "/networks/"+nwList[0].ID, nil, nil))
		if err != nil {
			return "", err
		}
		networkResource := &networkResource{}
		if err := json.NewDecoder(bytes.NewReader(obj)).Decode(networkResource); err != nil {
			return "", err
		}
		nwName = networkResource.Name
	}

	// Query service by name
	obj, statusCode, err := readBody(cli.call("GET", fmt.Sprintf("/services?name=%s", svNameID), nil, nil))
	if err != nil {
		return "", err
	}

	if statusCode != http.StatusOK {
		return "", fmt.Errorf("name query failed for %s due to: (%d) %s", svNameID, statusCode, string(obj))
	}

	var list []*serviceResource
	if err = json.Unmarshal(obj, &list); err != nil {
		return "", err
	}
	for _, sr := range list {
		if sr.Network == nwName {
			return sr.ID, nil
		}
	}

	// Query service by Partial-id (this covers full id as well)
	obj, statusCode, err = readBody(cli.call("GET", fmt.Sprintf("/services?partial-id=%s", svNameID), nil, nil))
	if err != nil {
		return "", err
	}

	if statusCode != http.StatusOK {
		return "", fmt.Errorf("partial-id match query failed for %s due to: (%d) %s", svNameID, statusCode, string(obj))
	}

	if err = json.Unmarshal(obj, &list); err != nil {
		return "", err
	}
	for _, sr := range list {
		if sr.Network == nwName {
			return sr.ID, nil
		}
	}

	return "", fmt.Errorf("Service %s not found on network %s", svNameID, nwName)
}

func lookupContainerID(cli *NetworkCli, cnNameID string) (string, error) {
	// Container is a Docker resource, ask docker about it.
	// In case of connecton error, we assume we are running in dnet and return whatever was passed to us
	obj, _, err := readBody(cli.call("GET", fmt.Sprintf("/containers/%s/json", cnNameID), nil, nil))
	if err != nil {
		// We are probably running outside of docker
		return cnNameID, nil
	}

	var x map[string]interface{}
	err = json.Unmarshal(obj, &x)
	if err != nil {
		return "", err
	}
	if iid, ok := x["Id"]; ok {
		if id, ok := iid.(string); ok {
			return id, nil
		}
		return "", errors.New("Unexpected data type for container ID in json response")
	}
	return "", errors.New("Cannot find container ID in json response")
}

func lookupSandboxID(cli *NetworkCli, containerID string) (string, error) {
	obj, _, err := readBody(cli.call("GET", fmt.Sprintf("/sandboxes?partial-container-id=%s", containerID), nil, nil))
	if err != nil {
		return "", err
	}

	var sandboxList []SandboxResource
	err = json.Unmarshal(obj, &sandboxList)
	if err != nil {
		return "", err
	}

	if len(sandboxList) == 0 {
		return "", fmt.Errorf("cannot find sandbox for container: %s", containerID)
	}

	return sandboxList[0].ID, nil
}

// CmdService handles the service UI
func (cli *NetworkCli) CmdService(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "service", "COMMAND [OPTIONS] [arg...]", serviceUsage(chain), false)
	cmd.Require(flag.Min, 1)
	err := cmd.ParseFlags(args, true)
	if err == nil {
		cmd.Usage()
		return fmt.Errorf("Invalid command : %v", args)
	}
	return err
}

// Parse service name for "SERVICE[.NETWORK]" format
func parseServiceName(name string) (string, string) {
	s := strings.Split(name, ".")
	var sName, nName string
	if len(s) > 1 {
		nName = s[len(s)-1]
		sName = strings.Join(s[:len(s)-1], ".")
	} else {
		sName = s[0]
	}
	return sName, nName
}

// CmdServicePublish handles service create UI
func (cli *NetworkCli) CmdServicePublish(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "publish", "SERVICE[.NETWORK]", "Publish a new service on a network", false)
	flAlias := opts.NewListOpts(netutils.ValidateAlias)
	cmd.Var(&flAlias, []string{"-alias"}, "Add alias to self")
	cmd.Require(flag.Exact, 1)
	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}

	sn, nn := parseServiceName(cmd.Arg(0))
	sc := serviceCreate{Name: sn, Network: nn, MyAliases: flAlias.GetAll()}
	obj, _, err := readBody(cli.call("POST", "/services", sc, nil))
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

// CmdServiceUnpublish handles service delete UI
func (cli *NetworkCli) CmdServiceUnpublish(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "unpublish", "SERVICE[.NETWORK]", "Removes a service", false)
	force := cmd.Bool([]string{"f", "-force"}, false, "force unpublish service")
	cmd.Require(flag.Exact, 1)
	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}

	sn, nn := parseServiceName(cmd.Arg(0))
	serviceID, err := lookupServiceID(cli, nn, sn)
	if err != nil {
		return err
	}

	sd := serviceDelete{Name: sn, Force: *force}
	_, _, err = readBody(cli.call("DELETE", "/services/"+serviceID, sd, nil))

	return err
}

// CmdServiceLs handles service list UI
func (cli *NetworkCli) CmdServiceLs(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "ls", "SERVICE", "Lists all the services on a network", false)
	flNetwork := cmd.String([]string{"net", "-network"}, "", "Only show the services that are published on the specified network")
	quiet := cmd.Bool([]string{"q", "-quiet"}, false, "Only display numeric IDs")
	noTrunc := cmd.Bool([]string{"#notrunc", "-no-trunc"}, false, "Do not truncate the output")

	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}

	var obj []byte
	if *flNetwork == "" {
		obj, _, err = readBody(cli.call("GET", "/services", nil, nil))
	} else {
		obj, _, err = readBody(cli.call("GET", "/services?network="+*flNetwork, nil, nil))
	}
	if err != nil {
		return err
	}

	var serviceResources []serviceResource
	err = json.Unmarshal(obj, &serviceResources)
	if err != nil {
		fmt.Println(err)
		return err
	}

	wr := tabwriter.NewWriter(cli.out, 20, 1, 3, ' ', 0)
	// unless quiet (-q) is specified, print field titles
	if !*quiet {
		fmt.Fprintln(wr, "SERVICE ID\tNAME\tNETWORK\tCONTAINER\tSANDBOX")
	}

	for _, sr := range serviceResources {
		ID := sr.ID
		bkID, sbID, err := getBackendID(cli, ID)
		if err != nil {
			return err
		}
		if !*noTrunc {
			ID = stringid.TruncateID(ID)
			bkID = stringid.TruncateID(bkID)
			sbID = stringid.TruncateID(sbID)
		}
		if !*quiet {
			fmt.Fprintf(wr, "%s\t%s\t%s\t%s\t%s\n", ID, sr.Name, sr.Network, bkID, sbID)
		} else {
			fmt.Fprintln(wr, ID)
		}
	}
	wr.Flush()

	return nil
}

func getBackendID(cli *NetworkCli, servID string) (string, string, error) {
	var (
		obj []byte
		err error
		bk  string
		sb  string
	)

	if obj, _, err = readBody(cli.call("GET", "/services/"+servID+"/backend", nil, nil)); err == nil {
		var sr SandboxResource
		if err := json.NewDecoder(bytes.NewReader(obj)).Decode(&sr); err == nil {
			bk = sr.ContainerID
			sb = sr.ID
		} else {
			// Only print a message, don't make the caller cli fail for this
			fmt.Fprintf(cli.out, "Failed to retrieve backend list for service %s (%v)\n", servID, err)
		}
	}

	return bk, sb, err
}

// CmdServiceInfo handles service info UI
func (cli *NetworkCli) CmdServiceInfo(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "info", "SERVICE[.NETWORK]", "Displays detailed information about a service", false)
	cmd.Require(flag.Min, 1)

	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}

	sn, nn := parseServiceName(cmd.Arg(0))
	serviceID, err := lookupServiceID(cli, nn, sn)
	if err != nil {
		return err
	}

	obj, _, err := readBody(cli.call("GET", "/services/"+serviceID, nil, nil))
	if err != nil {
		return err
	}

	sr := &serviceResource{}
	if err := json.NewDecoder(bytes.NewReader(obj)).Decode(sr); err != nil {
		return err
	}

	fmt.Fprintf(cli.out, "Service Id: %s\n", sr.ID)
	fmt.Fprintf(cli.out, "\tName: %s\n", sr.Name)
	fmt.Fprintf(cli.out, "\tNetwork: %s\n", sr.Network)

	return nil
}

// CmdServiceAttach handles service attach UI
func (cli *NetworkCli) CmdServiceAttach(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "attach", "CONTAINER SERVICE[.NETWORK]", "Sets a container as a service backend", false)
	flAlias := opts.NewListOpts(netutils.ValidateAlias)
	cmd.Var(&flAlias, []string{"-alias"}, "Add alias for another container")
	cmd.Require(flag.Min, 2)
	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}

	containerID, err := lookupContainerID(cli, cmd.Arg(0))
	if err != nil {
		return err
	}

	sandboxID, err := lookupSandboxID(cli, containerID)
	if err != nil {
		return err
	}

	sn, nn := parseServiceName(cmd.Arg(1))
	serviceID, err := lookupServiceID(cli, nn, sn)
	if err != nil {
		return err
	}

	nc := serviceAttach{SandboxID: sandboxID, Aliases: flAlias.GetAll()}

	_, _, err = readBody(cli.call("POST", "/services/"+serviceID+"/backend", nc, nil))

	return err
}

// CmdServiceDetach handles service detach UI
func (cli *NetworkCli) CmdServiceDetach(chain string, args ...string) error {
	cmd := cli.Subcmd(chain, "detach", "CONTAINER SERVICE", "Removes a container from service backend", false)
	cmd.Require(flag.Min, 2)
	err := cmd.ParseFlags(args, true)
	if err != nil {
		return err
	}

	sn, nn := parseServiceName(cmd.Arg(1))
	containerID, err := lookupContainerID(cli, cmd.Arg(0))
	if err != nil {
		return err
	}

	sandboxID, err := lookupSandboxID(cli, containerID)
	if err != nil {
		return err
	}

	serviceID, err := lookupServiceID(cli, nn, sn)
	if err != nil {
		return err
	}

	_, _, err = readBody(cli.call("DELETE", "/services/"+serviceID+"/backend/"+sandboxID, nil, nil))
	if err != nil {
		return err
	}
	return nil
}

func serviceUsage(chain string) string {
	help := "Commands:\n"

	for _, cmd := range serviceCommands {
		help += fmt.Sprintf("    %-10.10s%s\n", cmd.name, cmd.description)
	}

	help += fmt.Sprintf("\nRun '%s service COMMAND --help' for more information on a command.", chain)
	return help
}
