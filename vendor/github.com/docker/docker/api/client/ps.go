package client

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/docker/docker/api"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/opts"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/parsers/filters"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/stringutils"
	"github.com/docker/docker/pkg/units"
)

// CmdPs outputs a list of Docker containers.
//
// Usage: docker ps [OPTIONS]
func (cli *DockerCli) CmdPs(args ...string) error {
	var (
		err error

		psFilterArgs = filters.Args{}
		v            = url.Values{}

		cmd      = cli.Subcmd("ps", nil, "List containers", true)
		quiet    = cmd.Bool([]string{"q", "-quiet"}, false, "Only display numeric IDs")
		size     = cmd.Bool([]string{"s", "-size"}, false, "Display total file sizes")
		all      = cmd.Bool([]string{"a", "-all"}, false, "Show all containers (default shows just running)")
		noTrunc  = cmd.Bool([]string{"#notrunc", "-no-trunc"}, false, "Don't truncate output")
		nLatest  = cmd.Bool([]string{"l", "-latest"}, false, "Show the latest created container, include non-running")
		since    = cmd.String([]string{"#sinceId", "#-since-id", "-since"}, "", "Show created since Id or Name, include non-running")
		before   = cmd.String([]string{"#beforeId", "#-before-id", "-before"}, "", "Show only container created before Id or Name")
		last     = cmd.Int([]string{"n"}, -1, "Show n last created containers, include non-running")
		flFilter = opts.NewListOpts(nil)
	)
	cmd.Require(flag.Exact, 0)

	cmd.Var(&flFilter, []string{"f", "-filter"}, "Filter output based on conditions provided")

	cmd.ParseFlags(args, true)
	if *last == -1 && *nLatest {
		*last = 1
	}

	if *all {
		v.Set("all", "1")
	}

	if *last != -1 {
		v.Set("limit", strconv.Itoa(*last))
	}

	if *since != "" {
		v.Set("since", *since)
	}

	if *before != "" {
		v.Set("before", *before)
	}

	if *size {
		v.Set("size", "1")
	}

	// Consolidate all filter flags, and sanity check them.
	// They'll get processed in the daemon/server.
	for _, f := range flFilter.GetAll() {
		if psFilterArgs, err = filters.ParseFlag(f, psFilterArgs); err != nil {
			return err
		}
	}

	if len(psFilterArgs) > 0 {
		filterJSON, err := filters.ToParam(psFilterArgs)
		if err != nil {
			return err
		}

		v.Set("filters", filterJSON)
	}

	serverResp, err := cli.call("GET", "/containers/json?"+v.Encode(), nil, nil)
	if err != nil {
		return err
	}

	defer serverResp.body.Close()

	containers := []types.Container{}
	if err := json.NewDecoder(serverResp.body).Decode(&containers); err != nil {
		return err
	}

	w := tabwriter.NewWriter(cli.out, 20, 1, 3, ' ', 0)
	if !*quiet {
		fmt.Fprint(w, "CONTAINER ID\tIMAGE\tCOMMAND\tCREATED\tSTATUS\tPORTS\tNAMES")

		if *size {
			fmt.Fprintln(w, "\tSIZE")
		} else {
			fmt.Fprint(w, "\n")
		}
	}

	stripNamePrefix := func(ss []string) []string {
		for i, s := range ss {
			ss[i] = s[1:]
		}

		return ss
	}

	for _, container := range containers {
		ID := container.ID

		if !*noTrunc {
			ID = stringid.TruncateID(ID)
		}

		if *quiet {
			fmt.Fprintln(w, ID)

			continue
		}

		var (
			names       = stripNamePrefix(container.Names)
			command     = strconv.Quote(container.Command)
			displayPort string
		)

		if !*noTrunc {
			command = stringutils.Truncate(command, 20)

			// only display the default name for the container with notrunc is passed
			for _, name := range names {
				if len(strings.Split(name, "/")) == 1 {
					names = []string{name}
					break
				}
			}
		}

		image := container.Image
		if image == "" {
			image = "<no image>"
		}

		if container.HostConfig.NetworkMode == "host" {
			displayPort = "*/tcp, */udp"
		} else {
			displayPort = api.DisplayablePorts(container.Ports)
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s ago\t%s\t%s\t%s\t", ID, image, command,
			units.HumanDuration(time.Now().UTC().Sub(time.Unix(int64(container.Created), 0))),
			container.Status, displayPort, strings.Join(names, ","))

		if *size {
			if container.SizeRootFs > 0 {
				fmt.Fprintf(w, "%s (virtual %s)\n", units.HumanSize(float64(container.SizeRw)), units.HumanSize(float64(container.SizeRootFs)))
			} else {
				fmt.Fprintf(w, "%s\n", units.HumanSize(float64(container.SizeRw)))
			}

			continue
		}

		fmt.Fprint(w, "\n")
	}

	if !*quiet {
		w.Flush()
	}

	return nil
}
