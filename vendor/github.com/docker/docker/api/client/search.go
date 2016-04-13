package client

import (
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"text/tabwriter"

	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/stringutils"
	"github.com/docker/docker/registry"
)

// ByStars sorts search results in ascending order by number of stars.
type ByStars []registry.SearchResult

func (r ByStars) Len() int           { return len(r) }
func (r ByStars) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r ByStars) Less(i, j int) bool { return r[i].StarCount < r[j].StarCount }

// CmdSearch searches the Docker Hub for images.
//
// Usage: docker search [OPTIONS] TERM
func (cli *DockerCli) CmdSearch(args ...string) error {
	cmd := cli.Subcmd("search", []string{"TERM"}, "Search the Docker Hub for images", true)
	noTrunc := cmd.Bool([]string{"#notrunc", "-no-trunc"}, false, "Don't truncate output")
	trusted := cmd.Bool([]string{"#t", "#trusted", "#-trusted"}, false, "Only show trusted builds")
	automated := cmd.Bool([]string{"-automated"}, false, "Only show automated builds")
	stars := cmd.Uint([]string{"s", "#stars", "-stars"}, 0, "Only displays with at least x stars")
	cmd.Require(flag.Exact, 1)

	cmd.ParseFlags(args, true)

	name := cmd.Arg(0)
	v := url.Values{}
	v.Set("term", name)

	// Resolve the Repository name from fqn to hostname + name
	taglessRemote, _ := parsers.ParseRepositoryTag(name)
	repoInfo, err := registry.ParseRepositoryInfo(taglessRemote)
	if err != nil {
		return err
	}

	rdr, _, err := cli.clientRequestAttemptLogin("GET", "/images/search?"+v.Encode(), nil, nil, repoInfo.Index, "search")
	if err != nil {
		return err
	}

	defer rdr.Close()

	results := ByStars{}
	if err := json.NewDecoder(rdr).Decode(&results); err != nil {
		return err
	}

	sort.Sort(sort.Reverse(results))

	w := tabwriter.NewWriter(cli.out, 10, 1, 3, ' ', 0)
	fmt.Fprintf(w, "NAME\tDESCRIPTION\tSTARS\tOFFICIAL\tAUTOMATED\n")
	for _, res := range results {
		if ((*automated || *trusted) && (!res.IsTrusted && !res.IsAutomated)) || (int(*stars) > res.StarCount) {
			continue
		}
		desc := strings.Replace(res.Description, "\n", " ", -1)
		desc = strings.Replace(desc, "\r", " ", -1)
		if !*noTrunc && len(desc) > 45 {
			desc = stringutils.Truncate(desc, 42) + "..."
		}
		fmt.Fprintf(w, "%s\t%s\t%d\t", res.Name, desc, res.StarCount)
		if res.IsOfficial {
			fmt.Fprint(w, "[OK]")

		}
		fmt.Fprint(w, "\t")
		if res.IsAutomated || res.IsTrusted {
			fmt.Fprint(w, "[OK]")
		}
		fmt.Fprint(w, "\n")
	}
	w.Flush()
	return nil
}
