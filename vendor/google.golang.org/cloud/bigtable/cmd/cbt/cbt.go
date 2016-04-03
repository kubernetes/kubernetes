/*
Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

// Command docs are in cbtdoc.go.

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"text/template"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/cloud/bigtable"
	"google.golang.org/cloud/bigtable/internal/cbtrc"
)

var (
	oFlag = flag.String("o", "", "if set, redirect stdout to this file")

	config             *cbtrc.Config
	client             *bigtable.Client
	adminClient        *bigtable.AdminClient
	clusterAdminClient *bigtable.ClusterAdminClient
)

func getClient() *bigtable.Client {
	if client == nil {
		var err error
		client, err = bigtable.NewClient(context.Background(), config.Project, config.Zone, config.Cluster)
		if err != nil {
			log.Fatalf("Making bigtable.Client: %v", err)
		}
	}
	return client
}

func getAdminClient() *bigtable.AdminClient {
	if adminClient == nil {
		var err error
		adminClient, err = bigtable.NewAdminClient(context.Background(), config.Project, config.Zone, config.Cluster)
		if err != nil {
			log.Fatalf("Making bigtable.AdminClient: %v", err)
		}
	}
	return adminClient
}

func getClusterAdminClient() *bigtable.ClusterAdminClient {
	if clusterAdminClient == nil {
		var err error
		clusterAdminClient, err = bigtable.NewClusterAdminClient(context.Background(), config.Project)
		if err != nil {
			log.Fatalf("Making bigtable.ClusterAdminClient: %v", err)
		}
	}
	return clusterAdminClient
}

func main() {
	var err error
	config, err = cbtrc.Load()
	if err != nil {
		log.Fatal(err)
	}
	config.RegisterFlags()

	flag.Usage = usage
	flag.Parse()
	if err := config.CheckFlags(); err != nil {
		log.Fatal(err)
	}
	if config.Creds != "" {
		os.Setenv("GOOGLE_APPLICATION_CREDENTIALS", config.Creds)
	}
	if flag.NArg() == 0 {
		usage()
		os.Exit(1)
	}

	if *oFlag != "" {
		f, err := os.Create(*oFlag)
		if err != nil {
			log.Fatal(err)
		}
		defer func() {
			if err := f.Close(); err != nil {
				log.Fatal(err)
			}
		}()
		os.Stdout = f
	}

	ctx := context.Background()
	for _, cmd := range commands {
		if cmd.Name == flag.Arg(0) {
			cmd.do(ctx, flag.Args()[1:]...)
			return
		}
	}
	log.Fatalf("Unknown command %q", flag.Arg(0))
}

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: %s [flags] <command> ...\n", os.Args[0])
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, "\n%s", cmdSummary)
}

var cmdSummary string // generated in init, below

func init() {
	var buf bytes.Buffer
	tw := tabwriter.NewWriter(&buf, 10, 8, 4, '\t', 0)
	for _, cmd := range commands {
		fmt.Fprintf(tw, "cbt %s\t%s\n", cmd.Name, cmd.Desc)
	}
	tw.Flush()
	buf.WriteString(configHelp)
	cmdSummary = buf.String()
}

var configHelp = `
For convenience, values of the -project, -zone, -cluster and -creds flags
may be specified in ` + cbtrc.Filename() + ` in this format:
	project = my-project-123
	zone = us-central1-b
	cluster = my-cluster
	creds = path-to-account-key.json
All values are optional, and all will be overridden by flags.
`

var commands = []struct {
	Name, Desc string
	do         func(context.Context, ...string)
	Usage      string
}{
	{
		Name:  "count",
		Desc:  "Count rows in a table",
		do:    doCount,
		Usage: "cbt count <table>",
	},
	{
		Name:  "createfamily",
		Desc:  "Create a column family",
		do:    doCreateFamily,
		Usage: "cbt createfamily <table> <family>",
	},
	{
		Name:  "createtable",
		Desc:  "Create a table",
		do:    doCreateTable,
		Usage: "cbt createtable <table>",
	},
	{
		Name:  "deletefamily",
		Desc:  "Delete a column family",
		do:    doDeleteFamily,
		Usage: "cbt deletefamily <table> <family>",
	},
	{
		Name:  "deleterow",
		Desc:  "Delete a row",
		do:    doDeleteRow,
		Usage: "cbt deleterow <table> <row>",
	},
	{
		Name:  "deletetable",
		Desc:  "Delete a table",
		do:    doDeleteTable,
		Usage: "cbt deletetable <table>",
	},
	{
		Name:  "doc",
		Desc:  "Print documentation for cbt",
		do:    doDoc,
		Usage: "cbt doc",
	},
	{
		Name:  "help",
		Desc:  "Print help text",
		do:    doHelp,
		Usage: "cbt help [command]",
	},
	{
		Name:  "listclusters",
		Desc:  "List clusters in a project",
		do:    doListClusters,
		Usage: "cbt listclusters",
	},
	{
		Name:  "lookup",
		Desc:  "Read from a single row",
		do:    doLookup,
		Usage: "cbt lookup <table> <row>",
	},
	{
		Name: "ls",
		Desc: "List tables and column families",
		do:   doLS,
		Usage: "cbt ls			List tables\n" +
			"cbt ls <table>		List column families in <table>",
	},
	{
		Name: "read",
		Desc: "Read rows",
		do:   doRead,
		Usage: "cbt read <table> [start=<row>] [end=<row>] [prefix=<prefix>] [count=<n>]\n" +
			"  start=<row>		Start reading at this row\n" +
			"  end=<row>		Stop reading before this row\n" +
			"  prefix=<prefix>	Read rows with this prefix\n" +
			"  count=<n>		Read only this many rows\n",
	},
	{
		Name: "set",
		Desc: "Set value of a cell",
		do:   doSet,
		Usage: "cbt set <table> <row> family:column=val[@ts] ...\n" +
			"  family:column=val[@ts] may be repeated to set multiple cells.\n" +
			"\n" +
			"  ts is an optional integer timestamp.\n" +
			"  If it cannot be parsed, the `@ts` part will be\n" +
			"  interpreted as part of the value.",
	},
	/* TODO(dsymonds): Re-enable when there's a ClusterAdmin API.
	{
		Name:  "setclustersize",
		Desc:  "Set size of a cluster",
		do:    doSetClusterSize,
		Usage: "cbt setclustersize <num_nodes>",
	},
	*/
	{
		Name: "setgcpolicy",
		Desc: "Set the GC policy for a column family",
		do:   doSetGCPolicy,
		Usage: "cbt setgcpolicy <table> <family> ( maxage=<d> | maxversions=<n> )\n" +
			"\n" +
			`  maxage=<d>		Maximum timestamp age to preserve (e.g. "1h", "4d")` + "\n" +
			"  maxversions=<n>	Maximum number of versions to preserve",
	},
}

func doCount(ctx context.Context, args ...string) {
	if len(args) != 1 {
		log.Fatal("usage: cbt count <table>")
	}
	tbl := getClient().Open(args[0])

	n := 0
	err := tbl.ReadRows(ctx, bigtable.InfiniteRange(""), func(_ bigtable.Row) bool {
		n++
		return true
	}, bigtable.RowFilter(bigtable.StripValueFilter()))
	if err != nil {
		log.Fatalf("Reading rows: %v", err)
	}
	fmt.Println(n)
}

func doCreateFamily(ctx context.Context, args ...string) {
	if len(args) != 2 {
		log.Fatal("usage: cbt createfamily <table> <family>")
	}
	err := getAdminClient().CreateColumnFamily(ctx, args[0], args[1])
	if err != nil {
		log.Fatalf("Creating column family: %v", err)
	}
}

func doCreateTable(ctx context.Context, args ...string) {
	if len(args) != 1 {
		log.Fatal("usage: cbt createtable <table>")
	}
	err := getAdminClient().CreateTable(ctx, args[0])
	if err != nil {
		log.Fatalf("Creating table: %v", err)
	}
}

func doDeleteFamily(ctx context.Context, args ...string) {
	if len(args) != 2 {
		log.Fatal("usage: cbt deletefamily <table> <family>")
	}
	err := getAdminClient().DeleteColumnFamily(ctx, args[0], args[1])
	if err != nil {
		log.Fatalf("Deleting column family: %v", err)
	}
}

func doDeleteRow(ctx context.Context, args ...string) {
	if len(args) != 2 {
		log.Fatal("usage: cbt deleterow <table> <row>")
	}
	tbl := getClient().Open(args[0])
	mut := bigtable.NewMutation()
	mut.DeleteRow()
	if err := tbl.Apply(ctx, args[1], mut); err != nil {
		log.Fatalf("Deleting row: %v", err)
	}
}

func doDeleteTable(ctx context.Context, args ...string) {
	if len(args) != 1 {
		log.Fatalf("Can't do `cbt deletetable %s`", args)
	}
	err := getAdminClient().DeleteTable(ctx, args[0])
	if err != nil {
		log.Fatalf("Deleting table: %v", err)
	}
}

// to break circular dependencies
var (
	doDocFn  func(ctx context.Context, args ...string)
	doHelpFn func(ctx context.Context, args ...string)
)

func init() {
	doDocFn = doDocReal
	doHelpFn = doHelpReal
}

func doDoc(ctx context.Context, args ...string)  { doDocFn(ctx, args...) }
func doHelp(ctx context.Context, args ...string) { doHelpFn(ctx, args...) }

func doDocReal(ctx context.Context, args ...string) {
	data := map[string]interface{}{
		"Commands": commands,
	}
	var buf bytes.Buffer
	if err := docTemplate.Execute(&buf, data); err != nil {
		log.Fatalf("Bad doc template: %v", err)
	}
	out, err := format.Source(buf.Bytes())
	if err != nil {
		log.Fatalf("Bad doc output: %v", err)
	}
	os.Stdout.Write(out)
}

var docTemplate = template.Must(template.New("doc").Funcs(template.FuncMap{
	"indent": func(s, ind string) string {
		ss := strings.Split(s, "\n")
		for i, p := range ss {
			ss[i] = ind + p
		}
		return strings.Join(ss, "\n")
	},
}).
	Parse(`
// DO NOT EDIT. THIS IS AUTOMATICALLY GENERATED.
// Run "go generate" to regenerate.
//go:generate go run cbt.go -o cbtdoc.go doc

/*
Cbt is a tool for doing basic interactions with Cloud Bigtable.

Usage:

	cbt [options] command [arguments]

The commands are:
{{range .Commands}}
	{{printf "%-25s %s" .Name .Desc}}{{end}}

Use "cbt help <command>" for more information about a command.

{{range .Commands}}
{{.Desc}}

Usage:
{{indent .Usage "\t"}}



{{end}}
*/
package main
`))

func doHelpReal(ctx context.Context, args ...string) {
	if len(args) == 0 {
		fmt.Print(cmdSummary)
		return
	}
	for _, cmd := range commands {
		if cmd.Name == args[0] {
			fmt.Println(cmd.Usage)
			return
		}
	}
	log.Fatalf("Don't know command %q", args[0])
}

func doListClusters(ctx context.Context, args ...string) {
	if len(args) != 0 {
		log.Fatalf("usage: cbt listclusters")
	}
	cis, err := getClusterAdminClient().Clusters(ctx)
	if err != nil {
		log.Fatalf("Getting list of clusters: %v", err)
	}
	tw := tabwriter.NewWriter(os.Stdout, 10, 8, 4, '\t', 0)
	fmt.Fprintf(tw, "Cluster Name\tZone\tInfo\n")
	fmt.Fprintf(tw, "------------\t----\t----\n")
	for _, ci := range cis {
		fmt.Fprintf(tw, "%s\t%s\t%s (%d serve nodes)\n", ci.Name, ci.Zone, ci.DisplayName, ci.ServeNodes)
	}
	tw.Flush()
}

func doLookup(ctx context.Context, args ...string) {
	if len(args) != 2 {
		log.Fatalf("usage: cbt lookup <table> <row>")
	}
	table, row := args[0], args[1]
	tbl := getClient().Open(table)
	r, err := tbl.ReadRow(ctx, row)
	if err != nil {
		log.Fatalf("Reading row: %v", err)
	}
	printRow(r)
}

func printRow(r bigtable.Row) {
	fmt.Println(strings.Repeat("-", 40))
	fmt.Println(r.Key())

	var fams []string
	for fam := range r {
		fams = append(fams, fam)
	}
	sort.Strings(fams)
	for _, fam := range fams {
		ris := r[fam]
		sort.Sort(byColumn(ris))
		for _, ri := range ris {
			ts := time.Unix(0, int64(ri.Timestamp)*1e3)
			fmt.Printf("  %-40s @ %s\n", ri.Column, ts.Format("2006/01/02-15:04:05.000000"))
			fmt.Printf("    %q\n", ri.Value)
		}
	}
}

type byColumn []bigtable.ReadItem

func (b byColumn) Len() int           { return len(b) }
func (b byColumn) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byColumn) Less(i, j int) bool { return b[i].Column < b[j].Column }

func doLS(ctx context.Context, args ...string) {
	switch len(args) {
	default:
		log.Fatalf("Can't do `cbt ls %s`", args)
	case 0:
		tables, err := getAdminClient().Tables(ctx)
		if err != nil {
			log.Fatalf("Getting list of tables: %v", err)
		}
		sort.Strings(tables)
		for _, table := range tables {
			fmt.Println(table)
		}
	case 1:
		table := args[0]
		ti, err := getAdminClient().TableInfo(ctx, table)
		if err != nil {
			log.Fatalf("Getting table info: %v", err)
		}
		sort.Strings(ti.Families)
		for _, fam := range ti.Families {
			fmt.Println(fam)
		}
	}
}

func doRead(ctx context.Context, args ...string) {
	if len(args) < 1 {
		log.Fatalf("usage: cbt read <table> [args ...]")
	}
	tbl := getClient().Open(args[0])

	parsed := make(map[string]string)
	for _, arg := range args[1:] {
		i := strings.Index(arg, "=")
		if i < 0 {
			log.Fatalf("Bad arg %q", arg)
		}
		key, val := arg[:i], arg[i+1:]
		switch key {
		default:
			log.Fatalf("Unknown arg key %q", key)
		case "limit":
			// Be nicer; we used to support this, but renamed it to "end".
			log.Fatalf("Unknown arg key %q; did you mean %q?", key, "end")
		case "start", "end", "prefix", "count":
			parsed[key] = val
		}
	}
	if (parsed["start"] != "" || parsed["end"] != "") && parsed["prefix"] != "" {
		log.Fatal(`"start"/"end" may not be mixed with "prefix"`)
	}

	var rr bigtable.RowRange
	if start, end := parsed["start"], parsed["end"]; end != "" {
		rr = bigtable.NewRange(start, end)
	} else if start != "" {
		rr = bigtable.InfiniteRange(start)
	}
	if prefix := parsed["prefix"]; prefix != "" {
		rr = bigtable.PrefixRange(prefix)
	}

	var opts []bigtable.ReadOption
	if count := parsed["count"]; count != "" {
		n, err := strconv.ParseInt(count, 0, 64)
		if err != nil {
			log.Fatalf("Bad count %q: %v", count, err)
		}
		opts = append(opts, bigtable.LimitRows(n))
	}

	// TODO(dsymonds): Support filters.
	err := tbl.ReadRows(ctx, rr, func(r bigtable.Row) bool {
		printRow(r)
		return true
	}, opts...)
	if err != nil {
		log.Fatalf("Reading rows: %v", err)
	}
}

var setArg = regexp.MustCompile(`([^:]+):([^=]*)=(.*)`)

func doSet(ctx context.Context, args ...string) {
	if len(args) < 3 {
		log.Fatalf("usage: cbt set <table> <row> family:[column]=val[@ts] ...")
	}
	tbl := getClient().Open(args[0])
	row := args[1]
	mut := bigtable.NewMutation()
	for _, arg := range args[2:] {
		m := setArg.FindStringSubmatch(arg)
		if m == nil {
			log.Fatalf("Bad set arg %q", arg)
		}
		val := m[3]
		ts := bigtable.Now()
		if i := strings.LastIndex(val, "@"); i >= 0 {
			// Try parsing a timestamp.
			n, err := strconv.ParseInt(val[i+1:], 0, 64)
			if err == nil {
				val = val[:i]
				ts = bigtable.Timestamp(n)
			}
		}
		mut.Set(m[1], m[2], ts, []byte(val))
	}
	if err := tbl.Apply(ctx, row, mut); err != nil {
		log.Fatalf("Applying mutation: %v", err)
	}
}

/* TODO(dsymonds): Re-enable when there's a ClusterAdmin API.
func doSetClusterSize(ctx context.Context, args ...string) {
	if len(args) != 1 {
		log.Fatalf("usage: cbt setclustersize <num_nodes>")
	}
	n, err := strconv.ParseInt(args[0], 0, 32)
	if err != nil {
		log.Fatalf("Bad num_nodes value %q: %v", args[0], err)
	}
	if err := getAdminClient().SetClusterSize(ctx, int(n)); err != nil {
		log.Fatalf("Setting cluster size: %v", err)
	}
}
*/

func doSetGCPolicy(ctx context.Context, args ...string) {
	if len(args) < 3 {
		log.Fatalf("usage: cbt setgcpolicy <table> <family> ( maxage=<d> | maxversions=<n> )")
	}
	table := args[0]
	fam := args[1]

	var pol bigtable.GCPolicy
	switch p := args[2]; {
	case strings.HasPrefix(p, "maxage="):
		d, err := parseDuration(p[7:])
		if err != nil {
			log.Fatal(err)
		}
		pol = bigtable.MaxAgePolicy(d)
	case strings.HasPrefix(p, "maxversions="):
		n, err := strconv.ParseUint(p[12:], 10, 16)
		if err != nil {
			log.Fatal(err)
		}
		pol = bigtable.MaxVersionsPolicy(int(n))
	default:
		log.Fatalf("Bad GC policy %q", p)
	}
	if err := getAdminClient().SetGCPolicy(ctx, table, fam, pol); err != nil {
		log.Fatalf("Setting GC policy: %v", err)
	}
}

// parseDuration parses a duration string.
// It is similar to Go's time.ParseDuration, except with a different set of supported units,
// and only simple formats supported.
func parseDuration(s string) (time.Duration, error) {
	// [0-9]+[a-z]+

	// Split [0-9]+ from [a-z]+.
	i := 0
	for ; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			break
		}
	}
	ds, u := s[:i], s[i:]
	if ds == "" || u == "" {
		return 0, fmt.Errorf("invalid duration %q", s)
	}
	// Parse them.
	d, err := strconv.ParseUint(ds, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid duration %q: %v", s, err)
	}
	unit, ok := unitMap[u]
	if !ok {
		return 0, fmt.Errorf("unknown unit %q in duration %q", u, s)
	}
	if d > uint64((1<<63-1)/unit) {
		// overflow
		return 0, fmt.Errorf("invalid duration %q overflows", s)
	}
	return time.Duration(d) * unit, nil
}

var unitMap = map[string]time.Duration{
	"ms": time.Millisecond,
	"s":  time.Second,
	"m":  time.Minute,
	"h":  time.Hour,
	"d":  24 * time.Hour,
}
