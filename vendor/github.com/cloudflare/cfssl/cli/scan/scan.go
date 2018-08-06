package scan

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/scan"
)

var scanUsageText = `cfssl scan -- scan a host for issues
Usage of scan:
        cfssl scan [-family regexp] [-scanner regexp] [-timeout duration] [-ip IPAddr] [-num-workers num] [-max-hosts num] [-csv hosts.csv] HOST+
        cfssl scan -list

Arguments:
        HOST:    Host(s) to scan (including port)
Flags:
`
var scanFlags = []string{"list", "family", "scanner", "timeout", "ip", "ca-bundle", "num-workers", "csv", "max-hosts"}

func printJSON(v interface{}) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("%s\n\n", b)
}

type context struct {
	sync.WaitGroup
	c     cli.Config
	hosts chan string
}

func newContext(c cli.Config, numWorkers int) *context {
	ctx := &context{
		c:     c,
		hosts: make(chan string, numWorkers),
	}
	ctx.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go ctx.runWorker()
	}
	return ctx
}

func (ctx *context) runWorker() {
	for host := range ctx.hosts {
		fmt.Printf("Scanning %s...\n", host)
		results, err := scan.Default.RunScans(host, ctx.c.IP, ctx.c.Family, ctx.c.Scanner, ctx.c.Timeout)
		fmt.Printf("=== %s ===\n", host)
		if err != nil {
			log.Error(err)
		} else {
			printJSON(results)
		}
	}
	ctx.Done()
}

func parseCSV(hosts []string, csvFile string, maxHosts int) ([]string, error) {
	f, err := os.Open(csvFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	for err == nil && len(hosts) < maxHosts {
		var record []string
		record, err = r.Read()
		hosts = append(hosts, record[len(record)-1])
	}
	if err == io.EOF {
		err = nil
	}

	return hosts, err
}

func scanMain(args []string, c cli.Config) (err error) {
	if c.List {
		printJSON(scan.Default)
	} else {
		if err = scan.LoadRootCAs(c.CABundleFile); err != nil {
			return
		}

		if len(args) >= c.MaxHosts {
			log.Warningf("Only scanning max-hosts=%d out of %d args given", c.MaxHosts, len(args))
			args = args[:c.MaxHosts]
		} else if c.CSVFile != "" {
			args, err = parseCSV(args, c.CSVFile, c.MaxHosts)
			if err != nil {
				return
			}
		}

		ctx := newContext(c, c.NumWorkers)
		// Execute for each HOST argument given
		for len(args) > 0 {
			var host string
			host, args, err = cli.PopFirstArgument(args)
			if err != nil {
				return
			}

			ctx.hosts <- host
		}
		close(ctx.hosts)
		ctx.Wait()
	}
	return
}

// Command assembles the definition of Command 'scan'
var Command = &cli.Command{UsageText: scanUsageText, Flags: scanFlags, Main: scanMain}
