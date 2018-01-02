package report

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/tsdb/engine/tsm1"
	"github.com/retailnext/hllpp"
)

// Command represents the program execution for "influxd report".
type Command struct {
	Stderr io.Writer
	Stdout io.Writer

	dir      string
	pattern  string
	detailed bool
}

// NewCommand returns a new instance of Command.
func NewCommand() *Command {
	return &Command{
		Stderr: os.Stderr,
		Stdout: os.Stdout,
	}
}

// Run executes the command.
func (cmd *Command) Run(args ...string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	fs.StringVar(&cmd.pattern, "pattern", "", "Include only files matching a pattern")
	fs.BoolVar(&cmd.detailed, "detailed", false, "Report detailed cardinality estimates")

	fs.SetOutput(cmd.Stdout)
	fs.Usage = cmd.printUsage

	if err := fs.Parse(args); err != nil {
		return err
	}
	cmd.dir = fs.Arg(0)

	start := time.Now()

	files, err := filepath.Glob(filepath.Join(cmd.dir, fmt.Sprintf("*.%s", tsm1.TSMFileExtension)))
	if err != nil {
		return err
	}

	var filtered []string
	if cmd.pattern != "" {
		for _, f := range files {
			if strings.Contains(f, cmd.pattern) {
				filtered = append(filtered, f)
			}
		}
		files = filtered
	}

	if len(files) == 0 {
		return fmt.Errorf("no tsm files at %v\n", cmd.dir)
	}

	tw := tabwriter.NewWriter(cmd.Stdout, 8, 8, 1, '\t', 0)
	fmt.Fprintln(tw, strings.Join([]string{"File", "Series", "Load Time"}, "\t"))

	totalSeries := hllpp.New()
	tagCardialities := map[string]*hllpp.HLLPP{}
	measCardinalities := map[string]*hllpp.HLLPP{}
	fieldCardinalities := map[string]*hllpp.HLLPP{}

	ordering := make([]chan struct{}, 0, len(files))
	for range files {
		ordering = append(ordering, make(chan struct{}))
	}

	for _, f := range files {
		file, err := os.OpenFile(f, os.O_RDONLY, 0600)
		if err != nil {
			fmt.Fprintf(cmd.Stderr, "error: %s: %v. Skipping.\n", f, err)
			continue
		}

		loadStart := time.Now()
		reader, err := tsm1.NewTSMReader(file)
		if err != nil {
			fmt.Fprintf(cmd.Stderr, "error: %s: %v. Skipping.\n", file.Name(), err)
			continue
		}
		loadTime := time.Since(loadStart)

		seriesCount := reader.KeyCount()
		for i := 0; i < seriesCount; i++ {
			key, _ := reader.KeyAt(i)
			totalSeries.Add([]byte(key))

			if cmd.detailed {
				sep := strings.Index(string(key), "#!~#")
				seriesKey, field := key[:sep], key[sep+4:]
				measurement, tags, _ := models.ParseKey(seriesKey)

				measCount, ok := measCardinalities[measurement]
				if !ok {
					measCount = hllpp.New()
					measCardinalities[measurement] = measCount
				}
				measCount.Add([]byte(key))

				fieldCount, ok := fieldCardinalities[measurement]
				if !ok {
					fieldCount = hllpp.New()
					fieldCardinalities[measurement] = fieldCount
				}
				fieldCount.Add([]byte(field))

				for _, t := range tags {
					tagCount, ok := tagCardialities[string(t.Key)]
					if !ok {
						tagCount = hllpp.New()
						tagCardialities[string(t.Key)] = tagCount
					}
					tagCount.Add(t.Value)
				}
			}
		}
		reader.Close()

		fmt.Fprintln(tw, strings.Join([]string{
			filepath.Base(file.Name()),
			strconv.FormatInt(int64(seriesCount), 10),
			loadTime.String(),
		}, "\t"))
		tw.Flush()
	}

	tw.Flush()
	println()
	fmt.Printf("Statistics\n")
	fmt.Printf("  Series:\n")
	fmt.Printf("    Total (est): %d\n", totalSeries.Count())
	if cmd.detailed {
		fmt.Printf("  Measurements (est):\n")
		for t, card := range measCardinalities {
			fmt.Printf("    %v: %d (%d%%)\n", t, card.Count(), int((float64(card.Count())/float64(totalSeries.Count()))*100))
		}

		fmt.Printf("  Fields (est):\n")
		for t, card := range fieldCardinalities {
			fmt.Printf("    %v: %d\n", t, card.Count())
		}

		fmt.Printf("  Tags (est):\n")
		for t, card := range tagCardialities {
			fmt.Printf("    %v: %d\n", t, card.Count())
		}
	}

	fmt.Printf("Completed in %s\n", time.Since(start))
	return nil
}

// printUsage prints the usage message to STDERR.
func (cmd *Command) printUsage() {
	usage := `Displays shard level report.

Usage: influx_inspect report [flags]

    -pattern <pattern>
            Include only files matching a pattern.
    -detailed
            Report detailed cardinality estimates.
            Defaults to "false".
`

	fmt.Fprintf(cmd.Stdout, usage)
}
