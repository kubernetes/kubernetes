package main

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	bench "golang.org/x/tools/benchmark/parse"
)

var noPassthrough = flag.Bool("no-passthrough", false, "Don't print non-benchmark lines")

type BenchOutputGroup struct {
	Lines []*bench.Benchmark
	// Columns which are in use
	Measured int
}

type Table struct {
	MaxLengths []int
	Cells      [][]string
}

func (g *BenchOutputGroup) String() string {
	if len(g.Lines) == 0 {
		return ""
	}
	columnNames := []string{"benchmark", "iter", "time/iter"}
	if (g.Measured & bench.MBPerS) > 0 {
		columnNames = append(columnNames, "throughput")
	}
	if (g.Measured & bench.AllocedBytesPerOp) > 0 {
		columnNames = append(columnNames, "bytes alloc")
	}
	if (g.Measured & bench.AllocsPerOp) > 0 {
		columnNames = append(columnNames, "allocs")
	}
	table := &Table{Cells: [][]string{columnNames}}

	var underlines []string
	for _, name := range columnNames {
		underlines = append(underlines, strings.Repeat("-", len(name)))
	}
	table.Cells = append(table.Cells, underlines)
	timeFormatFunc := g.TimeFormatFunc()

	for _, line := range g.Lines {
		row := []string{line.Name, FormatIterations(line.N), timeFormatFunc(line.NsPerOp)}
		if (g.Measured & bench.MBPerS) > 0 {
			row = append(row, FormatMegaBytesPerSecond(line))
		}
		if (g.Measured & bench.AllocedBytesPerOp) > 0 {
			row = append(row, FormatBytesAllocPerOp(line))
		}
		if (g.Measured & bench.AllocsPerOp) > 0 {
			row = append(row, FormatAllocsPerOp(line))
		}
		table.Cells = append(table.Cells, row)
	}
	for i := range columnNames {
		maxLength := 0
		for _, row := range table.Cells {
			if len(row[i]) > maxLength {
				maxLength = len(row[i])
			}
		}
		table.MaxLengths = append(table.MaxLengths, maxLength)
	}
	var buf bytes.Buffer
	for _, row := range table.Cells {
		for i, cell := range row {
			var format string
			switch i {
			case 0:
				format = "%%-%ds   "
			case len(row) - 1:
				format = "%%%ds"
			default:
				format = "%%%ds   "
			}
			fmt.Fprintf(&buf, fmt.Sprintf(format, table.MaxLengths[i]), cell)
		}
		fmt.Fprint(&buf, "\n")
	}
	return buf.String()
}

func FormatIterations(iter int) string {
	return strconv.FormatInt(int64(iter), 10)
}

func (g *BenchOutputGroup) TimeFormatFunc() func(float64) string {
	// Find the smallest time
	smallest := g.Lines[0].NsPerOp
	for _, line := range g.Lines[1:] {
		if line.NsPerOp < smallest {
			smallest = line.NsPerOp
		}
	}
	switch {
	case smallest < float64(10000*time.Nanosecond):
		return func(ns float64) string {
			return fmt.Sprintf("%.2f ns/op", ns)
		}
	case smallest < float64(time.Millisecond):
		return func(ns float64) string {
			return fmt.Sprintf("%.2f μs/op", ns/1000)
		}
	case smallest < float64(10*time.Second):
		return func(ns float64) string {
			return fmt.Sprintf("%.2f ms/op", (ns / 1e6))
		}
	default:
		return func(ns float64) string {
			return fmt.Sprintf("%.2f s/op", ns/1e9)
		}
	}
}

func FormatMegaBytesPerSecond(l *bench.Benchmark) string {
	if (l.Measured & bench.MBPerS) == 0 {
		return ""
	}
	return fmt.Sprintf("%.2f MB/s", l.MBPerS)
}

func FormatBytesAllocPerOp(l *bench.Benchmark) string {
	if (l.Measured & bench.AllocedBytesPerOp) == 0 {
		return ""
	}
	return fmt.Sprintf("%d B/op", l.AllocedBytesPerOp)
}

func FormatAllocsPerOp(l *bench.Benchmark) string {
	if (l.Measured & bench.AllocsPerOp) == 0 {
		return ""
	}
	return fmt.Sprintf("%d allocs/op", l.AllocsPerOp)
}

func (g *BenchOutputGroup) AddLine(line *bench.Benchmark) {
	g.Lines = append(g.Lines, line)
	g.Measured |= line.Measured
}

var (
	benchLineMatcher = regexp.MustCompile(`^Benchmark.*\t.*\d+`)
	okLineMatcher    = regexp.MustCompile(`^ok\s`)
	notBenchLineErr  = errors.New("Not a bench line")
)

func ParseLine(line string) (*bench.Benchmark, error) {
	if !benchLineMatcher.MatchString(line) {
		return nil, notBenchLineErr
	}
	fields := strings.Split(line, "\t")
	if len(fields) < 3 {
		return nil, notBenchLineErr
	}

	return bench.ParseLine(line)
}

func main() {
	flag.Parse()
	currentBenchmark := &BenchOutputGroup{}
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		text := scanner.Text()
		line, err := ParseLine(text)
		switch err {
		case notBenchLineErr:
			if okLineMatcher.MatchString(text) {
				fmt.Print(currentBenchmark)
				currentBenchmark = &BenchOutputGroup{}
			}
			if !*noPassthrough {
				fmt.Println(text)
			}
		case nil:
			currentBenchmark.AddLine(line)
		default:
			fmt.Fprintln(os.Stderr, "prettybench unrecognized line:")
			fmt.Println(text)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
