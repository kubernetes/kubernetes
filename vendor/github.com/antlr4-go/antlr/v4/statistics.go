//go:build antlr.stats

package antlr

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

// This file allows the user to collect statistics about the runtime of the ANTLR runtime. It is not enabled by default
// and so incurs no time penalty. To enable it, you must build the runtime with the antlr.stats build tag.
//

// Tells various components to collect statistics - because it is only true when this file is included, it will
// allow the compiler to completely eliminate all the code that is only used when collecting statistics.
const collectStats = true

// goRunStats is a collection of all the various data the ANTLR runtime has collected about a particular run.
// It is exported so that it can be used by others to look for things that are not already looked for in the
// runtime statistics.
type goRunStats struct {

	// jStats is a slice of all the [JStatRec] records that have been created, which is one for EVERY collection created
	// during a run. It is exported so that it can be used by others to look for things that are not already looked for
	// within this package.
	//
	jStats            []*JStatRec
	jStatsLock        RWMutex
	topN              int
	topNByMax         []*JStatRec
	topNByUsed        []*JStatRec
	unusedCollections map[CollectionSource]int
	counts            map[CollectionSource]int
}

const (
	collectionsFile = "collections"
)

var (
	Statistics = &goRunStats{
		topN: 10,
	}
)

type statsOption func(*goRunStats) error

// Configure allows the statistics system to be configured as the user wants and override the defaults
func (s *goRunStats) Configure(options ...statsOption) error {
	for _, option := range options {
		err := option(s)
		if err != nil {
			return err
		}
	}
	return nil
}

// WithTopN sets the number of things to list in the report when we are concerned with the top N things.
//
// For example, if you want to see the top 20 collections by size, you can do:
//
//	antlr.Statistics.Configure(antlr.WithTopN(20))
func WithTopN(topN int) statsOption {
	return func(s *goRunStats) error {
		s.topN = topN
		return nil
	}
}

// Analyze looks through all the statistical records and computes all the outputs that might be useful to the user.
//
// The function gathers and analyzes a number of statistics about any particular run of
// an ANTLR generated recognizer. In the vast majority of cases, the statistics are only
// useful to maintainers of ANTLR itself, but they can be useful to users as well. They may be
// especially useful in tracking down bugs or performance problems when an ANTLR user could
// supply the output from this package, but cannot supply the grammar file(s) they are using, even
// privately to the maintainers.
//
// The statistics are gathered by the runtime itself, and are not gathered by the parser or lexer, but the user
// must call this function their selves to analyze the statistics. This is because none of the infrastructure is
// extant unless the calling program is built with the antlr.stats tag like so:
//
// go build -tags antlr.stats .
//
// When a program is built with the antlr.stats tag, the Statistics object is created and available outside
// the package. The user can then call the [Statistics.Analyze] function to analyze the statistics and then call the
// [Statistics.Report] function to report the statistics.
//
// Please forward any questions about this package to the ANTLR discussion groups on GitHub or send to them to
// me [Jim Idle] directly at jimi@idle.ws
//
// [Jim Idle]: https:://github.com/jim-idle
func (s *goRunStats) Analyze() {

	// Look for anything that looks strange and record it in our local maps etc for the report to present it
	//
	s.CollectionAnomalies()
	s.TopNCollections()
}

// TopNCollections looks through all the statistical records and gathers the top ten collections by size.
func (s *goRunStats) TopNCollections() {

	// Let's sort the stat records by MaxSize
	//
	sort.Slice(s.jStats, func(i, j int) bool {
		return s.jStats[i].MaxSize > s.jStats[j].MaxSize
	})

	for i := 0; i < len(s.jStats) && i < s.topN; i++ {
		s.topNByMax = append(s.topNByMax, s.jStats[i])
	}

	// Sort by the number of times used
	//
	sort.Slice(s.jStats, func(i, j int) bool {
		return s.jStats[i].Gets+s.jStats[i].Puts > s.jStats[j].Gets+s.jStats[j].Puts
	})
	for i := 0; i < len(s.jStats) && i < s.topN; i++ {
		s.topNByUsed = append(s.topNByUsed, s.jStats[i])
	}
}

// Report dumps a markdown formatted report of all the statistics collected during a run to the given dir output
// path, which should represent a directory. Generated files will be prefixed with the given prefix and will be
// given a type name such as `anomalies` and a time stamp such as `2021-09-01T12:34:56` and a .md suffix.
func (s *goRunStats) Report(dir string, prefix string) error {

	isDir, err := isDirectory(dir)
	switch {
	case err != nil:
		return err
	case !isDir:
		return fmt.Errorf("output directory `%s` is not a directory", dir)
	}
	s.reportCollections(dir, prefix)

	// Clean out any old data in case the user forgets
	//
	s.Reset()
	return nil
}

func (s *goRunStats) Reset() {
	s.jStats = nil
	s.topNByUsed = nil
	s.topNByMax = nil
}

func (s *goRunStats) reportCollections(dir, prefix string) {
	cname := filepath.Join(dir, ".asciidoctor")
	// If the file doesn't exist, create it, or append to the file
	f, err := os.OpenFile(cname, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	_, _ = f.WriteString(`// .asciidoctorconfig
++++
<style>
body {
font-family: "Quicksand", "Montserrat", "Helvetica";
background-color: black;
}
</style>
++++`)
	_ = f.Close()

	fname := filepath.Join(dir, prefix+"_"+"_"+collectionsFile+"_"+".adoc")
	// If the file doesn't exist, create it, or append to the file
	f, err = os.OpenFile(fname, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer func(f *os.File) {
		err := f.Close()
		if err != nil {
			log.Fatal(err)
		}
	}(f)
	_, _ = f.WriteString("= Collections for " + prefix + "\n\n")

	_, _ = f.WriteString("== Summary\n")

	if s.unusedCollections != nil {
		_, _ = f.WriteString("=== Unused Collections\n")
		_, _ = f.WriteString("Unused collections incur a penalty for allocation that makes them a candidate for either\n")
		_, _ = f.WriteString(" removal or optimization. If you are using a collection that is not used, you should\n")
		_, _ = f.WriteString(" consider removing it. If you are using a collection that is used, but not very often,\n")
		_, _ = f.WriteString(" you should consider using lazy initialization to defer the allocation until it is\n")
		_, _ = f.WriteString(" actually needed.\n\n")

		_, _ = f.WriteString("\n.Unused collections\n")
		_, _ = f.WriteString(`[cols="<3,>1"]` + "\n\n")
		_, _ = f.WriteString("|===\n")
		_, _ = f.WriteString("| Type | Count\n")

		for k, v := range s.unusedCollections {
			_, _ = f.WriteString("| " + CollectionDescriptors[k].SybolicName + " | " + strconv.Itoa(v) + "\n")
		}
		f.WriteString("|===\n\n")
	}

	_, _ = f.WriteString("\n.Summary of Collections\n")
	_, _ = f.WriteString(`[cols="<3,>1"]` + "\n\n")
	_, _ = f.WriteString("|===\n")
	_, _ = f.WriteString("| Type | Count\n")
	for k, v := range s.counts {
		_, _ = f.WriteString("| " + CollectionDescriptors[k].SybolicName + " | " + strconv.Itoa(v) + "\n")
	}
	_, _ = f.WriteString("| Total | " + strconv.Itoa(len(s.jStats)) + "\n")
	_, _ = f.WriteString("|===\n\n")

	_, _ = f.WriteString("\n.Summary of Top " + strconv.Itoa(s.topN) + " Collections by MaxSize\n")
	_, _ = f.WriteString(`[cols="<1,<3,>1,>1,>1,>1"]` + "\n\n")
	_, _ = f.WriteString("|===\n")
	_, _ = f.WriteString("| Source | Description | MaxSize | EndSize | Puts | Gets\n")
	for _, c := range s.topNByMax {
		_, _ = f.WriteString("| " + CollectionDescriptors[c.Source].SybolicName + "\n")
		_, _ = f.WriteString("| " + c.Description + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.MaxSize) + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.CurSize) + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.Puts) + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.Gets) + "\n")
		_, _ = f.WriteString("\n")
	}
	_, _ = f.WriteString("|===\n\n")

	_, _ = f.WriteString("\n.Summary of Top " + strconv.Itoa(s.topN) + " Collections by Access\n")
	_, _ = f.WriteString(`[cols="<1,<3,>1,>1,>1,>1,>1"]` + "\n\n")
	_, _ = f.WriteString("|===\n")
	_, _ = f.WriteString("| Source | Description | MaxSize | EndSize | Puts | Gets | P+G\n")
	for _, c := range s.topNByUsed {
		_, _ = f.WriteString("| " + CollectionDescriptors[c.Source].SybolicName + "\n")
		_, _ = f.WriteString("| " + c.Description + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.MaxSize) + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.CurSize) + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.Puts) + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.Gets) + "\n")
		_, _ = f.WriteString("| " + strconv.Itoa(c.Gets+c.Puts) + "\n")
		_, _ = f.WriteString("\n")
	}
	_, _ = f.WriteString("|===\n\n")
}

// AddJStatRec adds a [JStatRec] record to the [goRunStats] collection when build runtimeConfig antlr.stats is enabled.
func (s *goRunStats) AddJStatRec(rec *JStatRec) {
	s.jStatsLock.Lock()
	defer s.jStatsLock.Unlock()
	s.jStats = append(s.jStats, rec)
}

// CollectionAnomalies looks through all the statistical records and gathers any anomalies that have been found.
func (s *goRunStats) CollectionAnomalies() {
	s.jStatsLock.RLock()
	defer s.jStatsLock.RUnlock()
	s.counts = make(map[CollectionSource]int, len(s.jStats))
	for _, c := range s.jStats {

		// Accumlate raw counts
		//
		s.counts[c.Source]++

		// Look for allocated but unused collections and count them
		if c.MaxSize == 0 && c.Puts == 0 {
			if s.unusedCollections == nil {
				s.unusedCollections = make(map[CollectionSource]int)
			}
			s.unusedCollections[c.Source]++
		}
		if c.MaxSize > 6000 {
			fmt.Println("Collection ", c.Description, "accumulated a max size of ", c.MaxSize, " - this is probably too large and indicates a poorly formed grammar")
		}
	}

}
