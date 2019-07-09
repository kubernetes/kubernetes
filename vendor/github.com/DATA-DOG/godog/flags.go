package godog

import (
	"flag"
	"fmt"
	"io"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/DATA-DOG/godog/colors"
)

var descFeaturesArgument = "Optional feature(s) to run. Can be:\n" +
	s(4) + "- dir " + colors.Yellow("(features/)") + "\n" +
	s(4) + "- feature " + colors.Yellow("(*.feature)") + "\n" +
	s(4) + "- scenario at specific line " + colors.Yellow("(*.feature:10)") + "\n" +
	"If no feature paths are listed, suite tries " + colors.Yellow("features") + " path by default.\n"

var descConcurrencyOption = "Run the test suite with concurrency level:\n" +
	s(4) + "- " + colors.Yellow(`= 1`) + ": supports all types of formats.\n" +
	s(4) + "- " + colors.Yellow(`>= 2`) + ": only supports " + colors.Yellow("progress") + ". Note, that\n" +
	s(4) + "your context needs to support parallel execution."

var descTagsOption = "Filter scenarios by tags. Expression can be:\n" +
	s(4) + "- " + colors.Yellow(`"@wip"`) + ": run all scenarios with wip tag\n" +
	s(4) + "- " + colors.Yellow(`"~@wip"`) + ": exclude all scenarios with wip tag\n" +
	s(4) + "- " + colors.Yellow(`"@wip && ~@new"`) + ": run wip scenarios, but exclude new\n" +
	s(4) + "- " + colors.Yellow(`"@wip,@undone"`) + ": run wip or undone scenarios"

var descRandomOption = "Randomly shuffle the scenario execution order.\n" +
	"Specify SEED to reproduce the shuffling from a previous run.\n" +
	s(4) + `e.g. ` + colors.Yellow(`--random`) + " or " + colors.Yellow(`--random=5738`)

// FlagSet allows to manage flags by external suite runner
// builds flag.FlagSet with godog flags binded
func FlagSet(opt *Options) *flag.FlagSet {
	set := flag.NewFlagSet("godog", flag.ExitOnError)
	BindFlags("", set, opt)
	set.Usage = usage(set, opt.Output)
	return set
}

// BindFlags binds godog flags to given flag set prefixed
// by given prefix, without overriding usage
func BindFlags(prefix string, set *flag.FlagSet, opt *Options) {
	descFormatOption := "How to format tests output. Built-in formats:\n"
	// @TODO: sort by name
	for name, desc := range AvailableFormatters() {
		descFormatOption += s(4) + "- " + colors.Yellow(name) + ": " + desc + "\n"
	}
	descFormatOption = strings.TrimSpace(descFormatOption)

	// override flag defaults if any corresponding properties were supplied on the incoming `opt`
	defFormatOption := "pretty"
	if opt.Format != "" {
		defFormatOption = opt.Format
	}
	defTagsOption := ""
	if opt.Tags != "" {
		defTagsOption = opt.Tags
	}
	defConcurrencyOption := 1
	if opt.Concurrency != 0 {
		defConcurrencyOption = opt.Concurrency
	}
	defShowStepDefinitions := false
	if opt.ShowStepDefinitions {
		defShowStepDefinitions = opt.ShowStepDefinitions
	}
	defStopOnFailure := false
	if opt.StopOnFailure {
		defStopOnFailure = opt.StopOnFailure
	}
	defStrict := false
	if opt.Strict {
		defStrict = opt.Strict
	}
	defNoColors := false
	if opt.NoColors {
		defNoColors = opt.NoColors
	}

	set.StringVar(&opt.Format, prefix+"format", defFormatOption, descFormatOption)
	set.StringVar(&opt.Format, prefix+"f", defFormatOption, descFormatOption)
	set.StringVar(&opt.Tags, prefix+"tags", defTagsOption, descTagsOption)
	set.StringVar(&opt.Tags, prefix+"t", defTagsOption, descTagsOption)
	set.IntVar(&opt.Concurrency, prefix+"concurrency", defConcurrencyOption, descConcurrencyOption)
	set.IntVar(&opt.Concurrency, prefix+"c", defConcurrencyOption, descConcurrencyOption)
	set.BoolVar(&opt.ShowStepDefinitions, prefix+"definitions", defShowStepDefinitions, "Print all available step definitions.")
	set.BoolVar(&opt.ShowStepDefinitions, prefix+"d", defShowStepDefinitions, "Print all available step definitions.")
	set.BoolVar(&opt.StopOnFailure, prefix+"stop-on-failure", defStopOnFailure, "Stop processing on first failed scenario.")
	set.BoolVar(&opt.Strict, prefix+"strict", defStrict, "Fail suite when there are pending or undefined steps.")
	set.BoolVar(&opt.NoColors, prefix+"no-colors", defNoColors, "Disable ansi colors.")
	set.Var(&randomSeed{&opt.Randomize}, prefix+"random", descRandomOption)
}

type flagged struct {
	short, long, descr, dflt string
}

func (f *flagged) name() string {
	var name string
	switch {
	case len(f.short) > 0 && len(f.long) > 0:
		name = fmt.Sprintf("-%s, --%s", f.short, f.long)
	case len(f.long) > 0:
		name = fmt.Sprintf("--%s", f.long)
	case len(f.short) > 0:
		name = fmt.Sprintf("-%s", f.short)
	}

	if f.long == "random" {
		// `random` is special in that we will later assign it randomly
		// if the user specifies `--random` without specifying one,
		// so mask the "default" value here to avoid UI confusion about
		// what the value will end up being.
		name += "[=SEED]"
	} else if f.dflt != "true" && f.dflt != "false" {
		name += "=" + f.dflt
	}
	return name
}

func usage(set *flag.FlagSet, w io.Writer) func() {
	return func() {
		var list []*flagged
		var longest int
		set.VisitAll(func(f *flag.Flag) {
			var fl *flagged
			for _, flg := range list {
				if flg.descr == f.Usage {
					fl = flg
					break
				}
			}
			if nil == fl {
				fl = &flagged{
					dflt:  f.DefValue,
					descr: f.Usage,
				}
				list = append(list, fl)
			}
			if len(f.Name) > 2 {
				fl.long = f.Name
			} else {
				fl.short = f.Name
			}
		})

		for _, f := range list {
			if len(f.name()) > longest {
				longest = len(f.name())
			}
		}

		// prints an option or argument with a description, or only description
		opt := func(name, desc string) string {
			var ret []string
			lines := strings.Split(desc, "\n")
			ret = append(ret, s(2)+colors.Green(name)+s(longest+2-len(name))+lines[0])
			if len(lines) > 1 {
				for _, ln := range lines[1:] {
					ret = append(ret, s(2)+s(longest+2)+ln)
				}
			}
			return strings.Join(ret, "\n")
		}

		// --- GENERAL ---
		fmt.Fprintln(w, colors.Yellow("Usage:"))
		fmt.Fprintf(w, s(2)+"godog [options] [<features>]\n\n")
		// description
		fmt.Fprintln(w, "Builds a test package and runs given feature files.")
		fmt.Fprintf(w, "Command should be run from the directory of tested package and contain buildable go source.\n\n")

		// --- ARGUMENTS ---
		fmt.Fprintln(w, colors.Yellow("Arguments:"))
		// --> features
		fmt.Fprintln(w, opt("features", descFeaturesArgument))

		// --- OPTIONS ---
		fmt.Fprintln(w, colors.Yellow("Options:"))
		for _, f := range list {
			fmt.Fprintln(w, opt(f.name(), f.descr))
		}
		fmt.Fprintln(w, "")
	}
}

// randomSeed implements `flag.Value`, see https://golang.org/pkg/flag/#Value
type randomSeed struct {
	ref *int64
}

// Choose randomly assigns a convenient pseudo-random seed value.
// The resulting seed will be between `1-99999` for later ease of specification.
func makeRandomSeed() int64 {
	return rand.New(rand.NewSource(time.Now().UTC().UnixNano())).Int63n(99998) + 1
}

func (rs *randomSeed) Set(s string) error {
	if s == "true" {
		*rs.ref = makeRandomSeed()
		return nil
	}

	if s == "false" {
		*rs.ref = 0
		return nil
	}

	i, err := strconv.ParseInt(s, 10, 64)
	*rs.ref = i
	return err
}

func (rs *randomSeed) String() string {
	if rs.ref == nil {
		return "0"
	}
	return strconv.FormatInt(*rs.ref, 10)
}

// If a Value has an IsBoolFlag() bool method returning true, the command-line
// parser makes -name equivalent to -name=true rather than using the next
// command-line argument.
func (rs *randomSeed) IsBoolFlag() bool {
	return *rs.ref == 0
}
