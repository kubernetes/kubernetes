package cli_test

import (
	"github.com/codegangsta/cli"

	"fmt"
	"reflect"
	"strings"
	"testing"
)

var boolFlagTests = []struct {
	name     string
	expected string
}{
	{"help", "--help\t"},
	{"h", "-h\t"},
}

func TestBoolFlagHelpOutput(t *testing.T) {

	for _, test := range boolFlagTests {
		flag := cli.BoolFlag{Name: test.name}
		output := flag.String()

		if output != test.expected {
			t.Errorf("%s does not match %s", output, test.expected)
		}
	}
}

var stringFlagTests = []struct {
	name     string
	value    string
	expected string
}{
	{"help", "", "--help \t"},
	{"h", "", "-h \t"},
	{"h", "", "-h \t"},
	{"test", "Something", "--test 'Something'\t"},
}

func TestStringFlagHelpOutput(t *testing.T) {

	for _, test := range stringFlagTests {
		flag := cli.StringFlag{Name: test.name, Value: test.value}
		output := flag.String()

		if output != test.expected {
			t.Errorf("%s does not match %s", output, test.expected)
		}
	}
}

var intFlagTests = []struct {
	name     string
	expected string
}{
	{"help", "--help '0'\t"},
	{"h", "-h '0'\t"},
}

func TestIntFlagHelpOutput(t *testing.T) {

	for _, test := range intFlagTests {
		flag := cli.IntFlag{Name: test.name}
		output := flag.String()

		if output != test.expected {
			t.Errorf("%s does not match %s", output, test.expected)
		}
	}
}

var float64FlagTests = []struct {
	name     string
	expected string
}{
	{"help", "--help '0'\t"},
	{"h", "-h '0'\t"},
}

func TestFloat64FlagHelpOutput(t *testing.T) {

	for _, test := range float64FlagTests {
		flag := cli.Float64Flag{Name: test.name}
		output := flag.String()

		if output != test.expected {
			t.Errorf("%s does not match %s", output, test.expected)
		}
	}
}

func TestParseMultiString(t *testing.T) {
	(&cli.App{
		Flags: []cli.Flag{
			cli.StringFlag{Name: "serve, s"},
		},
		Action: func(ctx *cli.Context) {
			if ctx.String("serve") != "10" {
				t.Errorf("main name not set")
			}
			if ctx.String("s") != "10" {
				t.Errorf("short name not set")
			}
		},
	}).Run([]string{"run", "-s", "10"})
}

func TestParseMultiStringSlice(t *testing.T) {
	(&cli.App{
		Flags: []cli.Flag{
			cli.StringSliceFlag{Name: "serve, s", Value: &cli.StringSlice{}},
		},
		Action: func(ctx *cli.Context) {
			if !reflect.DeepEqual(ctx.StringSlice("serve"), []string{"10", "20"}) {
				t.Errorf("main name not set")
			}
			if !reflect.DeepEqual(ctx.StringSlice("s"), []string{"10", "20"}) {
				t.Errorf("short name not set")
			}
		},
	}).Run([]string{"run", "-s", "10", "-s", "20"})
}

func TestParseMultiInt(t *testing.T) {
	a := cli.App{
		Flags: []cli.Flag{
			cli.IntFlag{Name: "serve, s"},
		},
		Action: func(ctx *cli.Context) {
			if ctx.Int("serve") != 10 {
				t.Errorf("main name not set")
			}
			if ctx.Int("s") != 10 {
				t.Errorf("short name not set")
			}
		},
	}
	a.Run([]string{"run", "-s", "10"})
}

func TestParseMultiBool(t *testing.T) {
	a := cli.App{
		Flags: []cli.Flag{
			cli.BoolFlag{Name: "serve, s"},
		},
		Action: func(ctx *cli.Context) {
			if ctx.Bool("serve") != true {
				t.Errorf("main name not set")
			}
			if ctx.Bool("s") != true {
				t.Errorf("short name not set")
			}
		},
	}
	a.Run([]string{"run", "--serve"})
}

type Parser [2]string

func (p *Parser) Set(value string) error {
	parts := strings.Split(value, ",")
	if len(parts) != 2 {
		return fmt.Errorf("invalid format")
	}

	(*p)[0] = parts[0]
	(*p)[1] = parts[1]

	return nil
}

func (p *Parser) String() string {
	return fmt.Sprintf("%s,%s", p[0], p[1])
}

func TestParseGeneric(t *testing.T) {
	a := cli.App{
		Flags: []cli.Flag{
			cli.GenericFlag{Name: "serve, s", Value: &Parser{}},
		},
		Action: func(ctx *cli.Context) {
			if !reflect.DeepEqual(ctx.Generic("serve"), &Parser{"10", "20"}) {
				t.Errorf("main name not set")
			}
			if !reflect.DeepEqual(ctx.Generic("s"), &Parser{"10", "20"}) {
				t.Errorf("short name not set")
			}
		},
	}
	a.Run([]string{"run", "-s", "10,20"})
}
