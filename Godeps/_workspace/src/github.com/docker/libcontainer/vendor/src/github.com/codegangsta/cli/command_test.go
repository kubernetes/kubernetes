package cli_test

import (
	"flag"
	"github.com/codegangsta/cli"
	"testing"
)

func TestCommandDoNotIgnoreFlags(t *testing.T) {
	app := cli.NewApp()
	set := flag.NewFlagSet("test", 0)
	test := []string{"blah", "blah", "-break"}
	set.Parse(test)

	c := cli.NewContext(app, set, set)

	command := cli.Command {
		Name: "test-cmd",
		ShortName: "tc",
		Usage: "this is for testing",
		Description: "testing",
		Action: func(_ *cli.Context) { },
	}
	err := command.Run(c)

	expect(t, err.Error(), "flag provided but not defined: -break")
}

func TestCommandIgnoreFlags(t *testing.T) {
	app := cli.NewApp()
	set := flag.NewFlagSet("test", 0)
	test := []string{"blah", "blah"}
	set.Parse(test)

	c := cli.NewContext(app, set, set)

	command := cli.Command {
		Name: "test-cmd",
		ShortName: "tc",
		Usage: "this is for testing",
		Description: "testing",
		Action: func(_ *cli.Context) { },
		SkipFlagParsing: true,
	}
	err := command.Run(c)

	expect(t, err, nil)
}
