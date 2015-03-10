package cli_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/codegangsta/cli"
)

func ExampleApp() {
	// set args for examples sake
	os.Args = []string{"greet", "--name", "Jeremy"}

	app := cli.NewApp()
	app.Name = "greet"
	app.Flags = []cli.Flag{
		cli.StringFlag{Name: "name", Value: "bob", Usage: "a name to say"},
	}
	app.Action = func(c *cli.Context) {
		fmt.Printf("Hello %v\n", c.String("name"))
	}
	app.Run(os.Args)
	// Output:
	// Hello Jeremy
}

func ExampleAppSubcommand() {
	// set args for examples sake
	os.Args = []string{"say", "hi", "english", "--name", "Jeremy"}
	app := cli.NewApp()
	app.Name = "say"
	app.Commands = []cli.Command{
		{
			Name:        "hello",
			ShortName:   "hi",
			Usage:       "use it to see a description",
			Description: "This is how we describe hello the function",
			Subcommands: []cli.Command{
				{
					Name:        "english",
					ShortName:   "en",
					Usage:       "sends a greeting in english",
					Description: "greets someone in english",
					Flags: []cli.Flag{
						cli.StringFlag{Name: "name", Value: "Bob", Usage: "Name of the person to greet"},
					},
					Action: func(c *cli.Context) {
						fmt.Println("Hello,", c.String("name"))
					},
				},
			},
		},
	}

	app.Run(os.Args)
	// Output:
	// Hello, Jeremy
}

func ExampleAppHelp() {
	// set args for examples sake
	os.Args = []string{"greet", "h", "describeit"}

	app := cli.NewApp()
	app.Name = "greet"
	app.Flags = []cli.Flag{
		cli.StringFlag{Name: "name", Value: "bob", Usage: "a name to say"},
	}
	app.Commands = []cli.Command{
		{
			Name:        "describeit",
			ShortName:   "d",
			Usage:       "use it to see a description",
			Description: "This is how we describe describeit the function",
			Action: func(c *cli.Context) {
				fmt.Printf("i like to describe things")
			},
		},
	}
	app.Run(os.Args)
	// Output:
	// NAME:
	//    describeit - use it to see a description
	//
	// USAGE:
	//    command describeit [arguments...]
	//
	// DESCRIPTION:
	//    This is how we describe describeit the function
}

func ExampleAppBashComplete() {
	// set args for examples sake
	os.Args = []string{"greet", "--generate-bash-completion"}

	app := cli.NewApp()
	app.Name = "greet"
	app.EnableBashCompletion = true
	app.Commands = []cli.Command{
		{
			Name:        "describeit",
			ShortName:   "d",
			Usage:       "use it to see a description",
			Description: "This is how we describe describeit the function",
			Action: func(c *cli.Context) {
				fmt.Printf("i like to describe things")
			},
		}, {
			Name:        "next",
			Usage:       "next example",
			Description: "more stuff to see when generating bash completion",
			Action: func(c *cli.Context) {
				fmt.Printf("the next example")
			},
		},
	}

	app.Run(os.Args)
	// Output:
	// describeit
	// d
	// next
	// help
	// h
}

func TestApp_Run(t *testing.T) {
	s := ""

	app := cli.NewApp()
	app.Action = func(c *cli.Context) {
		s = s + c.Args().First()
	}

	err := app.Run([]string{"command", "foo"})
	expect(t, err, nil)
	err = app.Run([]string{"command", "bar"})
	expect(t, err, nil)
	expect(t, s, "foobar")
}

var commandAppTests = []struct {
	name     string
	expected bool
}{
	{"foobar", true},
	{"batbaz", true},
	{"b", true},
	{"f", true},
	{"bat", false},
	{"nothing", false},
}

func TestApp_Command(t *testing.T) {
	app := cli.NewApp()
	fooCommand := cli.Command{Name: "foobar", ShortName: "f"}
	batCommand := cli.Command{Name: "batbaz", ShortName: "b"}
	app.Commands = []cli.Command{
		fooCommand,
		batCommand,
	}

	for _, test := range commandAppTests {
		expect(t, app.Command(test.name) != nil, test.expected)
	}
}

func TestApp_CommandWithArgBeforeFlags(t *testing.T) {
	var parsedOption, firstArg string

	app := cli.NewApp()
	command := cli.Command{
		Name: "cmd",
		Flags: []cli.Flag{
			cli.StringFlag{Name: "option", Value: "", Usage: "some option"},
		},
		Action: func(c *cli.Context) {
			parsedOption = c.String("option")
			firstArg = c.Args().First()
		},
	}
	app.Commands = []cli.Command{command}

	app.Run([]string{"", "cmd", "my-arg", "--option", "my-option"})

	expect(t, parsedOption, "my-option")
	expect(t, firstArg, "my-arg")
}

func TestApp_Float64Flag(t *testing.T) {
	var meters float64

	app := cli.NewApp()
	app.Flags = []cli.Flag{
		cli.Float64Flag{Name: "height", Value: 1.5, Usage: "Set the height, in meters"},
	}
	app.Action = func(c *cli.Context) {
		meters = c.Float64("height")
	}

	app.Run([]string{"", "--height", "1.93"})
	expect(t, meters, 1.93)
}

func TestApp_ParseSliceFlags(t *testing.T) {
	var parsedOption, firstArg string
	var parsedIntSlice []int
	var parsedStringSlice []string

	app := cli.NewApp()
	command := cli.Command{
		Name: "cmd",
		Flags: []cli.Flag{
			cli.IntSliceFlag{Name: "p", Value: &cli.IntSlice{}, Usage: "set one or more ip addr"},
			cli.StringSliceFlag{Name: "ip", Value: &cli.StringSlice{}, Usage: "set one or more ports to open"},
		},
		Action: func(c *cli.Context) {
			parsedIntSlice = c.IntSlice("p")
			parsedStringSlice = c.StringSlice("ip")
			parsedOption = c.String("option")
			firstArg = c.Args().First()
		},
	}
	app.Commands = []cli.Command{command}

	app.Run([]string{"", "cmd", "my-arg", "-p", "22", "-p", "80", "-ip", "8.8.8.8", "-ip", "8.8.4.4"})

	IntsEquals := func(a, b []int) bool {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	StrsEquals := func(a, b []string) bool {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	var expectedIntSlice = []int{22, 80}
	var expectedStringSlice = []string{"8.8.8.8", "8.8.4.4"}

	if !IntsEquals(parsedIntSlice, expectedIntSlice) {
		t.Errorf("%v does not match %v", parsedIntSlice, expectedIntSlice)
	}

	if !StrsEquals(parsedStringSlice, expectedStringSlice) {
		t.Errorf("%v does not match %v", parsedStringSlice, expectedStringSlice)
	}
}

func TestApp_BeforeFunc(t *testing.T) {
	beforeRun, subcommandRun := false, false
	beforeError := fmt.Errorf("fail")
	var err error

	app := cli.NewApp()

	app.Before = func(c *cli.Context) error {
		beforeRun = true
		s := c.String("opt")
		if s == "fail" {
			return beforeError
		}

		return nil
	}

	app.Commands = []cli.Command{
		cli.Command{
			Name: "sub",
			Action: func(c *cli.Context) {
				subcommandRun = true
			},
		},
	}

	app.Flags = []cli.Flag{
		cli.StringFlag{Name: "opt"},
	}

	// run with the Before() func succeeding
	err = app.Run([]string{"command", "--opt", "succeed", "sub"})

	if err != nil {
		t.Fatalf("Run error: %s", err)
	}

	if beforeRun == false {
		t.Errorf("Before() not executed when expected")
	}

	if subcommandRun == false {
		t.Errorf("Subcommand not executed when expected")
	}

	// reset
	beforeRun, subcommandRun = false, false

	// run with the Before() func failing
	err = app.Run([]string{"command", "--opt", "fail", "sub"})

	// should be the same error produced by the Before func
	if err != beforeError {
		t.Errorf("Run error expected, but not received")
	}

	if beforeRun == false {
		t.Errorf("Before() not executed when expected")
	}

	if subcommandRun == true {
		t.Errorf("Subcommand executed when NOT expected")
	}

}

func TestAppHelpPrinter(t *testing.T) {
	oldPrinter := cli.HelpPrinter
	defer func() {
		cli.HelpPrinter = oldPrinter
	}()

	var wasCalled = false
	cli.HelpPrinter = func(template string, data interface{}) {
		wasCalled = true
	}

	app := cli.NewApp()
	app.Run([]string{"-h"})

	if wasCalled == false {
		t.Errorf("Help printer expected to be called, but was not")
	}
}

func TestAppCommandNotFound(t *testing.T) {
	beforeRun, subcommandRun := false, false
	app := cli.NewApp()

	app.CommandNotFound = func(c *cli.Context, command string) {
		beforeRun = true
	}

	app.Commands = []cli.Command{
		cli.Command{
			Name: "bar",
			Action: func(c *cli.Context) {
				subcommandRun = true
			},
		},
	}

	app.Run([]string{"command", "foo"})

	expect(t, beforeRun, true)
	expect(t, subcommandRun, false)
}

func TestGlobalFlagsInSubcommands(t *testing.T) {
	subcommandRun := false
	app := cli.NewApp()

	app.Flags = []cli.Flag{
		cli.BoolFlag{Name: "debug, d", Usage: "Enable debugging"},
	}

	app.Commands = []cli.Command{
		cli.Command{
			Name: "foo",
			Subcommands: []cli.Command{
				{
					Name: "bar",
					Action: func(c *cli.Context) {
						if c.GlobalBool("debug") {
							subcommandRun = true
						}
					},
				},
			},
		},
	}

	app.Run([]string{"command", "-d", "foo", "bar"})

	expect(t, subcommandRun, true)
}
