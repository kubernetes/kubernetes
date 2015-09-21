package cli_test

import (
	"flag"
	"github.com/codegangsta/cli"
	"testing"
)

func TestNewContext(t *testing.T) {
	set := flag.NewFlagSet("test", 0)
	set.Int("myflag", 12, "doc")
	globalSet := flag.NewFlagSet("test", 0)
	globalSet.Int("myflag", 42, "doc")
	command := cli.Command{Name: "mycommand"}
	c := cli.NewContext(nil, set, globalSet)
	c.Command = command
	expect(t, c.Int("myflag"), 12)
	expect(t, c.GlobalInt("myflag"), 42)
	expect(t, c.Command.Name, "mycommand")
}

func TestContext_Int(t *testing.T) {
	set := flag.NewFlagSet("test", 0)
	set.Int("myflag", 12, "doc")
	c := cli.NewContext(nil, set, set)
	expect(t, c.Int("myflag"), 12)
}

func TestContext_String(t *testing.T) {
	set := flag.NewFlagSet("test", 0)
	set.String("myflag", "hello world", "doc")
	c := cli.NewContext(nil, set, set)
	expect(t, c.String("myflag"), "hello world")
}

func TestContext_Bool(t *testing.T) {
	set := flag.NewFlagSet("test", 0)
	set.Bool("myflag", false, "doc")
	c := cli.NewContext(nil, set, set)
	expect(t, c.Bool("myflag"), false)
}

func TestContext_BoolT(t *testing.T) {
	set := flag.NewFlagSet("test", 0)
	set.Bool("myflag", true, "doc")
	c := cli.NewContext(nil, set, set)
	expect(t, c.BoolT("myflag"), true)
}

func TestContext_Args(t *testing.T) {
	set := flag.NewFlagSet("test", 0)
	set.Bool("myflag", false, "doc")
	c := cli.NewContext(nil, set, set)
	set.Parse([]string{"--myflag", "bat", "baz"})
	expect(t, len(c.Args()), 2)
	expect(t, c.Bool("myflag"), true)
}

func TestContext_IsSet(t *testing.T) {
	set := flag.NewFlagSet("test", 0)
	set.Bool("myflag", false, "doc")
	set.String("otherflag", "hello world", "doc")
	c := cli.NewContext(nil, set, set)
	set.Parse([]string{"--myflag", "bat", "baz"})
	expect(t, c.IsSet("myflag"), true)
	expect(t, c.IsSet("otherflag"), false)
	expect(t, c.IsSet("bogusflag"), false)
}
