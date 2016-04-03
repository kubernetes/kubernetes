package yamlutil

import (
	"flag"
	"testing"
)

func TestSetFlagsFromYaml(t *testing.T) {
	config := "A: foo\nC: woof"
	fs := flag.NewFlagSet("testing", flag.ExitOnError)
	fs.String("a", "", "")
	fs.String("b", "", "")
	fs.String("c", "", "")
	fs.Parse([]string{})

	// flags should be settable using yaml vars
	// and command-line flags
	if err := fs.Set("b", "bar"); err != nil {
		t.Fatal(err)
	}
	// command-line flags take precedence over the file
	if err := fs.Set("c", "quack"); err != nil {
		t.Fatal(err)
	}

	// first verify that flags are as expected before reading the file
	for f, want := range map[string]string{
		"a": "",
		"b": "bar",
		"c": "quack",
	} {
		if got := fs.Lookup(f).Value.String(); got != want {
			t.Fatalf("flag %q=%q, want %q", f, got, want)
		}
	}

	// now read the yaml and verify flags were updated as expected
	err := SetFlagsFromYaml(fs, []byte(config))
	if err != nil {
		t.Errorf("err=%v, want nil", err)
	}
	for f, want := range map[string]string{
		"a": "foo",
		"b": "bar",
		"c": "quack",
	} {
		if got := fs.Lookup(f).Value.String(); got != want {
			t.Errorf("flag %q=%q, want %q", f, got, want)
		}
	}
}

func TestSetFlagsFromYamlBad(t *testing.T) {
	// now verify that an error is propagated
	fs := flag.NewFlagSet("testing", flag.ExitOnError)
	fs.Int("x", 0, "")
	badConf := "X: not_a_number"
	if err := SetFlagsFromYaml(fs, []byte(badConf)); err == nil {
		t.Errorf("got err=nil, flag x=%q, want err != nil", fs.Lookup("x").Value.String())
	}
}

func TestSetFlagsFromYamlMultiError(t *testing.T) {
	fs := flag.NewFlagSet("testing", flag.ExitOnError)
	fs.Int("x", 0, "")
	fs.Int("y", 0, "")
	fs.Int("z", 0, "")
	conf := "X: foo\nY: bar\nZ: 3"
	err := SetFlagsFromYaml(fs, []byte(conf))
	if err == nil {
		t.Errorf("got err= nil, want err != nil")
	}
	es, ok := err.(ErrorSlice)
	if !ok {
		t.Errorf("Got ok=false want ok=true")
	}
	if len(es) != 2 {
		t.Errorf("2 errors should be contained in the error, got %d errors", len(es))
	}
}
