package flagutil

import (
	"flag"
	"io/ioutil"
	"os"
	"testing"
)

var envFile = `
# some secret env vars
MYPROJ_A=foo		
MYPROJ_C=woof
`

func TestSetFlagsFromEnvFile(t *testing.T) {
	fs := flag.NewFlagSet("testing", flag.ExitOnError)
	fs.String("a", "", "")
	fs.String("b", "", "")
	fs.String("c", "", "")
	fs.Parse([]string{})

	// add command-line flags
	if err := fs.Set("b", "bar"); err != nil {
		t.Fatal(err)
	}
	if err := fs.Set("c", "quack"); err != nil {
		t.Fatal(err)
	}

	// first verify that flags are as expected before reading the env
	for f, want := range map[string]string{
		"a": "",
		"b": "bar",
		"c": "quack",
	} {
		if got := fs.Lookup(f).Value.String(); got != want {
			t.Fatalf("flag %q=%q, want %q", f, got, want)
		}
	}

	file, err := ioutil.TempFile("", "env-file")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())
	file.Write([]byte(envFile))

	// read env file and verify flags were updated as expected
	err = SetFlagsFromEnvFile(fs, "MYPROJ", file.Name())
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

func TestSetFlagsFromEnvFile_FlagSetError(t *testing.T) {
	// now verify that an error is propagated
	fs := flag.NewFlagSet("testing", flag.ExitOnError)
	fs.Int("x", 0, "")
	file, err := ioutil.TempFile("", "env-file")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())
	file.Write([]byte("MYPROJ_X=not_a_number"))
	if err := SetFlagsFromEnvFile(fs, "MYPROJ", file.Name()); err == nil {
		t.Errorf("err=nil, want != nil")
	}
}

func TestParseLine(t *testing.T) {
	cases := []struct {
		line        string
		expectedKey string
		expectedVal string
		nilErr      bool
	}{
		{"key=value", "key", "value", true},
		{" key  =  value  	", "key", "value", true},
		{"key='#gopher' #blah", "key", "'#gopher' #blah", true},
		// invalid
		{"key:value", "", "", false},
		{"keyvalue", "", "", false},
	}
	for _, c := range cases {
		key, val, err := parseLine(c.line)
		if (err == nil) != c.nilErr {
			if c.nilErr {
				t.Errorf("got %s, want err=nil", err)
			} else {
				t.Errorf("got err=nil, want err!=nil")
			}
		}
		if c.expectedKey != key || c.expectedVal != val {
			t.Errorf("got %q=%q, want %q=%q", key, val, c.expectedKey, c.expectedVal)
		}
	}
}
