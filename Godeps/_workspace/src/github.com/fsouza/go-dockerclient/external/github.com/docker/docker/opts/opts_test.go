package opts

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestValidateIPAddress(t *testing.T) {
	if ret, err := ValidateIPAddress(`1.2.3.4`); err != nil || ret == "" {
		t.Fatalf("ValidateIPAddress(`1.2.3.4`) got %s %s", ret, err)
	}

	if ret, err := ValidateIPAddress(`127.0.0.1`); err != nil || ret == "" {
		t.Fatalf("ValidateIPAddress(`127.0.0.1`) got %s %s", ret, err)
	}

	if ret, err := ValidateIPAddress(`::1`); err != nil || ret == "" {
		t.Fatalf("ValidateIPAddress(`::1`) got %s %s", ret, err)
	}

	if ret, err := ValidateIPAddress(`127`); err == nil || ret != "" {
		t.Fatalf("ValidateIPAddress(`127`) got %s %s", ret, err)
	}

	if ret, err := ValidateIPAddress(`random invalid string`); err == nil || ret != "" {
		t.Fatalf("ValidateIPAddress(`random invalid string`) got %s %s", ret, err)
	}

}

func TestMapOpts(t *testing.T) {
	tmpMap := make(map[string]string)
	o := NewMapOpts(tmpMap, logOptsValidator)
	o.Set("max-size=1")
	if o.String() != "map[max-size:1]" {
		t.Errorf("%s != [map[max-size:1]", o.String())
	}

	o.Set("max-file=2")
	if len(tmpMap) != 2 {
		t.Errorf("map length %d != 2", len(tmpMap))
	}

	if tmpMap["max-file"] != "2" {
		t.Errorf("max-file = %s != 2", tmpMap["max-file"])
	}

	if tmpMap["max-size"] != "1" {
		t.Errorf("max-size = %s != 1", tmpMap["max-size"])
	}
	if o.Set("dummy-val=3") == nil {
		t.Errorf("validator is not being called")
	}
}

func TestValidateMACAddress(t *testing.T) {
	if _, err := ValidateMACAddress(`92:d0:c6:0a:29:33`); err != nil {
		t.Fatalf("ValidateMACAddress(`92:d0:c6:0a:29:33`) got %s", err)
	}

	if _, err := ValidateMACAddress(`92:d0:c6:0a:33`); err == nil {
		t.Fatalf("ValidateMACAddress(`92:d0:c6:0a:33`) succeeded; expected failure on invalid MAC")
	}

	if _, err := ValidateMACAddress(`random invalid string`); err == nil {
		t.Fatalf("ValidateMACAddress(`random invalid string`) succeeded; expected failure on invalid MAC")
	}
}

func TestListOptsWithoutValidator(t *testing.T) {
	o := NewListOpts(nil)
	o.Set("foo")
	if o.String() != "[foo]" {
		t.Errorf("%s != [foo]", o.String())
	}
	o.Set("bar")
	if o.Len() != 2 {
		t.Errorf("%d != 2", o.Len())
	}
	o.Set("bar")
	if o.Len() != 3 {
		t.Errorf("%d != 3", o.Len())
	}
	if !o.Get("bar") {
		t.Error("o.Get(\"bar\") == false")
	}
	if o.Get("baz") {
		t.Error("o.Get(\"baz\") == true")
	}
	o.Delete("foo")
	if o.String() != "[bar bar]" {
		t.Errorf("%s != [bar bar]", o.String())
	}
	listOpts := o.GetAll()
	if len(listOpts) != 2 || listOpts[0] != "bar" || listOpts[1] != "bar" {
		t.Errorf("Expected [[bar bar]], got [%v]", listOpts)
	}
	mapListOpts := o.GetMap()
	if len(mapListOpts) != 1 {
		t.Errorf("Expected [map[bar:{}]], got [%v]", mapListOpts)
	}

}

func TestListOptsWithValidator(t *testing.T) {
	// Re-using logOptsvalidator (used by MapOpts)
	o := NewListOpts(logOptsValidator)
	o.Set("foo")
	if o.String() != "[]" {
		t.Errorf("%s != []", o.String())
	}
	o.Set("foo=bar")
	if o.String() != "[]" {
		t.Errorf("%s != []", o.String())
	}
	o.Set("max-file=2")
	if o.Len() != 1 {
		t.Errorf("%d != 1", o.Len())
	}
	if !o.Get("max-file=2") {
		t.Error("o.Get(\"max-file=2\") == false")
	}
	if o.Get("baz") {
		t.Error("o.Get(\"baz\") == true")
	}
	o.Delete("max-file=2")
	if o.String() != "[]" {
		t.Errorf("%s != []", o.String())
	}
}

func TestValidateDNSSearch(t *testing.T) {
	valid := []string{
		`.`,
		`a`,
		`a.`,
		`1.foo`,
		`17.foo`,
		`foo.bar`,
		`foo.bar.baz`,
		`foo.bar.`,
		`foo.bar.baz`,
		`foo1.bar2`,
		`foo1.bar2.baz`,
		`1foo.2bar.`,
		`1foo.2bar.baz`,
		`foo-1.bar-2`,
		`foo-1.bar-2.baz`,
		`foo-1.bar-2.`,
		`foo-1.bar-2.baz`,
		`1-foo.2-bar`,
		`1-foo.2-bar.baz`,
		`1-foo.2-bar.`,
		`1-foo.2-bar.baz`,
	}

	invalid := []string{
		``,
		` `,
		`  `,
		`17`,
		`17.`,
		`.17`,
		`17-.`,
		`17-.foo`,
		`.foo`,
		`foo-.bar`,
		`-foo.bar`,
		`foo.bar-`,
		`foo.bar-.baz`,
		`foo.-bar`,
		`foo.-bar.baz`,
		`foo.bar.baz.this.should.fail.on.long.name.beause.it.is.longer.thanisshouldbethis.should.fail.on.long.name.beause.it.is.longer.thanisshouldbethis.should.fail.on.long.name.beause.it.is.longer.thanisshouldbethis.should.fail.on.long.name.beause.it.is.longer.thanisshouldbe`,
	}

	for _, domain := range valid {
		if ret, err := ValidateDNSSearch(domain); err != nil || ret == "" {
			t.Fatalf("ValidateDNSSearch(`"+domain+"`) got %s %s", ret, err)
		}
	}

	for _, domain := range invalid {
		if ret, err := ValidateDNSSearch(domain); err == nil || ret != "" {
			t.Fatalf("ValidateDNSSearch(`"+domain+"`) got %s %s", ret, err)
		}
	}
}

func TestValidateExtraHosts(t *testing.T) {
	valid := []string{
		`myhost:192.168.0.1`,
		`thathost:10.0.2.1`,
		`anipv6host:2003:ab34:e::1`,
		`ipv6local:::1`,
	}

	invalid := map[string]string{
		`myhost:192.notanipaddress.1`:  `invalid IP`,
		`thathost-nosemicolon10.0.0.1`: `bad format`,
		`anipv6host:::::1`:             `invalid IP`,
		`ipv6local:::0::`:              `invalid IP`,
	}

	for _, extrahost := range valid {
		if _, err := ValidateExtraHost(extrahost); err != nil {
			t.Fatalf("ValidateExtraHost(`"+extrahost+"`) should succeed: error %v", err)
		}
	}

	for extraHost, expectedError := range invalid {
		if _, err := ValidateExtraHost(extraHost); err == nil {
			t.Fatalf("ValidateExtraHost(`%q`) should have failed validation", extraHost)
		} else {
			if !strings.Contains(err.Error(), expectedError) {
				t.Fatalf("ValidateExtraHost(`%q`) error should contain %q", extraHost, expectedError)
			}
		}
	}
}

func TestValidateAttach(t *testing.T) {
	valid := []string{
		"stdin",
		"stdout",
		"stderr",
		"STDIN",
		"STDOUT",
		"STDERR",
	}
	if _, err := ValidateAttach("invalid"); err == nil {
		t.Fatalf("Expected error with [valid streams are STDIN, STDOUT and STDERR], got nothing")
	}

	for _, attach := range valid {
		value, err := ValidateAttach(attach)
		if err != nil {
			t.Fatal(err)
		}
		if value != strings.ToLower(attach) {
			t.Fatalf("Expected [%v], got [%v]", attach, value)
		}
	}
}

func TestValidateLink(t *testing.T) {
	valid := []string{
		"name",
		"dcdfbe62ecd0:alias",
		"7a67485460b7642516a4ad82ecefe7f57d0c4916f530561b71a50a3f9c4e33da",
		"angry_torvalds:linus",
	}
	invalid := map[string]string{
		"":               "empty string specified for links",
		"too:much:of:it": "bad format for links: too:much:of:it",
	}

	for _, link := range valid {
		if _, err := ValidateLink(link); err != nil {
			t.Fatalf("ValidateLink(`%q`) should succeed: error %q", link, err)
		}
	}

	for link, expectedError := range invalid {
		if _, err := ValidateLink(link); err == nil {
			t.Fatalf("ValidateLink(`%q`) should have failed validation", link)
		} else {
			if !strings.Contains(err.Error(), expectedError) {
				t.Fatalf("ValidateLink(`%q`) error should contain %q", link, expectedError)
			}
		}
	}
}

func TestValidatePath(t *testing.T) {
	valid := []string{
		"/home",
		"/home:/home",
		"/home:/something/else",
		"/with space",
		"/home:/with space",
		"relative:/absolute-path",
		"hostPath:/containerPath:ro",
		"/hostPath:/containerPath:rw",
		"/rw:/ro",
		"/path:rw",
		"/path:ro",
		"/rw:rw",
	}
	invalid := map[string]string{
		"":                "bad format for volumes: ",
		"./":              "./ is not an absolute path",
		"../":             "../ is not an absolute path",
		"/:../":           "../ is not an absolute path",
		"/:path":          "path is not an absolute path",
		":":               "bad format for volumes: :",
		"/tmp:":           " is not an absolute path",
		":test":           "bad format for volumes: :test",
		":/test":          "bad format for volumes: :/test",
		"tmp:":            " is not an absolute path",
		":test:":          "bad format for volumes: :test:",
		"::":              "bad format for volumes: ::",
		":::":             "bad format for volumes: :::",
		"/tmp:::":         "bad format for volumes: /tmp:::",
		":/tmp::":         "bad format for volumes: :/tmp::",
		"path:ro":         "path is not an absolute path",
		"/path:/path:sw":  "bad mount mode specified : sw",
		"/path:/path:rwz": "bad mount mode specified : rwz",
	}

	for _, path := range valid {
		if _, err := ValidatePath(path); err != nil {
			t.Fatalf("ValidatePath(`%q`) should succeed: error %q", path, err)
		}
	}

	for path, expectedError := range invalid {
		if _, err := ValidatePath(path); err == nil {
			t.Fatalf("ValidatePath(`%q`) should have failed validation", path)
		} else {
			if err.Error() != expectedError {
				t.Fatalf("ValidatePath(`%q`) error should contain %q, got %q", path, expectedError, err.Error())
			}
		}
	}
}
func TestValidateDevice(t *testing.T) {
	valid := []string{
		"/home",
		"/home:/home",
		"/home:/something/else",
		"/with space",
		"/home:/with space",
		"relative:/absolute-path",
		"hostPath:/containerPath:ro",
		"/hostPath:/containerPath:rw",
		"/hostPath:/containerPath:mrw",
	}
	invalid := map[string]string{
		"":        "bad format for volumes: ",
		"./":      "./ is not an absolute path",
		"../":     "../ is not an absolute path",
		"/:../":   "../ is not an absolute path",
		"/:path":  "path is not an absolute path",
		":":       "bad format for volumes: :",
		"/tmp:":   " is not an absolute path",
		":test":   "bad format for volumes: :test",
		":/test":  "bad format for volumes: :/test",
		"tmp:":    " is not an absolute path",
		":test:":  "bad format for volumes: :test:",
		"::":      "bad format for volumes: ::",
		":::":     "bad format for volumes: :::",
		"/tmp:::": "bad format for volumes: /tmp:::",
		":/tmp::": "bad format for volumes: :/tmp::",
		"path:ro": "ro is not an absolute path",
	}

	for _, path := range valid {
		if _, err := ValidateDevice(path); err != nil {
			t.Fatalf("ValidateDevice(`%q`) should succeed: error %q", path, err)
		}
	}

	for path, expectedError := range invalid {
		if _, err := ValidateDevice(path); err == nil {
			t.Fatalf("ValidateDevice(`%q`) should have failed validation", path)
		} else {
			if err.Error() != expectedError {
				t.Fatalf("ValidateDevice(`%q`) error should contain %q, got %q", path, expectedError, err.Error())
			}
		}
	}
}

func TestValidateEnv(t *testing.T) {
	invalids := map[string]string{
		"some spaces": "poorly formatted environment: variable 'some spaces' is not a valid environment variable",
		"asd!qwe":     "poorly formatted environment: variable 'asd!qwe' is not a valid environment variable",
		"1asd":        "poorly formatted environment: variable '1asd' is not a valid environment variable",
		"123":         "poorly formatted environment: variable '123' is not a valid environment variable",
	}
	valids := map[string]string{
		"a":                  "a",
		"something":          "something",
		"_=a":                "_=a",
		"env1=value1":        "env1=value1",
		"_env1=value1":       "_env1=value1",
		"env2=value2=value3": "env2=value2=value3",
		"env3=abc!qwe":       "env3=abc!qwe",
		"env_4=value 4":      "env_4=value 4",
		"PATH":               fmt.Sprintf("PATH=%v", os.Getenv("PATH")),
		"PATH=something":     "PATH=something",
	}
	for value, expectedError := range invalids {
		_, err := ValidateEnv(value)
		if err == nil {
			t.Fatalf("Expected ErrBadEnvVariable, got nothing")
		}
		if _, ok := err.(ErrBadEnvVariable); !ok {
			t.Fatalf("Expected ErrBadEnvVariable, got [%s]", err)
		}
		if err.Error() != expectedError {
			t.Fatalf("Expected ErrBadEnvVariable with message [%s], got [%s]", expectedError, err.Error())
		}
	}
	for value, expected := range valids {
		actual, err := ValidateEnv(value)
		if err != nil {
			t.Fatal(err)
		}
		if actual != expected {
			t.Fatalf("Expected [%v], got [%v]", expected, actual)
		}
	}
}

func TestValidateLabel(t *testing.T) {
	if _, err := ValidateLabel("label"); err == nil || err.Error() != "bad attribute format: label" {
		t.Fatalf("Expected an error [bad attribute format: label], go %v", err)
	}
	if actual, err := ValidateLabel("key1=value1"); err != nil || actual != "key1=value1" {
		t.Fatalf("Expected [key1=value1], got [%v,%v]", actual, err)
	}
	// Validate it's working with more than one =
	if actual, err := ValidateLabel("key1=value1=value2"); err != nil {
		t.Fatalf("Expected [key1=value1=value2], got [%v,%v]", actual, err)
	}
	// Validate it's working with one more
	if actual, err := ValidateLabel("key1=value1=value2=value3"); err != nil {
		t.Fatalf("Expected [key1=value1=value2=value2], got [%v,%v]", actual, err)
	}
}

func TestValidateHost(t *testing.T) {
	invalid := map[string]string{
		"anything":              "Invalid bind address format: anything",
		"something with spaces": "Invalid bind address format: something with spaces",
		"://":                "Invalid bind address format: ://",
		"unknown://":         "Invalid bind address format: unknown://",
		"tcp://":             "Invalid proto, expected tcp: ",
		"tcp://:port":        "Invalid bind address format: :port",
		"tcp://invalid":      "Invalid bind address format: invalid",
		"tcp://invalid:port": "Invalid bind address format: invalid:port",
	}
	valid := map[string]string{
		"fd://":                    "fd://",
		"fd://something":           "fd://something",
		"tcp://:2375":              "tcp://127.0.0.1:2375", // default ip address
		"tcp://:2376":              "tcp://127.0.0.1:2376", // default ip address
		"tcp://0.0.0.0:8080":       "tcp://0.0.0.0:8080",
		"tcp://192.168.0.0:12000":  "tcp://192.168.0.0:12000",
		"tcp://192.168:8080":       "tcp://192.168:8080",
		"tcp://0.0.0.0:1234567890": "tcp://0.0.0.0:1234567890", // yeah it's valid :P
		"tcp://docker.com:2375":    "tcp://docker.com:2375",
		"unix://":                  "unix:///var/run/docker.sock", // default unix:// value
		"unix://path/to/socket":    "unix://path/to/socket",
	}

	for value, errorMessage := range invalid {
		if _, err := ValidateHost(value); err == nil || err.Error() != errorMessage {
			t.Fatalf("Expected an error for %v with [%v], got [%v]", value, errorMessage, err)
		}
	}
	for value, expected := range valid {
		if actual, err := ValidateHost(value); err != nil || actual != expected {
			t.Fatalf("Expected for %v [%v], got [%v, %v]", value, expected, actual, err)
		}
	}
}

func logOptsValidator(val string) (string, error) {
	allowedKeys := map[string]string{"max-size": "1", "max-file": "2"}
	vals := strings.Split(val, "=")
	if allowedKeys[vals[0]] != "" {
		return val, nil
	}
	return "", fmt.Errorf("invalid key %s", vals[0])
}
