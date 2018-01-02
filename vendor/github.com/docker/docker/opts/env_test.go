package opts

import (
	"fmt"
	"os"
	"runtime"
	"testing"
)

func TestValidateEnv(t *testing.T) {
	valids := map[string]string{
		"a":                   "a",
		"something":           "something",
		"_=a":                 "_=a",
		"env1=value1":         "env1=value1",
		"_env1=value1":        "_env1=value1",
		"env2=value2=value3":  "env2=value2=value3",
		"env3=abc!qwe":        "env3=abc!qwe",
		"env_4=value 4":       "env_4=value 4",
		"PATH":                fmt.Sprintf("PATH=%v", os.Getenv("PATH")),
		"PATH=something":      "PATH=something",
		"asd!qwe":             "asd!qwe",
		"1asd":                "1asd",
		"123":                 "123",
		"some space":          "some space",
		"  some space before": "  some space before",
		"some space after  ":  "some space after  ",
	}
	// Environment variables are case in-sensitive on Windows
	if runtime.GOOS == "windows" {
		valids["PaTh"] = fmt.Sprintf("PaTh=%v", os.Getenv("PATH"))
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
