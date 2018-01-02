package scan

import (
	"fmt"
	"testing"
)

var TestingScanner = &Scanner{
	Description: "Tests common scan functions",
	scan: func(addr, hostname string) (Grade, Output, error) {
		switch addr {
		case "bad.example.com:443":
			return Bad, "bad.com", nil
		case "Warning.example.com:443":
			return Warning, "Warning.com", nil
		case "good.example.com:443":
			return Good, "good.com", nil
		case "skipped.example.com:443/0":
			return Skipped, "skipped", nil
		default:
			return Grade(-1), "invalid", fmt.Errorf("scan: invalid grade")
		}
	},
}

var TestingFamily = &Family{
	Description: "Tests the scan_common",
	Scanners: map[string]*Scanner{
		"TestingScanner": TestingScanner,
	},
}

func TestCommon(t *testing.T) {
	if TestingFamily.Scanners["TestingScanner"] != TestingScanner {
		t.FailNow()
	}

	var grade Grade
	var output Output
	var err error

	grade, output, err = TestingScanner.Scan("bad.example.com:443", "bad.example.com")
	if grade != Bad || output.(string) != "bad.com" || err != nil {
		t.FailNow()
	}

	grade, output, err = TestingScanner.Scan("Warning.example.com:443", "Warning.example.com")
	if grade != Warning || output.(string) != "Warning.com" || err != nil {
		t.FailNow()
	}

	grade, output, err = TestingScanner.Scan("good.example.com:443", "good.example.com")
	if grade != Good || output.(string) != "good.com" || err != nil {
		t.FailNow()
	}

	grade, output, err = TestingScanner.Scan("skipped.example.com:443/0", "")
	if grade != Skipped || output.(string) != "skipped" || err != nil {
		t.FailNow()
	}

	_, _, err = TestingScanner.Scan("invalid", "invalid")
	if err == nil {
		t.FailNow()
	}
}
