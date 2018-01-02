// +build windows

package dockerfile

import (
	"fmt"
	"testing"

	"github.com/docker/docker/pkg/testutil"
	"github.com/stretchr/testify/assert"
)

func TestNormaliseDest(t *testing.T) {
	tests := []struct{ current, requested, expected, etext string }{
		{``, `D:\`, ``, `Windows does not support destinations not on the system drive (C:)`},
		{``, `e:/`, ``, `Windows does not support destinations not on the system drive (C:)`},
		{`invalid`, `./c1`, ``, `Current WorkingDir invalid is not platform consistent`},
		{`C:`, ``, ``, `Current WorkingDir C: is not platform consistent`},
		{`C`, ``, ``, `Current WorkingDir C is not platform consistent`},
		{`D:\`, `.`, ``, "Windows does not support relative paths when WORKDIR is not the system drive"},
		{``, `D`, `D`, ``},
		{``, `./a1`, `.\a1`, ``},
		{``, `.\b1`, `.\b1`, ``},
		{``, `/`, `\`, ``},
		{``, `\`, `\`, ``},
		{``, `c:/`, `\`, ``},
		{``, `c:\`, `\`, ``},
		{``, `.`, `.`, ``},
		{`C:\wdd`, `./a1`, `\wdd\a1`, ``},
		{`C:\wde`, `.\b1`, `\wde\b1`, ``},
		{`C:\wdf`, `/`, `\`, ``},
		{`C:\wdg`, `\`, `\`, ``},
		{`C:\wdh`, `c:/`, `\`, ``},
		{`C:\wdi`, `c:\`, `\`, ``},
		{`C:\wdj`, `.`, `\wdj`, ``},
		{`C:\wdk`, `foo/bar`, `\wdk\foo\bar`, ``},
		{`C:\wdl`, `foo\bar`, `\wdl\foo\bar`, ``},
		{`C:\wdm`, `foo/bar/`, `\wdm\foo\bar\`, ``},
		{`C:\wdn`, `foo\bar/`, `\wdn\foo\bar\`, ``},
	}
	for _, testcase := range tests {
		msg := fmt.Sprintf("Input: %s, %s", testcase.current, testcase.requested)
		actual, err := normaliseDest(testcase.current, testcase.requested)
		if testcase.etext == "" {
			if !assert.NoError(t, err, msg) {
				continue
			}
			assert.Equal(t, testcase.expected, actual, msg)
		} else {
			testutil.ErrorContains(t, err, testcase.etext)
		}
	}
}
