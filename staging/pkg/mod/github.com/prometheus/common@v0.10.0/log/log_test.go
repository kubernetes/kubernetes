// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package log

import (
	"bytes"
	"regexp"
	"testing"

	"github.com/sirupsen/logrus"
)

func TestFileLineLogging(t *testing.T) {
	var buf bytes.Buffer
	origLogger.Out = &buf
	origLogger.Formatter = &logrus.TextFormatter{
		DisableColors: true,
	}

	// The default logging level should be "info".
	Debug("This debug-level line should not show up in the output.")
	Infof("This %s-level line should show up in the output.", "info")

	re := `^time=".*" level=info msg="This info-level line should show up in the output." source="log_test.go:33"\n$`
	if !regexp.MustCompile(re).Match(buf.Bytes()) {
		t.Fatalf("%q did not match expected regex %q", buf.String(), re)
	}
}
