// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"bytes"
	"fmt"
	"io/ioutil"

	"github.com/coreos/rkt/pkg/user"

	"github.com/appc/spec/schema/types"
)

const DefaultPath = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

var defaultEnv = map[string]string{
	"PATH":    DefaultPath,
	"SHELL":   "/bin/sh",
	"USER":    "root",
	"LOGNAME": "root",
	"HOME":    "/root",
}

// WriteEnvFile creates an environment file for given app name, the minimum
// required environment variables by the appc spec will be set to sensible
// defaults here if they're not provided by env.
func WriteEnvFile(env types.Environment, uidRange *user.UidRange, envFilePath string) error {
	ef := bytes.Buffer{}

	for dk, dv := range defaultEnv {
		if _, exists := env.Get(dk); !exists {
			fmt.Fprintf(&ef, "%s=%s\n", dk, dv)
		}
	}

	for _, e := range env {
		fmt.Fprintf(&ef, "%s=%s\n", e.Name, e.Value)
	}

	if err := ioutil.WriteFile(envFilePath, ef.Bytes(), 0644); err != nil {
		return err
	}

	if err := user.ShiftFiles([]string{envFilePath}, uidRange); err != nil {
		return err
	}

	return nil
}
