/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package viperconfig

import (
	"flag"
	"io/ioutil"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/test/e2e/framework/config"
)

func TestViperConfig(t *testing.T) {
	flags := flag.NewFlagSet("test", 0)
	type Context struct {
		Bool     bool          `default:"true"`
		Duration time.Duration `default:"1ms"`
		Float64  float64       `default:"1.23456789"`
		String   string        `default:"hello world"`
		Int      int           `default:"-1" usage:"some number"`
		Int64    int64         `default:"-1234567890123456789"`
		Uint     uint          `default:"1"`
		Uint64   uint64        `default:"1234567890123456789"`
	}
	var context Context
	require.NotPanics(t, func() {
		config.AddOptionsToSet(flags, &context, "")
	})

	viperConfig := `
bool: false
duration: 1s
float64: -1.23456789
string: pong
int: -2
int64: -9123456789012345678
uint: 2
uint64: 9123456789012345678
`
	tmpfile, err := ioutil.TempFile("", "viperconfig-*.yaml")
	require.NoError(t, err, "temp file")
	defer os.Remove(tmpfile.Name())
	if _, err := tmpfile.Write([]byte(viperConfig)); err != nil {
		require.NoError(t, err, "write config")
	}
	require.NoError(t, tmpfile.Close(), "close temp file")

	require.NoError(t, ViperizeFlags(tmpfile.Name(), "", flags), "read config file")
	require.Equal(t,
		Context{false, time.Second, -1.23456789, "pong",
			-2, -9123456789012345678, 2, 9123456789012345678,
		},
		context,
		"values from viper must match")
}
