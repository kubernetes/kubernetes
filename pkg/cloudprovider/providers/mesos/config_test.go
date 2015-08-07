/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package mesos

import (
	"bytes"
	"testing"
	"time"

	log "github.com/golang/glog"
)

// test mesos.createDefaultConfig
func Test_createDefaultConfig(t *testing.T) {
	defer log.Flush()

	config := createDefaultConfig()

	if config.MesosMaster != DefaultMesosMaster {
		t.Fatalf("Default config has the expected MesosMaster value")
	}

	if config.MesosHttpClientTimeout.Duration != DefaultHttpClientTimeout {
		t.Fatalf("Default config has the expected MesosHttpClientTimeout value")
	}

	if config.StateCacheTTL.Duration != DefaultStateCacheTTL {
		t.Fatalf("Default config has the expected StateCacheTTL value")
	}
}

// test mesos.readConfig
func Test_readConfig(t *testing.T) {
	defer log.Flush()

	configString := `
[mesos-cloud]
	mesos-master        = leader.mesos:5050
	http-client-timeout = 500ms
	state-cache-ttl     = 1h`

	reader := bytes.NewBufferString(configString)

	config, err := readConfig(reader)

	if err != nil {
		t.Fatalf("Reading configuration does not yield an error: %#v", err)
	}

	if config.MesosMaster != "leader.mesos:5050" {
		t.Fatalf("Parsed config has the expected MesosMaster value")
	}

	if config.MesosHttpClientTimeout.Duration != time.Duration(500)*time.Millisecond {
		t.Fatalf("Parsed config has the expected MesosHttpClientTimeout value")
	}

	if config.StateCacheTTL.Duration != time.Duration(1)*time.Hour {
		t.Fatalf("Parsed config has the expected StateCacheTTL value")
	}
}
