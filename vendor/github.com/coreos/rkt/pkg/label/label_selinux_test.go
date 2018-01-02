// Copyright 2014,2015 Red Hat, Inc
// Copyright 2014,2015 Docker, Inc
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
//
// +build selinux,linux

package label

import (
	"io/ioutil"
	"strings"
	"testing"

	"github.com/coreos/rkt/pkg/selinux"
)

func TestInit(t *testing.T) {
	if selinux.SelinuxEnabled() {
		var testNull []string
		plabel, mlabel, err := InitLabels("", testNull)
		if err != nil {
			t.Log("InitLabels Failed")
			t.Fatal(err)
		}
		testDisabled := []string{"disable"}
		plabel, mlabel, err = InitLabels("", testDisabled)
		if err != nil {
			t.Log("InitLabels Disabled Failed")
			t.Fatal(err)
		}
		if plabel != "" {
			t.Log("InitLabels Disabled Failed")
			t.Fatal(plabel)
		}
		testUser := []string{"user:user_u", "role:user_r", "type:user_t", "level:s0:c1,c15"}
		plabel, mlabel, err = InitLabels("", testUser)
		if err != nil {
			t.Log("InitLabels User Failed")
			t.Fatal(err)
		}
		if plabel != "user_u:user_r:user_t:s0:c1,c15" || mlabel != "user_u:object_r:svirt_sandbox_file_t:s0:c1,c15" {
			t.Log("InitLabels User Match Failed - unable to test policy")
			t.Log(plabel, mlabel)
			return
		}

		testBadData := []string{"user", "role:user_r", "type:user_t", "level:s0:c1,c15"}
		plabel, mlabel, err = InitLabels("", testBadData)
		if err == nil {
			t.Log("InitLabels Bad Failed")
			t.Fatal(err)
		}
	}
}
func TestDuplicateLabel(t *testing.T) {
	secopt := DupSecOpt("system_u:system_r:svirt_lxc_net_t:s0:c1,c2")
	t.Log(secopt)
	for _, opt := range secopt {
		con := strings.SplitN(opt, ":", 3)
		if len(con) != 3 || con[0] != "label" {
			t.Errorf("Invalid DupSecOpt return value")
			continue
		}
		if con[1] == "user" {
			if con[2] != "system_u" {
				t.Errorf("DupSecOpt Failed user incorrect")
			}
			continue
		}
		if con[1] == "role" {
			if con[2] != "system_r" {
				t.Errorf("DupSecOpt Failed role incorrect")
			}
			continue
		}
		if con[1] == "type" {
			if con[2] != "svirt_lxc_net_t" {
				t.Errorf("DupSecOpt Failed type incorrect")
			}
			continue
		}
		if con[1] == "level" {
			if con[2] != "s0:c1,c2" {
				t.Errorf("DupSecOpt Failed level incorrect")
			}
			continue
		}
		t.Errorf("DupSecOpt Failed invalid field %q", con[1])
	}
	secopt = DisableSecOpt()
	if secopt[0] != "label:disable" {
		t.Errorf("DisableSecOpt Failed level incorrect")
	}
}
func TestRelabel(t *testing.T) {
	plabel, _, err := InitLabels("", nil)
	if err != nil {
		t.Fatalf("InitLabels failed: %v", err)
	}
	if plabel == "" {
		t.Log("No svirt container policy, skipping")
		return
	}
	testdir, err := ioutil.TempDir("/tmp", "rkt")
	if err != nil {
		t.Fatalf("Unable to create test dir: %v", err)
	}
	label := "system_u:system_r:svirt_sandbox_file_t:s0:c1,c2"
	if err := Relabel(testdir, "", "z"); err != nil {
		t.Fatalf("Relabel with no label failed: %v", err)
	}
	if err := Relabel(testdir, label, ""); err != nil {
		t.Fatalf("Relabel with no relabel field failed: %v", err)
	}
	if err := Relabel(testdir, label, "z"); err != nil {
		if ae, ok := err.(*selinux.SelinuxError); ok {
			if ae.Errno == selinux.InvalidContext {
				println("Missing selinux contexts, skipping")
				return
			}
			t.Fatalf("Relabel shared failed: %v", err)
		}
	}
	if err := Relabel(testdir, label, "Z"); err != nil {
		t.Fatalf("Relabel unshared failed: %v", err)
	}
	if err := Relabel(testdir, label, "zZ"); err == nil {
		t.Fatal("Relabel with shared and unshared succeeded")
	}
	if err := Relabel("/etc", label, "zZ"); err == nil {
		t.Fatal("Relabel /etc succeeded")
	}
	if err := Relabel("/", label, ""); err == nil {
		t.Fatal("Relabel / succeeded")
	}
	if err := Relabel("/usr", label, "Z"); err == nil {
		t.Fatal("Relabel /usr succeeded")
	}
}
