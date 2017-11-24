/*
Copyright 2017 The Kubernetes Authors.

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

package nvme

//
//import (
//	"fmt"
//	"io/ioutil"
//	"github.com/golang/glog"
//	"strings"
//	"os"
//	"path/filepath"
//	"flag"
//)
//
//func main() {
//	flag.Parse()
//	found, err := findNvme()
//	if err != nil {
//		glog.Fatalf("error: %v", err)
//	}
//	glog.Infof("returned = %q", found)
//}
//
//func findNvme() (string, error) {
//	// Look for nvme devices
//	devices, err := ioutil.ReadDir("/dev")
//	if err != nil {
//		return "", fmt.Errorf("error listing /dev directory: %v", err)
//	}
//	for _, device := range devices {
//		name := device.Name()
//		if !strings.HasPrefix(name, "nvme") {
//			continue
//		}
//		glog.V(6).Infof("found nvme %q", name)
//
//		// Skip partition devices
//		{
//			num := 0
//			ns := 0
//			partition := 0
//			tokens, err := fmt.Sscanf(name, "nvme%dn%dp%d", &num, &ns, &partition)
//			if err == nil && tokens == 3 {
//				glog.V(4).Infof("skipping nvme partition device %q", name)
//				continue
//			}
//		}
//
//		p := filepath.Join("/dev", name)
//
//		// Skip non-block devices
//		{
//			s, err := os.Stat(p)
//			if err != nil {
//				glog.Warningf("ignoring error doing stat on %q: %v", p, err)
//				continue
//			}
//			t := s.Mode() & os.ModeType
//			glog.V(2).Infof("%q had mode %v", p, t)
//			if (t&os.ModeDevice == 0) || (t&os.ModeCharDevice != 0) {
//				glog.V(2).Infof("ignoring nvme device %q with unexpected mode %v", p, s.Mode())
//				continue
//			}
//		}
//
//		glog.V(2).Infof("found nvme candidate %q", p)
//
//		info, found, err := Identify(p)
//		if err != nil {
//			glog.Warningf("ignoring error identifying %q: %v", p, err)
//			continue
//		}
//		if !found {
//			glog.Warningf("nvme identification not supported for %q", p)
//			continue
//		}
//
//		glog.Infof("got info for %q: %q", p, info)
//	}
//
//	return "", nil
//}
