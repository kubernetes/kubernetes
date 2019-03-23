/*
Copyright 2015 The Kubernetes Authors.

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

package glusterfs

import (
	"bufio"
	"fmt"
	"os"

	"k8s.io/klog"
)

// readGlusterLog will take the last 2 lines of the log file
// on failure of gluster SetUp and return those so kubelet can
// properly expose them
// return error on any failure
func readGlusterLog(path string, podName string) error {

	var line1 string
	var line2 string
	linecount := 0

	klog.Infof("failure, now attempting to read the gluster log for pod %s", podName)

	// Check and make sure path exists
	if len(path) == 0 {
		return fmt.Errorf("log file does not exist for pod %s", podName)
	}

	// open the log file
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("could not open log file for pod %s", podName)
	}
	defer file.Close()

	// read in and scan the file using scanner
	// from stdlib
	fscan := bufio.NewScanner(file)

	// rather than guessing on bytes or using Seek
	// going to scan entire file and take the last two lines
	// generally the file should be small since it is pod specific
	for fscan.Scan() {
		if linecount > 0 {
			line1 = line2
		}
		line2 = "\n" + fscan.Text()

		linecount++
	}

	if linecount > 0 {
		return fmt.Errorf("%v", line1+line2+"\n")
	}
	return nil
}
