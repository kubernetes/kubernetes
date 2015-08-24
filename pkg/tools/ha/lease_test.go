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

package ha

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/client/testclient"
	"testing"
	"time"
)

func TestLeaseLifecycle(t *testing.T) {
	client := &testclient.Fake{}
	c := make(chan string)

	startCM := func(leaseUserInfo *LeaseUser) bool {
		leaseUserInfo.Running = true
		fmt.Printf("START!....... ")
		c <- "started!"
		return true
	}

	endCM := func(leaseUserInfo *LeaseUser) bool {
		leaseUserInfo.Running = false
		fmt.Printf("END..... DONE!")
		c <- "finished!"
		return true
	}

	ttl := uint64(2)
	timebomb := uint64(30)
	sleep := uint64(1)

	//This starts a thread that continues running.
	RunLeasedProcess(client, "", startCM, endCM, &Config{
		Key: "test",
		//I'll avoid a long comment and just say that this
		//should result in a test that runs in under 20 seconds.
		//See the lease impl for details.
		Ttl:      ttl,
		Timebomb: timebomb,
		Sleep:    sleep,
	})
	start := time.Now()

	//Gaurantee that callbacks are called.
	assert.Contains(t, <-c, "started")
	assert.Contains(t, <-c, "finished")

	elapsed := time.Since(start)
	min := time.Duration(timebomb) * time.Second
	max := time.Duration(ttl+timebomb+sleep) * time.Second

	//Gaurantee that timebomb and ttl are honored
	if elapsed < min || elapsed > max {
		t.Errorf("%d Needed to occur in the (%d,%d) seconds range !", elapsed, min, max)
	} else {
		fmt.Printf(fmt.Sprintf("Lease lifecycle concluded in %v (%v -> %v)", elapsed, start, time.Now()))
	}
}
