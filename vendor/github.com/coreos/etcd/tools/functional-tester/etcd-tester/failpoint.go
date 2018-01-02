// Copyright 2016 The etcd Authors
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

package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"
	"time"
)

type failpointStats struct {
	// crashes counts the number of crashes for a failpoint
	crashes map[string]int
	// mu protects crashes
	mu sync.Mutex
}

var fpStats failpointStats

func failpointFailures(c *cluster) (ret []failure, err error) {
	var fps []string
	fps, err = failpointPaths(c.Members[0].FailpointURL)
	if err != nil {
		return nil, err
	}
	// create failure objects for all failpoints
	for _, fp := range fps {
		if len(fp) == 0 {
			continue
		}
		fpFails := failuresFromFailpoint(fp)
		// wrap in delays so failpoint has time to trigger
		for i, fpf := range fpFails {
			if strings.Contains(fp, "Snap") {
				// hack to trigger snapshot failpoints
				fpFails[i] = &failureUntilSnapshot{fpf}
			} else {
				fpFails[i] = &failureDelay{fpf, 3 * time.Second}
			}
		}
		ret = append(ret, fpFails...)
	}
	fpStats.crashes = make(map[string]int)
	return ret, err
}

func failpointPaths(endpoint string) ([]string, error) {
	resp, err := http.Get(endpoint)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, rerr := ioutil.ReadAll(resp.Body)
	if rerr != nil {
		return nil, rerr
	}
	var fps []string
	for _, l := range strings.Split(string(body), "\n") {
		fp := strings.Split(l, "=")[0]
		fps = append(fps, fp)
	}
	return fps, nil
}

func failuresFromFailpoint(fp string) []failure {
	inject := makeInjectFailpoint(fp, `panic("etcd-tester")`)
	recov := makeRecoverFailpoint(fp)
	return []failure{
		&failureOne{
			description:   description("failpoint " + fp + " panic one"),
			injectMember:  inject,
			recoverMember: recov,
		},
		&failureAll{
			description:   description("failpoint " + fp + " panic all"),
			injectMember:  inject,
			recoverMember: recov,
		},
		&failureMajority{
			description:   description("failpoint " + fp + " panic majority"),
			injectMember:  inject,
			recoverMember: recov,
		},
		&failureLeader{
			failureByFunc{
				description:   description("failpoint " + fp + " panic leader"),
				injectMember:  inject,
				recoverMember: recov,
			},
			0,
		},
	}
}

func makeInjectFailpoint(fp, val string) injectMemberFunc {
	return func(m *member) (err error) {
		return putFailpoint(m.FailpointURL, fp, val)
	}
}

func makeRecoverFailpoint(fp string) recoverMemberFunc {
	return func(m *member) error {
		if err := delFailpoint(m.FailpointURL, fp); err == nil {
			return nil
		}
		// node not responding, likely dead from fp panic; restart
		fpStats.mu.Lock()
		fpStats.crashes[fp]++
		fpStats.mu.Unlock()
		return recoverStop(m)
	}
}

func putFailpoint(ep, fp, val string) error {
	req, _ := http.NewRequest(http.MethodPut, ep+"/"+fp, strings.NewReader(val))
	c := http.Client{}
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("failed to PUT %s=%s at %s (%v)", fp, val, ep, resp.Status)
	}
	return nil
}

func delFailpoint(ep, fp string) error {
	req, _ := http.NewRequest(http.MethodDelete, ep+"/"+fp, strings.NewReader(""))
	c := http.Client{}
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("failed to DELETE %s at %s (%v)", fp, ep, resp.Status)
	}
	return nil
}
