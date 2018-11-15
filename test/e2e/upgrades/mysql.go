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

package upgrades

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"path/filepath"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
)

const mysqlManifestPath = "test/e2e/testing-manifests/statefulset/mysql-upgrade"

// MySqlUpgradeTest implements an upgrade test harness that polls a replicated sql database.
type MySqlUpgradeTest struct {
	ip               string
	successfulWrites int
	nextWrite        int
	ssTester         *framework.StatefulSetTester
}

func (MySqlUpgradeTest) Name() string { return "mysql-upgrade" }

func (MySqlUpgradeTest) Skip(upgCtx UpgradeContext) bool {
	minVersion := version.MustParseSemantic("1.5.0")

	for _, vCtx := range upgCtx.Versions {
		if vCtx.Version.LessThan(minVersion) {
			return true
		}
	}
	return false
}

func mysqlKubectlCreate(ns, file string) {
	input := string(testfiles.ReadOrDie(filepath.Join(mysqlManifestPath, file), Fail))
	framework.RunKubectlOrDieInput(input, "create", "-f", "-", fmt.Sprintf("--namespace=%s", ns))
}

func (t *MySqlUpgradeTest) getServiceIP(f *framework.Framework, ns, svcName string) string {
	svc, err := f.ClientSet.CoreV1().Services(ns).Get(svcName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	ingress := svc.Status.LoadBalancer.Ingress
	if len(ingress) == 0 {
		return ""
	}
	return ingress[0].IP
}

// Setup creates a StatefulSet, HeadlessService, a Service to write to the db, and a Service to read
// from the db. It then connects to the db with the write Service and populates the db with a table
// and a few entries. Finally, it connects to the db with the read Service, and confirms the data is
// available. The db connections are left open to be used later in the test.
func (t *MySqlUpgradeTest) Setup(f *framework.Framework) {
	ns := f.Namespace.Name
	statefulsetPoll := 30 * time.Second
	statefulsetTimeout := 10 * time.Minute
	t.ssTester = framework.NewStatefulSetTester(f.ClientSet)

	By("Creating a configmap")
	mysqlKubectlCreate(ns, "configmap.yaml")

	By("Creating a mysql StatefulSet")
	t.ssTester.CreateStatefulSet(mysqlManifestPath, ns)

	By("Creating a mysql-test-server deployment")
	mysqlKubectlCreate(ns, "tester.yaml")

	By("Getting the ingress IPs from the test-service")
	err := wait.PollImmediate(statefulsetPoll, statefulsetTimeout, func() (bool, error) {
		if t.ip = t.getServiceIP(f, ns, "test-server"); t.ip == "" {
			return false, nil
		}
		if _, err := t.countNames(); err != nil {
			framework.Logf("Service endpoint is up but isn't responding")
			return false, nil
		}
		return true, nil
	})
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Service endpoint is up")

	By("Adding 2 names to the database")
	Expect(t.addName(strconv.Itoa(t.nextWrite))).NotTo(HaveOccurred())
	Expect(t.addName(strconv.Itoa(t.nextWrite))).NotTo(HaveOccurred())

	By("Verifying that the 2 names have been inserted")
	count, err := t.countNames()
	Expect(err).NotTo(HaveOccurred())
	Expect(count).To(Equal(2))
}

// Test continually polls the db using the read and write connections, inserting data, and checking
// that all the data is readable.
func (t *MySqlUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	var writeSuccess, readSuccess, writeFailure, readFailure int
	By("Continuously polling the database during upgrade.")
	go wait.Until(func() {
		_, err := t.countNames()
		if err != nil {
			framework.Logf("Error while trying to read data: %v", err)
			readFailure++
		} else {
			readSuccess++
		}
	}, framework.Poll, done)

	wait.Until(func() {
		err := t.addName(strconv.Itoa(t.nextWrite))
		if err != nil {
			framework.Logf("Error while trying to write data: %v", err)
			writeFailure++
		} else {
			writeSuccess++
		}
	}, framework.Poll, done)

	t.successfulWrites = writeSuccess
	framework.Logf("Successful reads: %d", readSuccess)
	framework.Logf("Successful writes: %d", writeSuccess)
	framework.Logf("Failed reads: %d", readFailure)
	framework.Logf("Failed writes: %d", writeFailure)

	// TODO: Not sure what the ratio defining a successful test run should be. At time of writing the
	// test, failures only seem to happen when a race condition occurs (read/write starts, doesn't
	// finish before upgrade interferes).

	readRatio := float64(readSuccess) / float64(readSuccess+readFailure)
	writeRatio := float64(writeSuccess) / float64(writeSuccess+writeFailure)
	if readRatio < 0.75 {
		framework.Failf("Too many failures reading data. Success ratio: %f", readRatio)
	}
	if writeRatio < 0.75 {
		framework.Failf("Too many failures writing data. Success ratio: %f", writeRatio)
	}
}

// Teardown performs one final check of the data's availability.
func (t *MySqlUpgradeTest) Teardown(f *framework.Framework) {
	count, err := t.countNames()
	Expect(err).NotTo(HaveOccurred())
	Expect(count >= t.successfulWrites).To(BeTrue())
}

// addName adds a new value to the db.
func (t *MySqlUpgradeTest) addName(name string) error {
	val := map[string][]string{"name": {name}}
	t.nextWrite++
	r, err := http.PostForm(fmt.Sprintf("http://%s:8080/addName", t.ip), val)
	if err != nil {
		return err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			return err
		}
		return fmt.Errorf(string(b))
	}
	return nil
}

// countNames checks to make sure the values in testing.users are available, and returns
// the count of them.
func (t *MySqlUpgradeTest) countNames() (int, error) {
	r, err := http.Get(fmt.Sprintf("http://%s:8080/countNames", t.ip))
	if err != nil {
		return 0, err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			return 0, err
		}
		return 0, fmt.Errorf(string(b))
	}
	var count int
	if err := json.NewDecoder(r.Body).Decode(&count); err != nil {
		return 0, err
	}
	return count, nil
}
