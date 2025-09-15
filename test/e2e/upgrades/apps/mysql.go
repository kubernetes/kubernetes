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

package apps

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2estatefulset "k8s.io/kubernetes/test/e2e/framework/statefulset"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/upgrades"
)

const mysqlManifestPath = "test/e2e/testing-manifests/statefulset/mysql-upgrade"

// MySQLUpgradeTest implements an upgrade test harness that polls a replicated sql database.
type MySQLUpgradeTest struct {
	ip               string
	successfulWrites int
	nextWrite        int
}

// Name returns the tracking name of the test.
func (MySQLUpgradeTest) Name() string { return "mysql-upgrade" }

// Skip returns true when this test can be skipped.
func (MySQLUpgradeTest) Skip(upgCtx upgrades.UpgradeContext) bool {
	minVersion := version.MustParseSemantic("1.5.0")

	for _, vCtx := range upgCtx.Versions {
		if vCtx.Version.LessThan(minVersion) {
			return true
		}
	}
	return false
}

func mysqlKubectlCreate(ns, file string) {
	data, err := e2etestfiles.Read(filepath.Join(mysqlManifestPath, file))
	if err != nil {
		framework.Fail(err.Error())
	}
	input := string(data)
	e2ekubectl.RunKubectlOrDieInput(ns, input, "create", "-f", "-")
}

func (t *MySQLUpgradeTest) getServiceIP(ctx context.Context, f *framework.Framework, ns, svcName string) string {
	svc, err := f.ClientSet.CoreV1().Services(ns).Get(ctx, svcName, metav1.GetOptions{})
	framework.ExpectNoError(err)
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
func (t *MySQLUpgradeTest) Setup(ctx context.Context, f *framework.Framework) {
	ns := f.Namespace.Name
	statefulsetPoll := 30 * time.Second
	statefulsetTimeout := 10 * time.Minute

	ginkgo.By("Creating a configmap")
	mysqlKubectlCreate(ns, "configmap.yaml")

	ginkgo.By("Creating a mysql StatefulSet")
	e2estatefulset.CreateStatefulSet(ctx, f.ClientSet, mysqlManifestPath, ns)

	ginkgo.By("Creating a mysql-test-server deployment")
	mysqlKubectlCreate(ns, "tester.yaml")

	ginkgo.By("Getting the ingress IPs from the test-service")
	err := wait.PollUntilContextTimeout(ctx, statefulsetPoll, statefulsetTimeout, true, func(ctx context.Context) (bool, error) {
		if t.ip = t.getServiceIP(ctx, f, ns, "test-server"); t.ip == "" {
			return false, nil
		}
		if _, err := t.countNames(); err != nil {
			framework.Logf("Service endpoint is up but isn't responding")
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err)
	framework.Logf("Service endpoint is up")

	ginkgo.By("Adding 2 names to the database")
	err = t.addName(strconv.Itoa(t.nextWrite))
	framework.ExpectNoError(err)
	err = t.addName(strconv.Itoa(t.nextWrite))
	framework.ExpectNoError(err)

	ginkgo.By("Verifying that the 2 names have been inserted")
	count, err := t.countNames()
	framework.ExpectNoError(err)
	gomega.Expect(count).To(gomega.Equal(2))
}

// Test continually polls the db using the read and write connections, inserting data, and checking
// that all the data is readable.
func (t *MySQLUpgradeTest) Test(ctx context.Context, f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	var writeSuccess, readSuccess, writeFailure, readFailure int
	ginkgo.By("Continuously polling the database during upgrade.")
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
func (t *MySQLUpgradeTest) Teardown(ctx context.Context, f *framework.Framework) {
	count, err := t.countNames()
	framework.ExpectNoError(err)
	gomega.Expect(count).To(gomega.BeNumerically(">=", t.successfulWrites), "count is too small")
}

// addName adds a new value to the db.
func (t *MySQLUpgradeTest) addName(name string) error {
	val := map[string][]string{"name": {name}}
	t.nextWrite++
	r, err := http.PostForm(fmt.Sprintf("http://%s/addName", net.JoinHostPort(t.ip, "8080")), val)
	if err != nil {
		return err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			return err
		}
		return errors.New(string(b))
	}
	return nil
}

// countNames checks to make sure the values in testing.users are available, and returns
// the count of them.
func (t *MySQLUpgradeTest) countNames() (int, error) {
	r, err := http.Get(fmt.Sprintf("http://%s/countNames", net.JoinHostPort(t.ip, "8080")))
	if err != nil {
		return 0, err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			return 0, err
		}
		return 0, errors.New(string(b))
	}
	var count int
	if err := json.NewDecoder(r.Body).Decode(&count); err != nil {
		return 0, err
	}
	return count, nil
}
