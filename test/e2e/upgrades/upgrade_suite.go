/*
Copyright 2021 The Kubernetes Authors.

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
	"context"
	"encoding/xml"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/junit"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

type chaosMonkeyAdapter struct {
	test        Test
	framework   *framework.Framework
	upgradeType UpgradeType
	upgCtx      UpgradeContext
}

func (cma *chaosMonkeyAdapter) Test(ctx context.Context, sem *chaosmonkey.Semaphore) {
	var once sync.Once
	ready := func() {
		once.Do(func() {
			sem.Ready()
		})
	}
	defer ready()
	if skippable, ok := cma.test.(Skippable); ok && skippable.Skip(cma.upgCtx) {
		ginkgo.By("skipping test " + cma.test.Name())
		return
	}

	ginkgo.DeferCleanup(cma.test.Teardown, cma.framework)
	cma.test.Setup(ctx, cma.framework)
	ready()
	cma.test.Test(ctx, cma.framework, sem.StopCh, cma.upgradeType)
}

func CreateUpgradeFrameworks(tests []Test) map[string]*framework.Framework {
	nsFilter := regexp.MustCompile("[^[:word:]-]+") // match anything that's not a word character or hyphen
	testFrameworks := map[string]*framework.Framework{}
	for _, t := range tests {
		ns := nsFilter.ReplaceAllString(t.Name(), "-") // and replace with a single hyphen
		ns = strings.Trim(ns, "-")
		f := framework.NewDefaultFramework(ns)
		f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
		testFrameworks[t.Name()] = f
	}
	return testFrameworks
}

// RunUpgradeSuite runs the actual upgrade tests.
func RunUpgradeSuite(
	ctx context.Context,
	upgCtx *UpgradeContext,
	tests []Test,
	testFrameworks map[string]*framework.Framework,
	testSuite *junit.TestSuite,
	upgradeType UpgradeType,
	upgradeFunc func(ctx context.Context),
) {
	cm := chaosmonkey.New(upgradeFunc)
	for _, t := range tests {
		testCase := &junit.TestCase{
			Name:      t.Name(),
			Classname: "upgrade_tests",
		}
		testSuite.TestCases = append(testSuite.TestCases, testCase)
		cma := chaosMonkeyAdapter{
			test:        t,
			framework:   testFrameworks[t.Name()],
			upgradeType: upgradeType,
			upgCtx:      *upgCtx,
		}
		cm.Register(cma.Test)
	}

	start := time.Now()
	defer func() {
		testSuite.Update()
		testSuite.Time = time.Since(start).Seconds()
		if framework.TestContext.ReportDir != "" {
			fname := filepath.Join(framework.TestContext.ReportDir, fmt.Sprintf("junit_%supgrades.xml", framework.TestContext.ReportPrefix))
			f, err := os.Create(fname)
			if err != nil {
				return
			}
			defer f.Close()
			xml.NewEncoder(f).Encode(testSuite)
		}
	}()
	cm.Do(ctx)
}
