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

package e2e

import (
	"flag"
	"fmt"
	"os"
	"path"
	"sort"
	"testing"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/framework/viperconfig"
	"k8s.io/kubernetes/test/e2e/generated"

	// test sources
	_ "k8s.io/kubernetes/test/e2e/apimachinery"
	_ "k8s.io/kubernetes/test/e2e/apps"
	_ "k8s.io/kubernetes/test/e2e/auth"
	_ "k8s.io/kubernetes/test/e2e/autoscaling"
	_ "k8s.io/kubernetes/test/e2e/common"
	_ "k8s.io/kubernetes/test/e2e/instrumentation"
	_ "k8s.io/kubernetes/test/e2e/kubectl"
	_ "k8s.io/kubernetes/test/e2e/lifecycle"
	_ "k8s.io/kubernetes/test/e2e/lifecycle/bootstrap"
	_ "k8s.io/kubernetes/test/e2e/network"
	_ "k8s.io/kubernetes/test/e2e/node"
	_ "k8s.io/kubernetes/test/e2e/scalability"
	_ "k8s.io/kubernetes/test/e2e/scheduling"
	_ "k8s.io/kubernetes/test/e2e/servicecatalog"
	_ "k8s.io/kubernetes/test/e2e/storage"
	_ "k8s.io/kubernetes/test/e2e/ui"
)

var (
	viperConfig     = flag.String("viper-config", "", "The name of a viper config file (https://github.com/spf13/viper#what-is-viper). All e2e command line parameters can also be configured in such a file. May contain a path and may or may not contain the file suffix. The default is to look for an optional file with `e2e` as base name. If a file is specified explicitly, it must be present.")
	urlCacheDir     = flag.String("testfiles.url.cache-dir", "test/e2e/testing-manifests/url-cache", "The directory in which test files referenced by URL are stored. Can be relative to the repo root or absolute.")
	urlDownload     = flag.Bool("testfiles.url.download", false, "Enables downloading files on demand. Disabled by default.")
	urlCacheRefresh = flag.Bool("testfiles.url.cache-refresh", false, "Downloads all URLs registered during test suite creation and puts them into the cache directory instead of running any tests.")
)

func TestMain(m *testing.M) {
	// Register framework flags, then handle flags and Viper config.
	framework.HandleFlags()
	if err := viperconfig.ViperizeFlags(*viperConfig, "e2e"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	framework.AfterReadingAllFlags(&framework.TestContext)

	// TODO: Deprecating repo-root over time... instead just use gobindata_util.go , see #23987.
	// Right now it is still needed, for example by
	// test/e2e/framework/ingress/ingress_utils.go
	// for providing the optional secret.yaml file and by
	// test/e2e/framework/util.go for cluster/log-dump.
	if framework.TestContext.RepoRoot != "" {
		testfiles.AddFileSource(testfiles.RootFileSource{Root: framework.TestContext.RepoRoot})
	}

	// Enable bindata file lookup as fallback.
	testfiles.AddFileSource(testfiles.BindataFileSource{
		Asset:      generated.Asset,
		AssetNames: generated.AssetNames,
	})

	// Deterministic output because we sort first.
	registered := testfiles.GetRegisteredFiles()
	sort.Strings(registered)

	// URLs as file path are also supported, but by default (see flags above)
	// must be cached as part of the test suite's source code.
	finalURLCacheDir := *urlCacheDir
	if finalURLCacheDir == "" {
		finalURLCacheDir = path.Join(framework.TestContext.RepoRoot, "test/e2e/testing-manifests/url-cache")
	}
	if *urlCacheRefresh {
		source := testfiles.URLFileSource{
			CacheDir: finalURLCacheDir,
		}
		var result int
		fmt.Printf("Refreshing URL cache directory %q instead of running tests.\n", finalURLCacheDir)
		for _, path := range registered {
			cachedFilePath, err := source.CachedFilePath(path)
			if err != nil {
				// Plain files do not need to be cached, so NotURL isn't an error
				// we care about.
				if errors.Cause(err) != testfiles.ErrNotURL {
					fmt.Printf("%s\n", err)
					result = 1
				}
				continue
			}
			fmt.Printf("%s -> %q: ", path, cachedFilePath)
			if err := source.CacheURL(path); err != nil {
				fmt.Printf("%s\n", err)
				result = 1
			} else {
				fmt.Printf("okay\n")
			}
		}
		os.Exit(result)
	}
	testfiles.AddFileSource(testfiles.URLFileSource{
		CacheDir: finalURLCacheDir,
		Download: *urlDownload,
	})

	// Verify that all registered files are available.
	for _, path := range registered {
		if _, err := testfiles.Read(path); err != nil {
			// The error is very detailed and includes information about available test files.
			// Also, retrieving files might fail for the same reason, like an incorrect
			// repo root. Therefore it makes no sense to continue after reporting the first
			// error.
			fmt.Fprintf(os.Stderr, "%s\n", err)
			os.Exit(1)
		}
	}

	// Run tests.
	os.Exit(m.Run())
}

func TestE2E(t *testing.T) {
	RunE2ETests(t)
}
