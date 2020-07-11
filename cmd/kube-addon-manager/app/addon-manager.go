/*
Copyright 2020 The Kubernetes Authors.

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

// Package app contains the application logic for the kube-addon-manager.
package app

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/component-base/version/verflag"
	"k8s.io/klog/v2"
)

const (
	defaultAdmissionControlsDir = "/etc/kubernetes/admission-controls"
	systemNamespace             = "kube-system"
	startAddonAttempts          = 100
	startAddonRetryDelay        = 10 * time.Second

	// Addons could use this label with two modes:
	// - ADDON_MANAGER_LABEL=Reconcile
	// - ADDON_MANAGER_LABEL=EnsureExists
	addonManagerLabel = "addonmanager.kubernetes.io/mode"

	// This label is deprecated (only for Addon Manager). In future release
	// addon-manager may not respect it anymore. Addons with
	// CLUSTER_SERVICE_LABEL=true and without ADDON_MANAGER_LABEL=EnsureExists
	// will be reconciled for now.
	clusterServiceLabel = "kubernetes.io/cluster-service"
)

var (
	// defaultKubectlPruneWhitelistResources is a list of resources whitelisted by default.
	// This is currently the same with the default in:
	// https://github.com/kubernetes/kubectl/blob/master/pkg/cmd/apply/prune.go.
	// To override the default list with other values, set
	// KUBECTL_PRUNE_WHITELIST_OVERRIDE environment variable to space-separated
	// names of resources to whitelist.
	defaultKubectlPruneWhitelistResources = []string{
		"core/v1/ConfigMap",
		"core/v1/Endpoints",
		"core/v1/Namespace",
		"core/v1/PersistentVolumeClaim",
		"core/v1/PersistentVolume",
		"core/v1/Pod",
		"core/v1/ReplicationController",
		"core/v1/Secret",
		"core/v1/Service",
		"batch/v1/Job",
		"batch/v1beta1/CronJob",
		"apps/v1/DaemonSet",
		"apps/v1/Deployment",
		"apps/v1/ReplicaSet",
		"apps/v1/StatefulSet",
		"extensions/v1beta1/Ingress",
	}
	leaderRegexp = regexp.MustCompile(`^.*"holderIdentity":"([^"]*)".*`)
)

// NewAddonManagerCommand creates a *cobra.Command object with default parameters
func NewAddonManagerCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:  "addon-manager",
		Long: `The addon manager ensures the existence and value of a set of addon cluster Objects.`,
		Run: func(cmd *cobra.Command, args []string) {
			verflag.PrintAndExitIfRequested()
			cliflag.PrintFlags(cmd.Flags())

			am, err := newAddonManager(os.Getenv)
			if err != nil {
				klog.Exit(err)
			}

			if err := am.Run(); err != nil {
				klog.Exit(err)
			}
		},
		Args: func(cmd *cobra.Command, args []string) error {
			for _, arg := range args {
				if len(arg) > 0 {
					return fmt.Errorf("%q does not take any positional arguments, got %q", cmd.CommandPath(), args)
				}
			}
			return nil
		},
	}

	fs := cmd.Flags()
	verflag.AddFlags(fs)
	globalflag.AddGlobalFlags(fs, cmd.Name())

	return cmd
}

// addonManager has all of addon-manager's the configurable parameters.
type addonManager struct {
	addonPath            string
	admissionControlsDir string
	checkInterval        time.Duration
	hostname             string
	kubectlOpts          string
	kubectlPath          string
	leaderElection       bool
	pruneWhitelistFlags  []string

	// stubbable for testing
	kubectl func(stdout, stderr io.Writer, args ...string) error
}

func newAddonManager(env func(key string) string) (*addonManager, error) {
	am := &addonManager{
		addonPath:            "/etc/kubernetes/addons",
		admissionControlsDir: defaultAdmissionControlsDir,
		checkInterval:        60 * time.Second,
		hostname:             env("HOSTNAME"),
		kubectlOpts:          env("KUBECTL_OPTS"),
		kubectlPath:          "/usr/local/bin/kubectl",
		leaderElection:       true,
	}
	am.kubectl = am.kubectlExec
	if addonPath := env("ADDON_PATH"); addonPath != "" {
		am.addonPath = addonPath
	}
	if kubectlPath := env("KUBECTL_BIN"); kubectlPath != "" {
		am.kubectlPath = kubectlPath
	}
	if amle := env("ADDON_MANAGER_LEADER_ELECTION"); amle != "" {
		b, err := strconv.ParseBool(amle)
		if err != nil {
			return nil, fmt.Errorf("error converting ADDON_MANAGER_LEADER_ELECTION value %q to bool: %v", amle, err)
		}
		am.leaderElection = b
	}
	if testIntervalSec := env("TEST_ADDON_CHECK_INTERVAL_SEC"); testIntervalSec != "" {
		secs, err := strconv.ParseInt(testIntervalSec, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("error converting TEST_ADDON_CHECK_INTERVAL_SEC value %q to int64: %v", testIntervalSec, err)
		}
		am.checkInterval = time.Duration(secs) * time.Second
	}
	am.pruneWhitelistFlags = makePruneWhitelistFlags(env)
	return am, nil
}

// makePruneWhitelistFlags generates kubectl prune-whitelist flags from provided environment variables.
func makePruneWhitelistFlags(env func(key string) string) []string {
	pw := defaultKubectlPruneWhitelistResources
	if pwo := env("KUBECTL_PRUNE_WHITELIST_OVERRIDE"); pwo != "" {
		pw = strings.Fields(pwo)
	}
	pwf := appendPruneWhitelistFlags(nil, pw)
	pwf = appendPruneWhitelistFlags(pwf, strings.Fields(env("KUBECTL_EXTRA_PRUNE_WHITELIST")))
	klog.Infof("== Generated kubectl prune whitelist flags: %s ==", strings.Join(pwf, " "))
	return pwf
}

// appendPruneWhitelistFlags generates kubectl prune-whitelist flags from provided resource list.
func appendPruneWhitelistFlags(pwf, resources []string) []string {
	for _, resource := range resources {
		pwf = append(pwf, "--prune-whitelist", resource)
	}
	return pwf
}

func (m *addonManager) Run() error {
	m.waitForSystemServiceAccount()
	if err := m.ensureDefaultAdmissionControlsObjects(); err != nil {
		if os.IsNotExist(err) {
			klog.Warningf("admissions control object directory not found: %v", err)
		} else {
			klog.Errorf("could not create admissions control objects: %v", err)
		}
	}
	for {
		start := time.Now()

		if !m.leaderElection || m.isLeader() {
			m.ensureAddons()
			m.reconcileAddons()
		} else {
			klog.Info("== Another addon-manager is acting as leader. Sleeping... ==")
		}

		elapsed := time.Since(start)
		if elapsed < m.checkInterval {
			time.Sleep(m.checkInterval - elapsed)
		}
	}
}

func (m *addonManager) waitForSystemServiceAccount() {
	var out, errOut bytes.Buffer
	for {
		err := m.kubectl(&out, &errOut, m.kubectlOpts, "get", "-n", systemNamespace, "serviceaccount", "default", "-o", "go-template={{with index .secrets 0}}{{.name}}{{end}}")
		if errStr := errOut.String(); errStr != "" {
			klog.Warning(errStr)
		}
		if err == nil && out.String() != "" {
			return
		}
		out.Reset()
		errOut.Reset()
		time.Sleep(500 * time.Millisecond)
	}
}

// Create admission_control objects if defined before any other addon services. If the limits
// are defined in a namespace other than default, we should still create the limits for the
// default namespace.
func (m *addonManager) ensureDefaultAdmissionControlsObjects() error {
	var out, errOut bytes.Buffer
	return filepath.Walk(m.admissionControlsDir,
		func(path string, _ os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			ext := filepath.Ext(path)
			if ext == ".yaml" || ext == ".json" {
				for attemptsRemaining := startAddonAttempts; attemptsRemaining > 0; attemptsRemaining-- {
					err = m.kubectl(&out, &errOut, m.kubectlOpts, "--namespace=default", "apply", "-f", path)
					filterLog(&out, "", klog.Info)
					filterLog(&errOut, "", klog.Warning)
					if err == nil {
						klog.Infof("== Successfully started %s in namespace default ==", path)
						return nil
					}
					klog.Warningf("== Failed to start %s in namespace default. %d tries remaining. ==", path, attemptsRemaining-1)
					if attemptsRemaining > 1 {
						time.Sleep(startAddonRetryDelay)
					}
				}
			}
			return err
		})
}

// IsLeader returns whether this instance of addonManager should act as leader in a multi-master cluster.
func (m *addonManager) isLeader() bool {
	var out, errOut bytes.Buffer
	klog.Info("== Determining addon-manager leader ==")
	err := m.kubectl(&out, &errOut, m.kubectlOpts, "-n", systemNamespace, "get", "ep", "kube-controller-manager", "-o", `go-template={{index .metadata.annotations "control-plane.alpha.kubernetes.io/leader"}}`)
	filterLog(&errOut, "", klog.Warning)
	if err != nil {
		// If the leader can't be positively established, assume that we're it
		return true
	}
	holderIdentity := extractHolderIdentity(out.String())
	return holderIdentity == "" || strings.HasPrefix(holderIdentity, m.hostname+"_")
}

func extractHolderIdentity(raw string) string {
	captureGroups := leaderRegexp.FindStringSubmatch(raw)
	if len(captureGroups) != 2 {
		// A match yields one capture group (index 1)
		return ""
	}
	return captureGroups[1]
}

func (m *addonManager) reconcileAddons() {
	var out, errOut bytes.Buffer
	klog.Info("== Reconciling with deprecated label ==")

	// TODO: Remove the first command in future release.
	// Adding this for backward compatibility. Old addons have CLUSTER_SERVICE_LABEL=true and don't have
	// ADDON_MANAGER_LABEL=EnsureExists will still be reconciled.
	m.kubectl(&out, &errOut, append([]string{
		m.kubectlOpts,
		"apply",
		"-f", m.addonPath,
		"--recursive",
		"-l", clusterServiceLabel + "=true," + addonManagerLabel + "!=EnsureExists",
		"--prune=true"},
		m.pruneWhitelistFlags...)...)
	// Filter out `configured` message to not noisily log. `created`, `pruned` and errors will be logged.
	filterLog(&out, "configured", klog.Info)
	filterLog(&errOut, "configured", klog.Warning)
	out.Reset()
	errOut.Reset()
	klog.Info("== Reconciling with addon-manager label ==")

	m.kubectl(&out, &errOut, append([]string{
		m.kubectlOpts,
		"apply",
		"-f", m.addonPath,
		"--recursive",
		"-l", clusterServiceLabel + "!=true," + addonManagerLabel + "=Reconcile",
		"--prune=true"},
		m.pruneWhitelistFlags...)...)
	filterLog(&out, "configured", klog.Info)
	filterLog(&errOut, "configured", klog.Warning)
	klog.Info("== Kubernetes addon reconcile completed ==")
}

func (m *addonManager) ensureAddons() {
	var out bytes.Buffer
	m.kubectl(&out, &out,
		m.kubectlOpts,
		"create",
		"-f", m.addonPath,
		"--recursive",
		"-l", addonManagerLabel+"=EnsureExists")
	filterLog(&out, "AlreadyExists", klog.Info)
	klog.Info("== Kubernetes addon ensure completed ==")
}

func (m *addonManager) kubectlExec(stdout, stderr io.Writer, args ...string) error {
	cmd := exec.Command(m.kubectlPath, args...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return cmd.Run()
}

func filterLog(r io.Reader, doesNotContain string, logFunc func(...interface{})) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		if doesNotContain == "" || !strings.Contains(line, doesNotContain) {
			logFunc(line)
		}
	}
}
