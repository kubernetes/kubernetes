/*
Copyright The Kubernetes Authors.

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

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// setupCacheDir resolves or initializes the directory to cache downloaded tarballs.
func setupCacheDir(cacheDir string) (string, func(), error) {
	if cacheDir == "" {
		tmpCache, err := os.MkdirTemp("", "kube-deps-download-*")
		if err != nil {
			return "", nil, fmt.Errorf("failed to create temp download dir: %w", err)
		}
		return tmpCache, func() { _ = os.RemoveAll(tmpCache) }, nil
	}

	info, err := os.Stat(cacheDir)
	if err != nil {
		return "", nil, fmt.Errorf("cache directory does not exist: %w", err)
	}
	if !info.IsDir() {
		return "", nil, fmt.Errorf("cache-dir '%s' is not a directory", cacheDir)
	}
	if abs, err := filepath.Abs(cacheDir); err == nil {
		cacheDir = abs
	}
	return cacheDir, func() {}, nil
}

// downloadAndExtractTarball ensures a specific release archive is present and extracts it.
func downloadAndExtractTarball(version, tarball, cacheDir, extractTempDir string) error {
	url := fmt.Sprintf("https://dl.k8s.io/release/%s/%s", version, tarball)
	targetPath := filepath.Join(cacheDir, tarball)
	tempTargetPath := filepath.Join(cacheDir, ".tmp."+tarball)

	info, err := os.Stat(targetPath)
	if err != nil || info.Size() == 0 {
		cmd := exec.Command("curl", "-fsSL", "--retry", "3", "--keepalive-time", "2", url, "-o", tempTargetPath)
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			_ = os.Remove(tempTargetPath)
			if strings.Contains(tarball, "-client-") || strings.Contains(tarball, "-node-") {
				fmt.Printf("Warning: Failed to download %s (skipping optional archive)...\n", tarball)
				return nil
			}
			return fmt.Errorf("failed to download critical archive %s: %w", tarball, err)
		}
		if err := os.Rename(tempTargetPath, targetPath); err != nil {
			return fmt.Errorf("failed to rename temp file for %s: %w", tarball, err)
		}
	}

	cmd := exec.Command("tar", "-xzf", targetPath, "-C", extractTempDir)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to extract %s: %w", tarball, err)
	}
	return nil
}

// fetchGoMod retrieves the go.mod file for the specified version, trying local git first.
func fetchGoMod(version, extractTempDir string) (string, error) {
	goModPath := filepath.Join(extractTempDir, "go.mod")
	gitCmd := exec.Command("git", "show", fmt.Sprintf("%s:go.mod", version))
	if out, err := gitCmd.Output(); err == nil {
		if err := os.WriteFile(goModPath, out, 0644); err != nil {
			return "", fmt.Errorf("failed to write go.mod from git tag: %w", err)
		}
		return goModPath, nil
	}

	goModURL := fmt.Sprintf("https://raw.githubusercontent.com/kubernetes/kubernetes/%s/go.mod", version)
	curlCmd := exec.Command("curl", "-fsSL", goModURL, "-o", goModPath)
	curlCmd.Stderr = os.Stderr
	if err := curlCmd.Run(); err != nil {
		return "", fmt.Errorf("failed to retrieve go.mod for version %s from %s: %w", version, goModURL, err)
	}
	return goModPath, nil
}

// processRemoteVersion downloads and extracts release tarballs and go.mod for a given Kubernetes version,
// then scans the extracted binaries and parses the go.mod file.
func processRemoteVersion(version, cacheDir string) (map[string]bool, *GoModInfo, error) {
	cacheDir, cleanup, err := setupCacheDir(cacheDir)
	if err != nil {
		return nil, nil, err
	}
	defer cleanup()

	extractTempDir, err := os.MkdirTemp(cacheDir, "kube-deps-extract-*")
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create extraction temp dir: %w", err)
	}
	defer func() { _ = os.RemoveAll(extractTempDir) }()

	tarballs := []string{
		"kubernetes-server-linux-amd64.tar.gz",
		"kubernetes-server-linux-arm64.tar.gz",
		"kubernetes-server-linux-ppc64le.tar.gz",
		"kubernetes-server-linux-s390x.tar.gz",
		"kubernetes-client-darwin-amd64.tar.gz",
		"kubernetes-client-darwin-arm64.tar.gz",
		"kubernetes-client-windows-amd64.tar.gz",
		"kubernetes-node-windows-amd64.tar.gz",
	}

	for _, tarball := range tarballs {
		if err := downloadAndExtractTarball(version, tarball, cacheDir, extractTempDir); err != nil {
			return nil, nil, err
		}
	}

	goModPath, err := fetchGoMod(version, extractTempDir)
	if err != nil {
		return nil, nil, err
	}

	modInfo, err := parseGoMod(goModPath)
	if err != nil {
		return nil, nil, err
	}

	binariesDir := filepath.Join(extractTempDir, "kubernetes")
	productionDeps, err := scanBinaries(binariesDir)
	if err != nil {
		return nil, nil, err
	}

	return productionDeps, modInfo, nil
}
