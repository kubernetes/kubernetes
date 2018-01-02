// Copyright 2016 The rkt Authors
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

// +build coreos kvm host fly

package main

import (
	"fmt"
	"os"
	"path"
	"syscall"
	"testing"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/tests/testutils"
)

// TODO: unite these tests with rkt_run_pod_manifest_test.go

var (
	boolTrue  = true
	boolFalse = false

	mountName     types.ACName = "mnt"
	mountDir                   = "/mnt"
	mountFilePath              = "/mnt/subDirRW/file"

	volDir      = mustTempDir("rkt-tests-volume-data")
	volSubDirRW = path.Join(volDir, "subDirRW")
	volFilePath = path.Join(volSubDirRW, "file")

	innerFileContent = "inner"
	outerFileContent = "outer"
)

func prepareTmpDirWithRecursiveMountsAndFiles(t *testing.T) []func() {
	cleanupFuncs := make([]func(), 0)

	// create directories on the host
	if err := os.MkdirAll(volSubDirRW, 0); err != nil {
		t.Fatalf("Can't create directory %q: %v", volSubDirRW, err)
	}
	cleanupFuncs = append(cleanupFuncs, func() { os.RemoveAll(volDir) })

	// create the file in subDirRW before the mount
	tmpdir2outerfile, err := os.Create(volFilePath)
	if err != nil {
		executeFuncsReverse(cleanupFuncs)
		t.Fatalf("Can't create outer file %q: %v", volSubDirRW, err)
	}
	cleanupFuncs = append(cleanupFuncs, func() { tmpdir2outerfile.Close() })

	if _, err := tmpdir2outerfile.WriteString(outerFileContent); err != nil {
		executeFuncsReverse(cleanupFuncs)
		t.Fatalf("Can't write to file %q after mounting: %v", tmpdir2outerfile, err)
	}

	// mount tmpfs for /dir1/subDirRW
	if err := syscall.Mount("", volSubDirRW, "tmpfs", 0, ""); err != nil {
		executeFuncsReverse(cleanupFuncs)
		t.Fatalf("Can't mount tmpfs on inner temp directory %q: %v", volSubDirRW, err)
	}
	cleanupFuncs = append(cleanupFuncs, func() {
		if err := syscall.Unmount(volSubDirRW, syscall.MNT_DETACH); err != nil {
			t.Errorf("could not unmount %q: %v", volSubDirRW, err)
		}
	})
	cleanupFuncs = append(cleanupFuncs, func() { os.RemoveAll(volDir) })

	// create the file in subDirRW after the mount
	tmpdir2innerfile, err := os.Create(volFilePath)
	if err != nil {
		executeFuncsReverse(cleanupFuncs)
		t.Fatalf("Can't create inner file %q: %v", volSubDirRW, err)
	}
	cleanupFuncs = append(cleanupFuncs, func() { tmpdir2innerfile.Close() })

	if _, err := tmpdir2innerfile.WriteString(innerFileContent); err != nil {
		executeFuncsReverse(cleanupFuncs)
		t.Fatalf("Can't write to file %q after mounting: %v", tmpdir2innerfile, err)
	}

	return cleanupFuncs
}

type volumeMountTestCase struct {
	description string
	// [image name]:[image patches]
	images         []imagePatch
	cmdArgs        string
	podManifest    *schema.PodManifest
	expectedResult string
}

var (
	volumeMountTestCasesRecursiveCLI = []volumeMountTestCase{
		{
			"CLI: recursive mount read file",
			[]imagePatch{
				{
					"rkt-test-run-read-file.aci",
					[]string{fmt.Sprintf("--exec=/inspect --read-file --file-name %s", mountFilePath)},
				},
			},
			fmt.Sprintf(
				"--volume=test1,kind=host,source=%s,recursive=true --mount volume=test1,target=%s",
				volDir, mountDir,
			),
			nil,
			innerFileContent,
		},
		{
			"CLI: recursive read-only mount write file must fail",
			[]imagePatch{
				{
					"rkt-test-run-write-file.aci",
					[]string{fmt.Sprintf("--exec=/inspect --write-file --file-name %s", mountFilePath)},
				},
			},
			fmt.Sprintf(
				"--volume=test1,kind=host,source=%s,recursive=true,readOnly=true --mount volume=test1,target=%s",
				volDir, mountDir,
			),
			nil,
			"read-only file system",
		},
	}

	volumeMountTestCasesNonRecursiveCLI = []volumeMountTestCase{
		{
			"CLI: read file with non-recursive mount",
			[]imagePatch{
				{
					"rkt-test-run-read-file.aci",
					[]string{fmt.Sprintf("--exec=/inspect --read-file --file-name %s", mountFilePath)},
				},
			},
			fmt.Sprintf(
				"--volume=test,kind=host,source=%s,recursive=false --mount volume=test,target=%s",
				volDir, mountDir,
			),
			nil,
			outerFileContent,
		},
		{
			"CLI: read-only non-recursive write file must fail",
			[]imagePatch{
				{
					"rkt-test-run-write-file.aci",
					[]string{fmt.Sprintf("--exec=/inspect --write-file --file-name %s", "/mnt/lol")},
				},
			},
			fmt.Sprintf(
				"--volume=test1,kind=host,source=%s,readOnly=true,recursive=false --mount volume=test1,target=%s",
				volDir, mountDir,
			),
			nil,
			"read-only file system",
		},
	}

	volumeMountTestCasesRecursivePodManifest = []volumeMountTestCase{
		{
			"Read of nested file for recursive mount",
			[]imagePatch{
				{"rkt-test-run-pod-manifest-recursive-mount-stat.aci", []string{}},
			},
			"",
			&schema.PodManifest{
				Apps: []schema.RuntimeApp{
					{
						Name: baseAppName,
						App: &types.App{
							Exec:  []string{"/inspect", "--read-file"},
							User:  "0",
							Group: "0",
							Environment: []types.EnvironmentVariable{
								{"FILE", mountFilePath},
							},
							MountPoints: []types.MountPoint{
								{mountName, mountDir, false},
							},
						},
					},
				},
				Volumes: []types.Volume{
					{Name: mountName, Kind: "host", Source: volDir,
						ReadOnly: nil, Recursive: &boolTrue,
						Mode: nil, UID: nil, GID: nil},
				},
			},
			innerFileContent,
		},
		{
			"Write of nested file for recursive/read-only mount must fail",
			[]imagePatch{
				{"rkt-test-run-pod-manifest-recursive-mount-write.aci", []string{}},
			},
			"",
			&schema.PodManifest{
				Apps: []schema.RuntimeApp{
					{
						Name: baseAppName,
						App: &types.App{
							Exec:  []string{"/inspect", "--write-file"},
							User:  "0",
							Group: "0",
							Environment: []types.EnvironmentVariable{
								{"FILE", mountFilePath},
								{"CONTENT", "should-not-see-me"},
							},
							MountPoints: []types.MountPoint{
								{mountName, mountDir, false},
							},
						},
					},
				},
				Volumes: []types.Volume{
					{Name: mountName, Kind: "host", Source: volDir,
						ReadOnly: &boolTrue, Recursive: &boolTrue,
						Mode: nil, UID: nil, GID: nil},
				},
			},
			"read-only file system",
		},
	}

	volumeMountTestCasesNonRecursivePodManifest = []volumeMountTestCase{
		{
			"PodManifest: Simple read after write with volume non-recursive mounted in a read-only rootfs.",
			[]imagePatch{
				{"rkt-test-run-pod-manifest-read-only-rootfs-vol-rw.aci", []string{}},
			},
			"",
			&schema.PodManifest{
				Apps: []schema.RuntimeApp{
					{
						Name: baseAppName,
						App: &types.App{
							Exec:  []string{"/inspect", "--write-file", "--read-file"},
							User:  "0",
							Group: "0",
							Environment: []types.EnvironmentVariable{
								{"FILE", path.Join(mountDir, "file")},
								{"CONTENT", "host:foo"},
							},
							MountPoints: []types.MountPoint{
								{mountName, mountDir, false},
							},
						},
						ReadOnlyRootFS: true,
					},
				},
				Volumes: []types.Volume{
					{Name: mountName, Kind: "host", Source: volDir,
						ReadOnly: nil, Recursive: &boolFalse,
						Mode: nil, UID: nil, GID: nil},
				},
			},
			"host:foo",
		},
	}

	volumeMountTestCasesNonRecursive = []volumeMountTestCase{
		{
			"Read of nested file for non-recursive mount",
			[]imagePatch{
				{"rkt-test-run-pod-manifest-recursive-mount-stat.aci", []string{}},
			},
			"",
			&schema.PodManifest{
				Apps: []schema.RuntimeApp{
					{
						Name: baseAppName,
						App: &types.App{
							Exec:  []string{"/inspect", "--read-file"},
							User:  "0",
							Group: "0",
							Environment: []types.EnvironmentVariable{
								{"FILE", mountFilePath},
							},
							MountPoints: []types.MountPoint{
								{mountName, mountDir, false},
							},
						},
					},
				},
				Volumes: []types.Volume{
					{Name: mountName, Kind: "host", Source: volDir,
						ReadOnly: nil, Recursive: &boolFalse,
						Mode: nil, UID: nil, GID: nil},
				},
			},
			outerFileContent,
		},
	}
)

func NewTestVolumeMount(volumeMountTestCases [][]volumeMountTestCase) testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		deferredFuncs := prepareTmpDirWithRecursiveMountsAndFiles(t)
		defer executeFuncsReverse(deferredFuncs)

		for _, testCases := range volumeMountTestCases {
			for i, tt := range testCases {
				var hashesToRemove []string
				for j, v := range tt.images {
					hash, err := patchImportAndFetchHash(v.name, v.patches, t, ctx)
					if err != nil {
						t.Fatalf("error running patchImportAndFetchHash: %v", err)
					}

					hashesToRemove = append(hashesToRemove, hash)
					if tt.podManifest != nil {
						imgName := types.MustACIdentifier(v.name)
						imgID, err := types.NewHash(hash)
						if err != nil {
							t.Fatalf("Cannot generate types.Hash from %v: %v", hash, err)
						}
						tt.podManifest.Apps[j].Image.Name = imgName
						tt.podManifest.Apps[j].Image.ID = *imgID
					}
				}

				manifestFile := ""
				if tt.podManifest != nil {
					tt.podManifest.ACKind = schema.PodManifestKind
					tt.podManifest.ACVersion = schema.AppContainerVersion

					manifestFile = generatePodManifestFile(t, tt.podManifest)
					defer os.Remove(manifestFile)
				}

				// 1. Test 'rkt run'.
				runCmd := fmt.Sprintf("%s run --mds-register=false", ctx.Cmd())
				if manifestFile != "" {
					runCmd += fmt.Sprintf(" --pod-manifest=%s", manifestFile)
				} else {
					// TODO: run the tests for more than just the first image
					runCmd += fmt.Sprintf(" %s %s", tt.cmdArgs, hashesToRemove[0])
				}
				t.Logf("Running 'run' test #%v: %q", i, tt.description)
				child := spawnOrFail(t, runCmd)
				ctx.RegisterChild(child)

				if tt.expectedResult != "" {
					if _, out, err := expectRegexWithOutput(child, tt.expectedResult); err != nil {
						t.Fatalf("Expected %q but not found: %v\n%s", tt.expectedResult, err, out)
					}
				}
				child.Wait()
				verifyHostFile(t, volDir, "file", i, tt.expectedResult)

				// 2. Test 'rkt prepare' + 'rkt run-prepared'.
				prepareCmd := fmt.Sprintf("%s prepare", ctx.Cmd())
				if manifestFile != "" {
					prepareCmd += fmt.Sprintf(" --pod-manifest=%s", manifestFile)
				} else {
					// TODO: run the tests for more than just the first image
					prepareCmd += fmt.Sprintf(" %s %s", tt.cmdArgs, hashesToRemove[0])
				}
				uuid := runRktAndGetUUID(t, prepareCmd)

				runPreparedCmd := fmt.Sprintf("%s run-prepared --mds-register=false %s", ctx.Cmd(), uuid)
				t.Logf("Running 'run-prepared' test #%v: %q", i, tt.description)
				child = spawnOrFail(t, runPreparedCmd)

				if tt.expectedResult != "" {
					if _, out, err := expectRegexWithOutput(child, tt.expectedResult); err != nil {
						t.Fatalf("Expected %q but not found: %v\n%s", tt.expectedResult, err, out)
					}
				}
				child.Wait()
				verifyHostFile(t, volDir, "file", i, tt.expectedResult)

				// we run the garbage collector and remove the imported images to save
				// space
				runGC(t, ctx)
				for _, h := range hashesToRemove {
					removeFromCas(t, ctx, h)
				}
			}
		}
	})
}
