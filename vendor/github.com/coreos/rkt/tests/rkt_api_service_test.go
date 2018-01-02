// Copyright 2015 The rkt Authors
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

// +build host coreos src kvm

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/api/v1alpha"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/tests/testutils"
	"golang.org/x/net/context"
)

func startAPIService(t *testing.T, ctx *testutils.RktRunCtx) *gexpect.ExpectSubprocess {
	noRktGid := false
	rktGid, err := common.LookupGid(common.RktGroup)
	if err != nil {
		t.Logf("no %q group, will run api service with root, ONLY DO THIS FOR TESTING!", common.RktGroup)
		noRktGid = true
	} else {
		if err := ctx.SetupDataDir(); err != nil {
			t.Fatalf("failed to setup data directory: %v", err)
		}
	}

	uid, _ := ctx.GetUidGidRktBinOwnerNotRoot()

	t.Logf("Running rkt api service")
	apisvcCmd := fmt.Sprintf("%s api-service", ctx.Cmd())

	if noRktGid {
		return startRktAndCheckOutput(t, apisvcCmd, "API service running")
	}
	return startRktAsUidGidAndCheckOutput(t, apisvcCmd, "API service running", false, uid, rktGid)
}

func stopAPIService(t *testing.T, svc *gexpect.ExpectSubprocess) {
	if err := svc.Cmd.Process.Signal(syscall.SIGINT); err != nil {
		t.Fatalf("Failed to stop the api service: %v", err)
	}
	waitOrFail(t, svc, 0)
}

func checkPodState(t *testing.T, rawState string, apiState v1alpha.PodState) {
	switch rawState {
	case "embryo":
		if apiState == v1alpha.PodState_POD_STATE_EMBRYO {
			return
		}
	case "preparing":
		if apiState == v1alpha.PodState_POD_STATE_PREPARING {
			return
		}
	case "aborted prepare":
		if apiState == v1alpha.PodState_POD_STATE_ABORTED_PREPARE {
			return
		}
	case "running":
		if apiState == v1alpha.PodState_POD_STATE_RUNNING {
			return
		}
	case "deleting":
		if apiState == v1alpha.PodState_POD_STATE_DELETING {
			return
		}
	case "exited":
		if apiState == v1alpha.PodState_POD_STATE_EXITED {
			return
		}
	case "garbage", "exited garbage":
		if apiState == v1alpha.PodState_POD_STATE_GARBAGE {
			return
		}
	default:
		t.Fatalf("Unexpected state: %v", rawState)
	}
	t.Errorf("Pod state returned by api-service (%q) is not equivalent to the state returned by 'rkt status' (%q)", apiState, rawState)
}

func checkPodApps(t *testing.T, rawPod *podInfo, apiApps []*v1alpha.App, hasAppState bool) {
	rawApps := rawPod.apps
	if len(rawApps) != len(apiApps) {
		t.Errorf("Expected %d apps, saw %d apps returned by api service %v", len(rawApps), len(apiApps), apiApps)
	}

	for _, app := range apiApps {
		appInfo, ok := rawApps[app.Name]
		if !ok {
			t.Errorf("Expected app (name: %q) in the app list", app.Name)
			continue
		}

		appACName := types.MustACName(app.Name)
		runtimeApp := rawPod.manifest.Apps.Get(*appACName)
		if runtimeApp == nil {
			t.Errorf("Expected app (name: %q) in the pod manifest", app.Name)
		}

		if hasAppState && appInfo.exitCode != int(app.ExitCode) {
			t.Errorf("Expected %v, saw %v", appInfo.exitCode, app.ExitCode)
		}
		// Image hash in the pod manifest can be partial hash.
		if !strings.HasPrefix(app.Image.Id, appInfo.image.id) {
			t.Errorf("Expected partial hash of %q, saw %q", appInfo.image.id, app.Image.Id)
		}

		// Check app annotations.
		checkAnnotations(t, runtimeApp.Annotations, app.Annotations)
	}
}

func checkPodNetworks(t *testing.T, rawNets map[string]*networkInfo, apiNets []*v1alpha.Network) {
	if len(rawNets) != len(apiNets) {
		t.Errorf("Expected %d networks, saw %d networks returned by api service", len(rawNets), len(apiNets))
	}

	// Each network should have a unique name, so iteration over one list is enough given
	// the lengths of the two lists are equal.
	for _, net := range apiNets {
		if netInfo, ok := rawNets[net.Name]; ok {
			if netInfo.ipv4 != net.Ipv4 {
				t.Errorf("Expected %q, saw %q", netInfo.ipv4, net.Ipv4)
			}
		} else {
			t.Errorf("Expected network (name: %q, ipv4: %q) in networks", netInfo.name, netInfo.ipv4)
		}
	}
}

// Check the pod's information by 'rkt status'.
func checkPod(t *testing.T, ctx *testutils.RktRunCtx, p *v1alpha.Pod, hasAppState, hasManifest bool, expectedGCTime time.Time) {
	t.Logf("API Pod info: %v", p)

	podInfo := getPodInfo(t, ctx, p.Id)
	t.Logf("Pod info: %+v", podInfo)

	if podInfo.id != p.Id {
		t.Errorf("Expected %q, saw %q", podInfo.id, p.Id)
	}
	if podInfo.pid != int(p.Pid) {
		t.Errorf("Expected %d, saw %d", podInfo.pid, p.Pid)
	}
	// The time accuracy returned by 'rkt status' stops at milliseconds.
	accuracy := time.Millisecond.Nanoseconds()
	if podInfo.createdAt/accuracy != p.CreatedAt/accuracy {
		t.Errorf("Expected %d, saw %d", podInfo.createdAt, p.CreatedAt)
	}
	if podInfo.startedAt/accuracy != p.StartedAt/accuracy {
		t.Errorf("Expected %d, saw %d", podInfo.startedAt, p.StartedAt)
	}

	// If expectedGCTime.IsZero() == true, then p.GcMarkedAt should also be zero.
	actualTime := time.Unix(0, p.GcMarkedAt)
	if !compareTime(expectedGCTime, actualTime) {
		t.Errorf("API service returned an incorrect GC marked time. Got %q, Expect: %q", actualTime, expectedGCTime)
	}
	checkPodState(t, podInfo.state, p.State)
	checkPodApps(t, podInfo, p.Apps, hasAppState)
	checkPodNetworks(t, podInfo.networks, p.Networks)

	expectedCgroupSuffix := ""
	if podInfo.state == "running" {
		machineID := fmt.Sprintf("rkt-%s", p.Id)
		escapedmID := strings.Replace(machineID, "-", "\\x2d", -1)
		expectedCgroupSuffix = fmt.Sprintf("/machine-%s.scope", escapedmID)
	}

	if !strings.HasSuffix(p.Cgroup, expectedCgroupSuffix) {
		t.Errorf("Expected the cgroup suffix to have %q, but saw %q", expectedCgroupSuffix, p.Cgroup)
	}

	if hasManifest && podInfo.manifest.Annotations != nil {
		checkAnnotations(t, podInfo.manifest.Annotations, p.Annotations)
	}

	msft, err := json.Marshal(podInfo.manifest)
	if err != nil {
		t.Errorf("Cannot marshal manifest: %v", err)
	}

	if hasManifest && !bytes.Equal(msft, p.Manifest) {
		t.Errorf("Expected %q, saw %q", string(msft), string(p.Manifest))
	} else if !hasManifest && p.Manifest != nil {
		t.Errorf("Expected nil manifest")
	}
}

func checkPodBasicsWithGCTime(t *testing.T, ctx *testutils.RktRunCtx, p *v1alpha.Pod, expectedGCTime time.Time) {
	checkPod(t, ctx, p, false, false, expectedGCTime)
}

func checkPodBasics(t *testing.T, ctx *testutils.RktRunCtx, p *v1alpha.Pod) {
	checkPod(t, ctx, p, false, false, time.Time{})
}

func checkPodDetails(t *testing.T, ctx *testutils.RktRunCtx, p *v1alpha.Pod) {
	checkPod(t, ctx, p, true, true, time.Time{})
}

// Check the image's information by 'rkt image list'.
func checkImage(t *testing.T, ctx *testutils.RktRunCtx, m *v1alpha.Image, hasManifest bool) {
	imgInfo := getImageInfo(t, ctx, m.Id)
	if imgInfo.id != m.Id {
		t.Errorf("Expected %q, saw %q", imgInfo.id, m.Id)
	}
	if imgInfo.name != m.Name {
		t.Errorf("Expected %q, saw %q", imgInfo.name, m.Name)
	}
	if imgInfo.version != m.Version {
		t.Errorf("Expected %q, saw %q", imgInfo.version, m.Version)
	}
	if imgInfo.importTime != m.ImportTimestamp {
		t.Errorf("Expected %q, saw %q", imgInfo.importTime, m.ImportTimestamp)
	}
	if imgInfo.size != m.Size {
		t.Errorf("Expected size %d, saw %d", imgInfo.size, m.Size)
	}

	if hasManifest {
		var mfst schema.ImageManifest
		err := json.Unmarshal(imgInfo.manifest, &mfst)
		if err != nil {
			t.Fatal(err)
		}
		if mfst.Annotations != nil {
			checkAnnotations(t, mfst.Annotations, m.Annotations)
		}
	}

	if hasManifest && !bytes.Equal(imgInfo.manifest, m.Manifest) {
		t.Errorf("Expected %q, saw %q", string(imgInfo.manifest), string(m.Manifest))
	} else if !hasManifest && m.Manifest != nil {
		t.Errorf("Expected nil manifest")
	}
}

func checkAnnotations(t *testing.T, expected types.Annotations, actual []*v1alpha.KeyValue) {
	if len(expected) != len(actual) {
		t.Fatalf("Expected annotation counts to equal, expected %d, got %d", len(expected), len(actual))
	}
	for _, a := range actual {
		val, ok := expected.Get(a.Key)
		if !ok {
			t.Fatalf("Expected annotation for key %q, got nothing", a.Key)
		}
		if val != a.Value {
			t.Fatalf("Incorrect Annotation value, expected %q, got %q", val, a.Value)
		}
	}
}

func checkImageBasics(t *testing.T, ctx *testutils.RktRunCtx, m *v1alpha.Image) {
	checkImage(t, ctx, m, false)
}

func checkImageDetails(t *testing.T, ctx *testutils.RktRunCtx, m *v1alpha.Image) {
	checkImage(t, ctx, m, true)
}

func TestAPIServiceGetInfo(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	svc := startAPIService(t, ctx)
	defer stopAPIService(t, svc)

	c, conn := newAPIClientOrFail(t, "localhost:15441")
	defer conn.Close()

	resp, err := c.GetInfo(context.Background(), &v1alpha.GetInfoRequest{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expectedAPIVersion := "1.0.0-alpha"
	if resp.Info.ApiVersion != expectedAPIVersion {
		t.Errorf("Expected api version to be %q, but saw %q", expectedAPIVersion, resp.Info.ApiVersion)
	}

	expectedGlobalFlags := &v1alpha.GlobalFlags{
		Dir:             ctx.DataDir(),
		SystemConfigDir: ctx.SystemDir(),
		LocalConfigDir:  ctx.LocalDir(),
		UserConfigDir:   ctx.UserDir(),
		InsecureFlags:   "none",
	}
	if !reflect.DeepEqual(resp.Info.GlobalFlags, expectedGlobalFlags) {
		t.Errorf("Expected global flags to be %v, but saw %v", expectedGlobalFlags, resp.Info.GlobalFlags)
	}
}

func NewAPIServiceListInspectPodsTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		svc := startAPIService(t, ctx)
		defer stopAPIService(t, svc)

		c, conn := newAPIClientOrFail(t, "localhost:15441")
		defer conn.Close()

		resp, err := c.ListPods(context.Background(), &v1alpha.ListPodsRequest{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(resp.Pods) != 0 {
			t.Errorf("Unexpected result: %v, should see zero pods", resp.Pods)
		}

		patches := []string{"--exec=/inspect --print-msg=HELLO_API --exit-code=0"}
		imageHash, err := patchImportAndFetchHash("rkt-inspect-print.aci", patches, t, ctx)
		if err != nil {
			t.Fatalf("%v", err)
		}
		imgID, err := types.NewHash(imageHash)
		if err != nil {
			t.Fatalf("Cannot generate types.Hash from %v: %v", imageHash, err)
		}

		podManifests := []struct {
			mfst             schema.PodManifest
			net              string
			expectedExitCode int
		}{
			{
				// 1, Good pod.
				schema.PodManifest{
					ACKind:    schema.PodManifestKind,
					ACVersion: schema.AppContainerVersion,
					Apps: []schema.RuntimeApp{
						{
							Name: types.ACName("rkt-inspect"),
							Image: schema.RuntimeImage{
								Name: types.MustACIdentifier("coreos.com/rkt-inspect"),
								ID:   *imgID,
							},
							Annotations: []types.Annotation{{Name: types.ACIdentifier("app-test"), Value: "app-test"}},
						},
					},
					Annotations: []types.Annotation{
						{Name: types.ACIdentifier("test"), Value: "test"},
					},
				},
				"default",
				0,
			},
			{
				// 2, Bad pod, won't be launched correctly.
				schema.PodManifest{
					ACKind:    schema.PodManifestKind,
					ACVersion: schema.AppContainerVersion,
					Apps: []schema.RuntimeApp{
						{
							Name: types.ACName("rkt-inspect"),
							Image: schema.RuntimeImage{
								Name: types.MustACIdentifier("coreos.com/rkt-inspect"),
								ID:   *imgID,
							},
						},
					},
				},
				"non-existent-network",
				254,
			},
		}

		// Launch the pods.
		for _, entry := range podManifests {
			manifestFile := generatePodManifestFile(t, &entry.mfst)
			defer os.Remove(manifestFile)

			runCmd := fmt.Sprintf("%s run --net=%s --pod-manifest=%s", ctx.Cmd(), entry.net, manifestFile)
			waitOrFail(t, spawnOrFail(t, runCmd), entry.expectedExitCode)
		}

		time.Sleep(delta)

		gcCmd := fmt.Sprintf("%s gc --mark-only=true", ctx.Cmd())
		waitOrFail(t, spawnOrFail(t, gcCmd), 0)

		gcTime := time.Now()

		// ListPods(detail=false).
		resp, err = c.ListPods(context.Background(), &v1alpha.ListPodsRequest{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(resp.Pods) != len(podManifests) {
			t.Errorf("Unexpected result: %v, should see %v pods", len(resp.Pods), len(podManifests))
		}

		for _, p := range resp.Pods {
			checkPodBasicsWithGCTime(t, ctx, p, gcTime)

			// Test InspectPod().
			inspectResp, err := c.InspectPod(context.Background(), &v1alpha.InspectPodRequest{Id: p.Id})
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			checkPodDetails(t, ctx, inspectResp.Pod)
		}

		// ListPods(detail=true).
		resp, err = c.ListPods(context.Background(), &v1alpha.ListPodsRequest{Detail: true})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(resp.Pods) != len(podManifests) {
			t.Errorf("Unexpected result: %v, should see %v pods", len(resp.Pods), len(podManifests))
		}

		for _, p := range resp.Pods {
			checkPodDetails(t, ctx, p)
		}

		// ListPods with corrupt pod directory
		// Note that we don't checkPodDetails here, the failure this is testing is
		// the api server panicing, which results in a list call hanging for ages
		// and then failing.
		// TODO: do further validation on the partial pods returned
		for _, p := range resp.Pods {
			numRemoved := 0
			podDir := getPodDir(t, ctx, p.Id)
			filepath.Walk(filepath.Join(podDir, "appsinfo"), filepath.WalkFunc(func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				if info.Name() == "manifest" {
					os.Remove(path)
					numRemoved++
				}
				return nil
			}))
			if numRemoved == 0 {
				t.Fatalf("Expected to remove at least one app manifest for pod %v", p)
			}
		}

		// ListPods(detail=true).
		resp, err = c.ListPods(context.Background(), &v1alpha.ListPodsRequest{Detail: true})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if len(resp.Pods) != len(podManifests) {
			t.Fatalf("Expected %v pods, got %v pods", len(podManifests), len(resp.Pods))
		}
	})
}

func TestAPIServiceListInspectImages(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	svc := startAPIService(t, ctx)
	defer stopAPIService(t, svc)

	c, conn := newAPIClientOrFail(t, "localhost:15441")
	defer conn.Close()

	resp, err := c.ListImages(context.Background(), &v1alpha.ListImagesRequest{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(resp.Images) != 0 {
		t.Errorf("Unexpected result: %v, should see zero images", resp.Images)
	}

	_, err = patchImportAndFetchHash("rkt-inspect-sleep.aci", []string{"--exec=/inspect"}, t, ctx)
	if err != nil {
		t.Fatalf("%v", err)
	}

	// ListImages(detail=false).
	resp, err = c.ListImages(context.Background(), &v1alpha.ListImagesRequest{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(resp.Images) == 0 {
		t.Errorf("Unexpected result: %v, should see non-zero images", resp.Images)
	}

	for _, m := range resp.Images {
		checkImageBasics(t, ctx, m)

		// Test InspectImage().
		inspectResp, err := c.InspectImage(context.Background(), &v1alpha.InspectImageRequest{Id: m.Id})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		checkImageDetails(t, ctx, inspectResp.Image)
	}

	// ListImages(detail=true).
	resp, err = c.ListImages(context.Background(), &v1alpha.ListImagesRequest{Detail: true})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(resp.Images) == 0 {
		t.Errorf("Unexpected result: %v, should see non-zero images", resp.Images)
	}

	for _, m := range resp.Images {
		checkImageDetails(t, ctx, m)
	}
}

func NewAPIServiceCgroupTest() testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		svc := startAPIService(t, ctx)
		defer stopAPIService(t, svc)

		c, conn := newAPIClientOrFail(t, "localhost:15441")
		defer conn.Close()

		aciFileName := patchTestACI("rkt-inspect-interactive.aci", "--exec=/inspect --read-stdin")
		defer os.Remove(aciFileName)

		runCmd := fmt.Sprintf("%s --insecure-options=image run --interactive %s", ctx.Cmd(), aciFileName)
		child := spawnOrFail(t, runCmd)

		var resp *v1alpha.ListPodsResponse
		var err error
		done := make(chan struct{})

		// Wait the pods to be running.
		go func() {
			for {
				// ListPods(detail=false).
				resp, err = c.ListPods(context.Background(), &v1alpha.ListPodsRequest{})
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}

				if len(resp.Pods) != 0 {
					allRunning := true
					for _, p := range resp.Pods {
						if p.State != v1alpha.PodState_POD_STATE_RUNNING || p.Pid == -1 {
							allRunning = false
							break
						}
					}
					if allRunning {
						t.Logf("Pods are running")
						close(done)
						return
					}
				}
				t.Logf("Pods are not in RUNNING state")
				time.Sleep(time.Second)
			}
		}()

		testutils.WaitOrTimeout(t, time.Second*60, done)

		var cgroups []string
		var subcgroups []string

		for _, p := range resp.Pods {
			checkPodBasics(t, ctx, p)

			// Test InspectPod().
			inspectResp, err := c.InspectPod(context.Background(), &v1alpha.InspectPodRequest{Id: p.Id})
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			checkPodDetails(t, ctx, inspectResp.Pod)
			if p.Cgroup != "" {
				cgroups = append(cgroups, p.Cgroup)
				subcgroups = append(subcgroups, filepath.Join(p.Cgroup, "system.slice"))
			}
		}

		// ListPods(detail=true). Filter according to the cgroup.
		t.Logf("Calling ListPods with cgroup filter %v", cgroups)
		resp, err = c.ListPods(context.Background(), &v1alpha.ListPodsRequest{
			Detail:  true,
			Filters: []*v1alpha.PodFilter{{Cgroups: cgroups}},
		})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(resp.Pods) == 0 {
			t.Errorf("Unexpected result: %v, should see non-zero pods", resp.Pods)
		}

		for _, p := range resp.Pods {
			checkPodDetails(t, ctx, p)
		}

		t.Logf("Calling ListPods with subcgroup filter %v", subcgroups)
		resp, err = c.ListPods(context.Background(), &v1alpha.ListPodsRequest{
			Detail:  true,
			Filters: []*v1alpha.PodFilter{{PodSubCgroups: subcgroups}},
		})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(resp.Pods) == 0 {
			t.Errorf("Unexpected result: %v, should see non-zero pods", resp.Pods)
		}

		for _, p := range resp.Pods {
			checkPodDetails(t, ctx, p)
		}

		// Terminate the pod.
		if err := child.SendLine("Good bye"); err != nil {
			t.Fatalf("Failed to send message to the pod: %v", err)
		}
		waitOrFail(t, child, 0)

		// Check that there's no cgroups returned for non-running pods.
		cgroups = []string{}
		resp, err = c.ListPods(context.Background(), &v1alpha.ListPodsRequest{})
		for _, p := range resp.Pods {
			checkPodBasics(t, ctx, p)
			if p.Cgroup != "" {
				cgroups = append(cgroups, p.Cgroup)
			}
		}
		if len(cgroups) != 0 {
			t.Errorf("Unexpected cgroup returned by pods: %v", cgroups)
		}
	})
}
