/*
Copyright 2024 The Kubernetes Authors.

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

package state

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2/ktesting"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

const testCheckpoint = "pod_status_manager_state"

func newTestStateCheckpoint(t *testing.T) *stateCheckpoint {
	logger, _ := ktesting.NewTestContext(t)
	testingDir := getTestDir(t)
	cache := newStateMemory(logger, PodMap{})
	checkpointManager, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoError(t, err, "failed to create checkpoint manager")
	checkpointName := "pod_state_checkpoint"
	sc := &stateCheckpoint{
		cache:             cache,
		checkpointManager: checkpointManager,
		checkpointName:    checkpointName,
	}
	return sc
}

func getTestDir(t *testing.T) string {
	testingDir, err := os.MkdirTemp("", "pod_resource_allocation_state_test")
	require.NoError(t, err, "failed to create temp dir")
	t.Cleanup(func() {
		if err := os.RemoveAll(testingDir); err != nil {
			t.Fatal(err)
		}
	})
	return testingDir
}

func verifyPodResourceAllocation(t *testing.T, expected, actual *PodMap, msgAndArgs string) {
	require.Len(t, *actual, len(*expected), msgAndArgs)
	for podUID, expectedPod := range *expected {
		actualPod, exists := (*actual)[podUID]
		require.True(t, exists, "actual state missing pod %s", podUID)

		diff := cmp.Diff(expectedPod, actualPod, cmp.Comparer(func(x, y resource.Quantity) bool {
			return x.Equal(y)
		}))
		require.Empty(t, diff, msgAndArgs)
	}
}

func getPodMap(s State) PodMap {
	pods := PodMap{}
	for _, podUID := range s.GetPodUIDs() {
		pod, _ := s.GetPod(podUID)
		pods[podUID] = pod
	}
	return pods
}

// newTestPod returns a pod with one container that requests the given quantity of CPU and memory.
// It also has pod-level resources and an emptyDir volume with a size limit if asked to.
func newTestPod(uid types.UID, qStr string, podLevel, volumeLimit bool) *v1.Pod {
	requests := func() v1.ResourceList {
		return v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse(qStr),
			v1.ResourceMemory: resource.MustParse(qStr),
		}
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: uid},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "container1", Resources: v1.ResourceRequirements{Requests: requests()}},
			},
		},
	}
	if podLevel {
		pod.Spec.Resources = &v1.ResourceRequirements{Requests: requests()}
	}
	if volumeLimit {
		limit := resource.MustParse(qStr)
		pod.Spec.Volumes = []v1.Volume{
			{Name: "volume1", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{SizeLimit: &limit}}},
		}
	}
	return pod
}

func Test_stateCheckpoint_storeState(t *testing.T) {
	type args struct {
		podMap PodMap
	}
	type testCase struct {
		name string
		args args
	}

	var tests []testCase
	suffix := []string{"Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "n", "u", "m", "k", "M", "G", "T", "P", "E", ""}
	factor := []string{"1", "0.1", "0.03", "10", "100", "512", "1000", "1024", "700", "10000"}
	for _, fact := range factor {
		for _, suf := range suffix {
			if (suf == "E" || suf == "Ei") && (fact == "1000" || fact == "10000") {
				// when fact is 1000 or 10000, suffix "E" or "Ei", the quantity value is overflow
				// see detail https://github.com/kubernetes/apimachinery/blob/95b78024e3feada7739b40426690b4f287933fd8/pkg/api/resource/quantity.go#L301
				continue
			}
			qStr := fmt.Sprintf("%s%s", fact, suf)

			// Test case 1: All fields populated
			tests = append(tests, testCase{
				name: fmt.Sprintf("resource - %s - all fields populated", qStr),
				args: args{podMap: PodMap{"pod1": newTestPod("pod1", qStr, true, true)}},
			})

			// Test case 2: Only container resources populated (pod level and volume limits are nil)
			tests = append(tests, testCase{
				name: fmt.Sprintf("resource - %s - only container", qStr),
				args: args{podMap: PodMap{"pod1": newTestPod("pod1", qStr, false, false)}},
			})

			// Test case 3: Container resources and volume limits populated (pod level resources is nil)
			tests = append(tests, testCase{
				name: fmt.Sprintf("resource - %s - container and volume limits", qStr),
				args: args{podMap: PodMap{"pod1": newTestPod("pod1", qStr, false, true)}},
			})

			// Test case 4: Container resources and pod level resources populated (volume limits is nil)
			tests = append(tests, testCase{
				name: fmt.Sprintf("resource - %s - container and pod-level", qStr),
				args: args{podMap: PodMap{"pod1": newTestPod("pod1", qStr, true, false)}},
			})
		}
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			testDir := getTestDir(t)
			originalSC, err := NewStateCheckpoint(logger, testDir, testCheckpoint)
			require.NoError(t, err)

			for _, pod := range tt.args.podMap {
				err = originalSC.SetPod(logger, pod)
				require.NoError(t, err)
			}

			actual := getPodMap(originalSC)
			verifyPodResourceAllocation(t, &tt.args.podMap, &actual, "stored pod resource allocation is not equal to original pod resource allocation")

			newSC, err := NewStateCheckpoint(logger, testDir, testCheckpoint)
			require.NoError(t, err)

			actual = getPodMap(newSC)
			verifyPodResourceAllocation(t, &tt.args.podMap, &actual, "restored pod resource allocation is not equal to original pod resource allocation")

			checkpointPath := filepath.Join(testDir, testCheckpoint)
			require.FileExists(t, checkpointPath)
			require.NoError(t, os.Remove(checkpointPath)) // Remove the checkpoint file to track whether it's re-written.

			// Setting the pod allocations to the same values should not re-write the checkpoint.
			for _, pod := range tt.args.podMap {
				require.NoError(t, originalSC.SetPod(logger, pod))
				require.NoFileExists(t, checkpointPath, "checkpoint should not be re-written")
			}

			// Setting a new value should update the checkpoint.
			require.NoError(t, originalSC.SetPod(logger, newTestPod("foo-bar", "1", true, true)))
			require.FileExists(t, checkpointPath, "checkpoint should be re-written")
		})
	}
}

func Test_stateCheckpoint_formatUpgraded(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	// Based on the PodResourceAllocationInfo struct, it's mostly possible that new field will be added
	// in struct PodResourceAllocationInfo, rather than in struct PodResourceAllocationInfo.AllocationEntries.
	// Emulate upgrade scenario by pretending that `ResizeStatusEntries` is a new field.
	// The checkpoint content doesn't have it and that shouldn't prevent the checkpoint from being loaded.
	sc := newTestStateCheckpoint(t)

	// prepare old checkpoint, ResizeStatusEntries is unset,
	// pretend that the old checkpoint is unaware for the field ResizeStatusEntries
	const checkpointContent = `{"data":"{\"entries\":{\"pod1\":{\"ContainerResources\":{\"container1\":{\"requests\":{\"cpu\":\"1Ki\",\"memory\":\"1Ki\"}}}}}}","checksum":1178570812}`
	expectedPodResourceAllocation := PodMap{
		"pod1": {
			ObjectMeta: metav1.ObjectMeta{UID: "pod1"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1Ki"),
								v1.ResourceMemory: resource.MustParse("1Ki"),
							},
						},
					},
				},
			},
		},
	}
	checkpoint := &Checkpoint{}
	err := checkpoint.UnmarshalCheckpoint([]byte(checkpointContent))
	require.NoError(t, err, "failed to unmarshal checkpoint")

	err = sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	require.NoError(t, err, "failed to create old checkpoint")

	actualPodResourceAllocation, _, migrated, err := restoreState(logger, sc.checkpointManager, sc.checkpointName)
	require.NoError(t, err, "failed to restore state")
	require.True(t, migrated, "a checkpoint without a version should be migrated")

	verifyPodResourceAllocation(t, &expectedPodResourceAllocation, &actualPodResourceAllocation, "pod resource allocation info is not equal")

	sc.cache = newStateMemory(logger, actualPodResourceAllocation)

	actualPodResourceAllocation = getPodMap(sc.cache)

	verifyPodResourceAllocation(t, &expectedPodResourceAllocation, &actualPodResourceAllocation, "pod resource allocation info is not equal")
}

func Test_stateCheckpoint_migrationIsWrittenOnce(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testDir := getTestDir(t)
	checkpointPath := filepath.Join(testDir, testCheckpoint)

	const legacyContent = `{"data":"{\"entries\":{\"pod1\":{\"ContainerResources\":{\"container1\":{\"requests\":{\"cpu\":\"1Ki\",\"memory\":\"1Ki\"}}}}}}","checksum":1178570812}`
	require.NoError(t, os.WriteFile(checkpointPath, []byte(legacyContent), 0o600))

	sc, err := NewStateCheckpoint(logger, testDir, testCheckpoint)
	require.NoError(t, err)
	resources, found := sc.GetContainerResources("pod1", "container1")
	require.True(t, found)
	require.True(t, resource.MustParse("1Ki").Equal(*resources.Requests.Cpu()))

	// Starting up converted the checkpoint file.
	checkpointManager, err := checkpointmanager.NewCheckpointManager(testDir)
	require.NoError(t, err)
	checkpoint := &Checkpoint{}
	require.NoError(t, checkpointManager.GetCheckpoint(testCheckpoint, checkpoint))
	require.Equal(t, checkpointVersionV2, checkpoint.Version)

	// The next start finds nothing to migrate.
	_, _, migrated, err := restoreState(logger, checkpointManager, testCheckpoint)
	require.NoError(t, err)
	require.False(t, migrated)
}

func Test_stateCheckpoint_currentFormatIsNotRewritten(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testDir := getTestDir(t)
	checkpointPath := filepath.Join(testDir, testCheckpoint)

	// The pods are not sorted by UID, so any rewrite of the checkpoint would change the file.
	podList := &v1.PodList{Items: []v1.Pod{*newTestPod("pod-b", "1", false, false), *newTestPod("pod-a", "2", true, true)}}
	protoBytes, err := podList.Marshal()
	require.NoError(t, err)
	data, err := json.Marshal(CheckpointData{PodListProto: protoBytes})
	require.NoError(t, err)
	checkpoint := &Checkpoint{Version: checkpointVersionV2, Data: string(data), Checksum: checksum.New(string(data))}
	blob, err := checkpoint.MarshalCheckpoint()
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(checkpointPath, blob, 0o600))

	sc, err := NewStateCheckpoint(logger, testDir, testCheckpoint)
	require.NoError(t, err)
	require.ElementsMatch(t, []types.UID{"pod-a", "pod-b"}, sc.GetPodUIDs())

	actual, err := os.ReadFile(checkpointPath)
	require.NoError(t, err)
	require.Equal(t, blob, actual, "a checkpoint in the current format should not be rewritten on startup")
}

// blockingCheckpointManager holds every write of a checkpoint until release is called, as a slow disk
// would, so that a test can use the state while a write is in progress.
type blockingCheckpointManager struct {
	checkpointmanager.CheckpointManager // Only CreateCheckpoint is called.

	// started receives a value when a write begins, if nothing is waiting on the previous one.
	started  chan struct{}
	released chan struct{}
	release  func()

	mux     sync.Mutex
	written []*Checkpoint
}

func newBlockingCheckpointManager() *blockingCheckpointManager {
	released := make(chan struct{})
	return &blockingCheckpointManager{
		started:  make(chan struct{}, 1),
		released: released,
		release:  sync.OnceFunc(func() { close(released) }),
	}
}

func (m *blockingCheckpointManager) CreateCheckpoint(_ string, checkpoint checkpointmanager.Checkpoint) error {
	select {
	case m.started <- struct{}{}:
	default:
	}
	<-m.released

	m.mux.Lock()
	defer m.mux.Unlock()
	m.written = append(m.written, checkpoint.(*Checkpoint))
	return nil
}

func (m *blockingCheckpointManager) lastWritten() *Checkpoint {
	m.mux.Lock()
	defer m.mux.Unlock()
	return m.written[len(m.written)-1]
}

func Test_stateCheckpoint_slowWriteDoesNotBlockReaders(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	fake := newBlockingCheckpointManager()
	sc := &stateCheckpoint{
		cache:             newStateMemory(logger, PodMap{}),
		checkpointManager: fake,
		checkpointName:    testCheckpoint,
	}

	var wg sync.WaitGroup
	t.Cleanup(func() {
		// Don't leave anything stuck behind the write if the test failed.
		fake.release()
		wg.Wait()
	})

	const pods = 8
	podUID := func(i int) types.UID { return types.UID(fmt.Sprintf("pod%d", i)) }
	setPodErrs := make(chan error, pods)
	setPod := func(i int) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			setPodErrs <- sc.SetPod(logger, newTestPod(podUID(i), "1", true, true))
		}()
	}

	// The first write is stuck on the disk.
	setPod(0)
	select {
	case <-fake.started:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("the checkpoint was never written")
	}

	// The other updates still reach the cache, and everything in it can be read, while that write is stuck.
	for i := 1; i < pods; i++ {
		setPod(i)
	}
	readersDone := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(readersDone)
		assert.Eventually(t, func() bool { return len(sc.GetPodUIDs()) == pods }, wait.ForeverTestTimeout, time.Millisecond)
		for i := range pods {
			uid := podUID(i)
			assert.True(t, sc.HasPod(uid), "pod %s", uid)
			_, ok := sc.GetPod(uid)
			assert.True(t, ok, "pod %s", uid)
			_, ok = sc.GetContainerResources(uid, "container1")
			assert.True(t, ok, "pod %s", uid)
			_, ok = sc.GetPodLevelResources(uid)
			assert.True(t, ok, "pod %s", uid)
			_, ok = sc.GetEmptyDirVolumeLimit(uid, "volume1")
			assert.True(t, ok, "pod %s", uid)
		}
	}()
	select {
	case <-readersDone:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("the state could not be read while the checkpoint was being written")
	}

	// A write reads the state once its turn comes, so it also has what the cache got while it waited,
	// even though the update that added it has not written anything.
	require.NoError(t, sc.cache.SetPod(logger, newTestPod("late", "1", true, true)))

	// Once the disk catches up, the updates complete, and the last write has all of them.
	fake.release()
	wg.Wait()
	close(setPodErrs)
	for err := range setPodErrs {
		require.NoError(t, err)
	}

	podList, _, err := fake.lastWritten().GetPodList()
	require.NoError(t, err)
	expectedUIDs := []types.UID{"late"}
	for i := range pods {
		expectedUIDs = append(expectedUIDs, podUID(i))
	}
	var writtenUIDs []types.UID
	for _, pod := range podList.Items {
		writtenUIDs = append(writtenUIDs, pod.UID)
	}
	require.ElementsMatch(t, expectedUIDs, writtenUIDs, "the last write should have every update")
}

func Test_stateCheckpoint_concurrentUpdates(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testDir := getTestDir(t)
	sc, err := NewStateCheckpoint(logger, testDir, testCheckpoint)
	require.NoError(t, err)

	const pods, updates = 8, 20
	stop := make(chan struct{})
	var writers, readers sync.WaitGroup
	for i := range pods {
		podUID := types.UID(fmt.Sprintf("pod%d", i))
		writers.Add(1)
		go func() {
			defer writers.Done()
			for j := range updates {
				requests := v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(int64(j+1), resource.DecimalSI)}
				assert.NoError(t, sc.SetContainerResources(logger, podUID, "container1", podutil.Containers, v1.ResourceRequirements{Requests: requests}))
				assert.NoError(t, sc.SetPodLevelResources(logger, podUID, &v1.ResourceRequirements{Requests: requests}))
				limit := resource.MustParse(fmt.Sprintf("%dMi", j+1))
				assert.NoError(t, sc.SetEmptyDirVolumeLimit(podUID, "volume1", &limit))
			}
		}()
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				sc.GetPodUIDs()
				sc.HasPod(podUID)
				sc.GetPod(podUID)
				sc.GetContainerResources(podUID, "container1")
				sc.GetPodLevelResources(podUID)
				sc.GetEmptyDirVolumeLimit(podUID, "volume1")
			}
		}()
	}
	writers.Wait()
	close(stop)
	readers.Wait()

	// Whichever write came last, the checkpoint has the final state.
	restored, err := NewStateCheckpoint(logger, testDir, testCheckpoint)
	require.NoError(t, err)
	expected, actual := getPodMap(sc), getPodMap(restored)
	require.Len(t, expected, pods)
	verifyPodResourceAllocation(t, &expected, &actual, "the checkpoint does not have the latest state")
}
