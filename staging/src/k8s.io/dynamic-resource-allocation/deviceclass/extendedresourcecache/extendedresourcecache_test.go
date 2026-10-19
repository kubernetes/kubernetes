/*
Copyright 2025 The Kubernetes Authors.

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

package extendedresourcecache

import (
	"testing"
	"time"

	g "github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clientcache "k8s.io/client-go/tools/cache"
	"k8s.io/ktesting"
	"k8s.io/utils/ptr"
)

type deviceClassResolver interface {
	GetDeviceClass(resourceName v1.ResourceName) *resourceapi.DeviceClass
}

func TestNil(t *testing.T) { testNil(ktesting.Init(t)) }
func testNil(tCtx ktesting.TContext) {
	var cache *ExtendedResourceCache
	var resolver deviceClassResolver = cache
	tCtx.Assert(resolver.GetDeviceClass("example.com/gpu")).To(g.BeNil(), "nil class from a nil instance")
}

func TestHandlers(t *testing.T) { testHandlers(ktesting.Init(t)) }
func testHandlers(tCtx ktesting.TContext) {
	var numAdd, numUpdate, numDelete int

	resourceName := v1.ResourceName("example.com/gpu")
	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: (*string)(&resourceName),
		},
	}
	updatedClass := class.DeepCopy()
	updatedClass.Spec.ExtendedResourceName = nil

	firstHandler := &clientcache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			tCtx.Assert(obj).To(g.BeIdenticalTo(class), "first handler expected added object")
			numAdd++
			tCtx.Assert(numAdd).To(g.Equal(1), "first handler expected Add to be called first")
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			tCtx.Assert(oldObj).To(g.BeIdenticalTo(class), "first handler expected old object")
			tCtx.Assert(newObj).To(g.BeIdenticalTo(updatedClass), "first handler expected new object")
			numUpdate++
			tCtx.Assert(numUpdate).To(g.Equal(1), "first handler expected Update to be called first")
		},
		DeleteFunc: func(obj interface{}) {
			tCtx.Assert(obj).To(g.BeIdenticalTo(updatedClass), "first handler expected deleted object")
			numDelete++
			tCtx.Assert(numDelete).To(g.Equal(1), "first handler expected Delete to be called first")
		},
	}
	erCache := NewExtendedResourceCache(tCtx.Logger(), firstHandler)
	secondHandler := &clientcache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			tCtx.Assert(obj).To(g.BeIdenticalTo(class), "second handler expected added object")
			numAdd++
			tCtx.Assert(numAdd).To(g.Equal(2), "second handler expected Add to be called last")
			tCtx.Assert(erCache.GetDeviceClass(resourceName)).To(g.HaveValue(g.HaveField("Name", class.Name)), "device class visible to second handler's AddFunc")
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			tCtx.Assert(oldObj).To(g.BeIdenticalTo(class), "second handler expected old object")
			tCtx.Assert(newObj).To(g.BeIdenticalTo(updatedClass), "second handler expected new object")
			numUpdate++
			tCtx.Assert(numUpdate).To(g.Equal(2), "second handler expected Update to be called last")
			tCtx.Assert(erCache.GetDeviceClass(resourceName)).To(g.BeNil(), "device class visible to second handler's UpdateFunc")
		},
		DeleteFunc: func(obj interface{}) {
			tCtx.Assert(obj).To(g.BeIdenticalTo(updatedClass), "second handler expected deleted object")
			numDelete++
			tCtx.Assert(numDelete).To(g.Equal(2), "second handler expected Delete to be called last")
			tCtx.Assert(erCache.GetDeviceClass(resourceName)).To(g.BeNil(), "device class visible to second handler's DeleteFunc")
		},
	}
	erCache.AddEventHandler(secondHandler)

	erCache.OnAdd(class, false)
	erCache.OnUpdate(class, updatedClass)
	erCache.OnDelete(updatedClass)
}

func TestExtendedResourceCache(t *testing.T) {
	ktesting.Init(t).SyncTest("", testExtendedResourceCache)
}
func testExtendedResourceCache(tCtx ktesting.TContext) {
	tCtx, client, cache := setup(tCtx)

	// Test with a device class that has an explicit extended resource name
	now := time.Now()
	deviceClass1 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class",
			CreationTimestamp: metav1.Time{
				Time: now,
			},
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}

	// Test with a device class that uses the default mapping
	deviceClass2 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "fpga-class",
		},
		Spec: resourceapi.DeviceClassSpec{
			// No explicit extended resource name
		},
	}
	deviceClass3 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class-3",
			CreationTimestamp: metav1.Time{
				Time: now.Add(-24 * time.Hour),
			},
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}
	deviceClass4 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class-4",
			CreationTimestamp: metav1.Time{
				Time: now.Add(time.Hour),
			},
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}
	deviceClass0 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class-0",
			CreationTimestamp: metav1.Time{
				Time: now.Add(time.Hour),
			},
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}

	// Test adding device classes
	_, err := client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass1, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create device class")
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass2, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create device class")
	tCtx.Wait()

	// Verify explicit mapping
	deviceClass := cache.GetDeviceClass("example.com/gpu")
	tCtx.Assert(deviceClass).To(g.HaveField("Name", "gpu-class"), "device class for 'example.com/gpu'")

	// Verify default mapping
	defaultResourceName := v1.ResourceName("deviceclass.resource.kubernetes.io/fpga-class")
	deviceClass = cache.GetDeviceClass(defaultResourceName)
	tCtx.Assert(deviceClass).To(g.HaveField("Name", "fpga-class"), "device class for %q", defaultResourceName)

	// Verify both device classes have default mappings
	deviceClass = cache.GetDeviceClass("deviceclass.resource.kubernetes.io/gpu-class")
	tCtx.Assert(deviceClass).To(g.HaveField("Name", "gpu-class"), "default mapping for gpu-class")

	// deviceClass3 is older than deviceClass1, hence it won't replace deviceClass1
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass3, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create device class")
	tCtx.Wait()

	// should keep deviceClass1, since it is newer than deviceClass3
	deviceClass = cache.GetDeviceClass("example.com/gpu")
	tCtx.Assert(deviceClass).To(g.HaveField("Name", "gpu-class"), "device class for 'example.com/gpu' after adding an older class")

	// deviceClass4 is newer than deviceClass1, hence it will replace deviceClass1
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass4, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create device class")
	tCtx.Wait()

	// deviceClass4 replaces deviceClass1, since it is newer with the same example.com/gpu extended resource name
	deviceClass = cache.GetDeviceClass("example.com/gpu")
	tCtx.Assert(deviceClass).To(g.HaveField("Name", "gpu-class-4"), "device class for 'example.com/gpu' after adding a newer class")

	// deviceClass0 is created at the same time as deviceClass4, but its name is alphabetically ordered earlier,
	//  hence it will replace deviceClass4
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass0, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create device class")
	tCtx.Wait()

	// deviceClass0 replaces deviceClass4, it is created at the same time as deviceClass4, but its name is
	// alphabetically ordered earlier
	deviceClass = cache.GetDeviceClass("example.com/gpu")
	tCtx.Assert(deviceClass).To(g.HaveField("Name", "gpu-class-0"), "device class for 'example.com/gpu' after adding a lexicographically earlier class with equal timestamp")

	// Test modifying a device class
	deviceClass0Modified := deviceClass0.DeepCopy()
	deviceClass0Modified.Spec.ExtendedResourceName = ptr.To("test.com/gpu")
	_, err = client.ResourceV1().DeviceClasses().Update(tCtx, deviceClass0Modified, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "update device class")
	tCtx.Wait()

	// Should have the new mapping
	deviceClass = cache.GetDeviceClass("test.com/gpu")
	tCtx.Assert(deviceClass).To(g.HaveField("Name", "gpu-class-0"), "device class for 'test.com/gpu' after modification")
	// Should not have the old mapping for example.com/gpu
	deviceClass = cache.GetDeviceClass("example.com/gpu")
	tCtx.Assert(deviceClass).To(g.HaveField("Name", "gpu-class-4"), "'example.com/gpu' promoted to 'gpu-class-4' after modification")

	// Test deleting a device class
	err = client.ResourceV1().DeviceClasses().Delete(tCtx, deviceClass0.Name, metav1.DeleteOptions{})
	tCtx.ExpectNoError(err, "delete device class")
	tCtx.Wait()

	tCtx.Assert(cache.GetDeviceClass("test.com/gpu")).To(g.BeNil(), "'test.com/gpu' removed after deleting device class")
	// Verify the default mapping is removed
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/gpu-class-0")).To(g.BeNil(), "'deviceclass.resource.kubernetes.io/gpu-class-0' removed after deleting device class")
}

func TestDeviceClassMapping(t *testing.T) { ktesting.Init(t).SyncTest("", testDeviceClassMapping) }
func testDeviceClassMapping(tCtx ktesting.TContext) {
	tCtx, client, cache := setup(tCtx)

	deviceClass1 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}

	deviceClass2 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "tpu-class",
		},
	}

	// Test adding device classes
	_, err := client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass1, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create device class")
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass2, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create device class")

	// Wait for background goroutines to handle the new classes.
	tCtx.Wait()
	tCtx.Assert(cache.GetExtendedResource("gpu-class")).To(g.Equal("example.com/gpu"), "extended resource for 'gpu-class'")
	tCtx.Assert(cache.GetExtendedResource("tpu-class")).To(g.BeEmpty(), "extended resource for 'tpu-class'")

	// Test updating device classes
	deviceClass1Modified := deviceClass1.DeepCopy()
	deviceClass1Modified.Spec.ExtendedResourceName = ptr.To("my.com/gpu")
	_, err = client.ResourceV1().DeviceClasses().Update(tCtx, deviceClass1Modified, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "update device class")

	tCtx.Wait()
	tCtx.Assert(cache.GetExtendedResource("gpu-class")).To(g.Equal("my.com/gpu"), "extended resource for 'gpu-class' after modification")

	// Test deleting device classes
	err = client.ResourceV1().DeviceClasses().Delete(tCtx, deviceClass1.Name, metav1.DeleteOptions{})
	tCtx.ExpectNoError(err, "delete device class")

	tCtx.Wait()
	tCtx.Assert(cache.GetExtendedResource("gpu-class")).To(g.BeEmpty(), "extended resource for 'gpu-class' after deletion")
}

func newDeviceClass(name, explicitName string, created time.Time) *resourceapi.DeviceClass {
	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			CreationTimestamp: metav1.Time{Time: created},
		},
	}
	if explicitName != "" {
		class.Spec.ExtendedResourceName = new(string)
		*class.Spec.ExtendedResourceName = explicitName
	}
	return class
}

func TestReadersCannotObservePartialUpdate(t *testing.T) {
	testReadersCannotObservePartialUpdate(ktesting.Init(t))
}
func testReadersCannotObservePartialUpdate(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	class := newDeviceClass("class-a", "example.com/gpu", time.Unix(100, 0))
	cache.OnAdd(class, false)

	renamed := class.DeepCopy()
	renamed.Spec.ExtendedResourceName = new(string)
	*renamed.Spec.ExtendedResourceName = "my.com/gpu"

	// Suspend the update right between the forward and reverse mapping
	// updates, then check that readers still observe a consistent state.
	entered := make(chan struct{})
	release := make(chan struct{})
	cache.testHook = func() {
		close(entered)
		<-release
	}

	updateDone := make(chan struct{})
	go func() {
		cache.OnUpdate(class, renamed)
		close(updateDone)
	}()

	<-entered
	// The forward mapping was updated already, the reverse mapping is not
	// yet. A reader starting now must not observe anything until the whole
	// event has been applied.
	read := make(chan string, 1)
	go func() {
		read <- cache.GetExtendedResource("class-a")
	}()
	select {
	case name := <-read:
		close(release)
		tCtx.Fatalf("reader observed the reverse mapping while the update was in flight: %q", name)
	case <-time.After(time.Second):
	}
	close(release)
	<-updateDone
	tCtx.Assert(<-read).To(g.Equal("my.com/gpu"), "expected the reverse mapping to be updated atomically with the forward mapping")
	tCtx.Assert(cache.GetDeviceClass("my.com/gpu")).To(g.BeIdenticalTo(renamed), "expected the new explicit mapping to be visible")
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.BeNil(), "expected the old explicit mapping to be removed")
}

func TestSameClassUpdateReplacesStaleObject(t *testing.T) {
	testSameClassUpdateReplacesStaleObject(ktesting.Init(t))
}
func testSameClassUpdateReplacesStaleObject(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	class := newDeviceClass("class-a", "example.com/gpu", time.Unix(100, 0))
	cache.OnAdd(class, false)

	// Update the class while keeping the same extended resource name and
	// creation timestamp. The cache must serve the freshly updated object,
	// not the stale one.
	updated := class.DeepCopy()
	updated.Spec.Config = []resourceapi.DeviceClassConfiguration{{}}
	cache.OnUpdate(class, updated)

	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(updated), "explicit mapping should point at the updated object")
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/class-a")).To(g.Equal(updated), "default mapping should point at the updated object")
	tCtx.Assert(cache.GetExtendedResource("class-a")).To(g.Equal("example.com/gpu"), "reverse mapping should be preserved")
}

func TestCollisionLoserKeepsImplicitMapping(t *testing.T) {
	testCollisionLoserKeepsImplicitMapping(ktesting.Init(t))
}
func testCollisionLoserKeepsImplicitMapping(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	winner := newDeviceClass("class-winner", "example.com/gpu", time.Unix(200, 0))
	loser := newDeviceClass("class-loser", "example.com/gpu", time.Unix(100, 0))
	cache.OnAdd(winner, false)
	cache.OnAdd(loser, false)

	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(winner), "expected the newer class to win the explicit mapping")
	// The loser stays reachable via its own unique implicit name, which
	// cannot collide with the explicit name of another class.
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/class-loser")).To(g.Equal(loser), "expected the loser's default mapping to be registered")
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/class-winner")).To(g.Equal(winner), "expected the winner's default mapping to be registered")
}

func TestCollisionLoserDeleteKeepsWinner(t *testing.T) {
	testCollisionLoserDeleteKeepsWinner(ktesting.Init(t))
}
func testCollisionLoserDeleteKeepsWinner(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	winner := newDeviceClass("class-winner", "example.com/gpu", time.Unix(200, 0))
	loser := newDeviceClass("class-loser", "example.com/gpu", time.Unix(100, 0))
	cache.OnAdd(winner, false)
	cache.OnAdd(loser, false)

	cache.OnDelete(loser)

	// Deleting the loser must not take down the winner's mapping for the
	// shared explicit name.
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(winner), "expected the winner to keep the explicit mapping")
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/class-loser")).To(g.BeNil(), "expected the loser's default mapping to be removed")
	tCtx.Assert(cache.GetExtendedResource("class-loser")).To(g.BeEmpty(), "expected the loser's reverse mapping to be removed")
}

func TestCollisionWinnerDeletePromotesRunnerUp(t *testing.T) {
	testCollisionWinnerDeletePromotesRunnerUp(ktesting.Init(t))
}
func testCollisionWinnerDeletePromotesRunnerUp(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	older := newDeviceClass("class-older", "example.com/gpu", time.Unix(100, 0))
	newer := newDeviceClass("class-newer", "example.com/gpu", time.Unix(200, 0))
	cache.OnAdd(older, false)
	cache.OnAdd(newer, false)

	// Deleting the winner must promote the runner-up.
	cache.OnDelete(newer)
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(older), "expected the runner-up to be promoted")

	cache.OnDelete(older)
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.BeNil(), "expected the explicit mapping to be removed once all candidates are gone")
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/class-newer")).To(g.BeNil(), "expected the winner's default mapping to be removed")
	tCtx.Assert(cache.GetExtendedResource("class-newer")).To(g.BeEmpty(), "expected the winner's reverse mapping to be removed")
}

func TestCollisionWinnerRenamePromotesRunnerUp(t *testing.T) {
	testCollisionWinnerRenamePromotesRunnerUp(ktesting.Init(t))
}
func testCollisionWinnerRenamePromotesRunnerUp(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	older := newDeviceClass("class-older", "example.com/gpu", time.Unix(100, 0))
	newer := newDeviceClass("class-newer", "example.com/gpu", time.Unix(200, 0))
	cache.OnAdd(older, false)
	cache.OnAdd(newer, false)

	renamed := newer.DeepCopy()
	renamed.Spec.ExtendedResourceName = new(string)
	*renamed.Spec.ExtendedResourceName = "new.example.com/gpu"
	cache.OnUpdate(newer, renamed)

	// Renaming the winner must promote the runner-up for the old name.
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(older), "expected the runner-up to be promoted after winner rename")
	tCtx.Assert(cache.GetDeviceClass("new.example.com/gpu")).To(g.Equal(renamed), "expected the renamed class to own its new name")
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/class-newer")).To(g.Equal(renamed), "expected the renamed class's default mapping to be updated")
	tCtx.Assert(cache.GetExtendedResource("class-newer")).To(g.Equal("new.example.com/gpu"), "expected the renamed class's reverse mapping to be updated")
}

func TestCollisionLoserRenameKeepsWinner(t *testing.T) {
	testCollisionLoserRenameKeepsWinner(ktesting.Init(t))
}
func testCollisionLoserRenameKeepsWinner(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	winner := newDeviceClass("class-winner", "example.com/gpu", time.Unix(200, 0))
	loser := newDeviceClass("class-loser", "example.com/gpu", time.Unix(100, 0))
	cache.OnAdd(winner, false)
	cache.OnAdd(loser, false)

	renamed := loser.DeepCopy()
	renamed.Spec.ExtendedResourceName = new(string)
	*renamed.Spec.ExtendedResourceName = "new.example.com/gpu"
	cache.OnUpdate(loser, renamed)

	// Renaming the loser must not take down the winner's mapping for the
	// old name, even though the loser used to declare it.
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(winner), "expected the winner to keep the explicit mapping")
	tCtx.Assert(cache.GetDeviceClass("new.example.com/gpu")).To(g.Equal(renamed), "expected the renamed class to own its new name")
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/class-loser")).To(g.Equal(renamed), "expected the renamed class's default mapping to be updated")
	tCtx.Assert(cache.GetExtendedResource("class-loser")).To(g.Equal("new.example.com/gpu"), "expected the renamed loser's reverse mapping to be updated")
}

func TestCollisionWinnerKeyOnlyTombstonePromotesRunnerUp(t *testing.T) {
	testCollisionWinnerKeyOnlyTombstonePromotesRunnerUp(ktesting.Init(t))
}
func testCollisionWinnerKeyOnlyTombstonePromotesRunnerUp(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	older := newDeviceClass("class-older", "example.com/gpu", time.Unix(100, 0))
	newer := newDeviceClass("class-newer", "example.com/gpu", time.Unix(200, 0))
	cache.OnAdd(older, false)
	cache.OnAdd(newer, false)

	// DeltaFIFO.Replace can emit a tombstone whose Obj is nil when the key
	// is no longer available from knownObjects. All mappings are keyed by
	// class name, so the key alone must suffice to remove the deleted class.
	cache.OnDelete(clientcache.DeletedFinalStateUnknown{Key: newer.Name, Obj: nil})

	// The runner-up must be promoted despite the missing object.
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(older), "expected the runner-up to be promoted")
	// The default mapping and the reverse mapping must be removed.
	tCtx.Assert(cache.GetDeviceClass("deviceclass.resource.kubernetes.io/class-newer")).To(g.BeNil(), "expected the deleted class's default mapping to be removed")
	tCtx.Assert(cache.GetExtendedResource("class-newer")).To(g.BeEmpty(), "expected the deleted class's reverse mapping to be removed")
}

func TestCollisionPromotedLoserIsFresh(t *testing.T) {
	testCollisionPromotedLoserIsFresh(ktesting.Init(t))
}
func testCollisionPromotedLoserIsFresh(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	winner := newDeviceClass("class-winner", "example.com/gpu", time.Unix(200, 0))
	loser := newDeviceClass("class-loser", "example.com/gpu", time.Unix(100, 0))
	cache.OnAdd(winner, false)
	cache.OnAdd(loser, false)

	// Update the loser while it is still the runner-up.
	updatedLoser := loser.DeepCopy()
	updatedLoser.Spec.Config = []resourceapi.DeviceClassConfiguration{{}}
	cache.OnUpdate(loser, updatedLoser)

	// Deleting the winner must promote the freshly updated runner-up, not a
	// stale copy of the loser.
	cache.OnDelete(winner)
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(updatedLoser), "expected the fresh runner-up to be promoted")
}

func TestCollisionEqualTimestampTieBreak(t *testing.T) {
	testCollisionEqualTimestampTieBreak(ktesting.Init(t))
}
func testCollisionEqualTimestampTieBreak(tCtx ktesting.TContext) {
	cache := NewExtendedResourceCache(tCtx.Logger())

	classA := newDeviceClass("class-a", "example.com/gpu", time.Unix(100, 0))
	classB := newDeviceClass("class-b", "example.com/gpu", time.Unix(100, 0))
	cache.OnAdd(classA, false)
	cache.OnAdd(classB, false)

	// Equal creation timestamps: the lexicographically first name wins.
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(classA), "expected the lexicographically first name to win the tie")

	renamed := classA.DeepCopy()
	renamed.Spec.ExtendedResourceName = new(string)
	*renamed.Spec.ExtendedResourceName = "new.example.com/gpu"
	cache.OnUpdate(classA, renamed)

	// Renaming the tie winner promotes the runner-up.
	tCtx.Assert(cache.GetDeviceClass("example.com/gpu")).To(g.Equal(classB), "expected the runner-up to be promoted after tie winner rename")
	tCtx.Assert(cache.GetDeviceClass("new.example.com/gpu")).To(g.Equal(renamed), "expected the renamed class to own its new name")
	tCtx.Assert(cache.GetExtendedResource("class-a")).To(g.Equal("new.example.com/gpu"), "expected the renamed class's reverse mapping to be updated")
}

func TestBetterDeviceClass(t *testing.T) { testBetterDeviceClass(ktesting.Init(t)) }
func testBetterDeviceClass(tCtx ktesting.TContext) {
	older := newDeviceClass("class-older", "example.com/gpu", time.Unix(100, 0))
	newer := newDeviceClass("class-newer", "example.com/gpu", time.Unix(200, 0))
	other := newDeviceClass("class-a", "example.com/gpu", time.Unix(100, 0))

	// Newer classes win over older ones.
	if betterDeviceClass(older, newer) {
		tCtx.Errorf("older class should lose")
	}
	if !betterDeviceClass(newer, older) {
		tCtx.Errorf("newer class should win")
	}
	// Equal creation timestamps: the lexicographically first name wins.
	if !betterDeviceClass(other, older) {
		tCtx.Errorf("lexicographically first name should win the tie")
	}
	if betterDeviceClass(older, other) {
		tCtx.Errorf("lexicographically later name should lose the tie")
	}
	// A class is never better than itself.
	if betterDeviceClass(older, older) {
		tCtx.Errorf("a class should not be better than itself")
	}
	// A nil class never wins, and never blocks a non-nil one.
	if betterDeviceClass(nil, older) {
		tCtx.Errorf("a nil class should lose")
	}
	if betterDeviceClass(nil, nil) {
		tCtx.Errorf("a nil class should lose against another nil class")
	}
	if !betterDeviceClass(older, nil) {
		tCtx.Errorf("a class should win against a nil incumbent")
	}
}

func setup(tCtx ktesting.TContext) (ktesting.TContext, *fake.Clientset, *ExtendedResourceCache) {
	tCtx = tCtx.WithCancel()

	client := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	ec := NewExtendedResourceCache(tCtx.Logger())
	handle, err := informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandler(ec)
	tCtx.ExpectNoError(err, "failed to add device class informer event handler")
	informerFactory.Start(tCtx.Done())
	tCtx.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCtx.Cancel("test is done")
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})
	informerFactory.WaitForCacheSync(tCtx.Done())
	clientcache.WaitForNamedCacheSyncWithContext(tCtx, handle.HasSynced)

	// fake.Clientset suffers from a race condition related to informers:
	// it does not implement resource version support in its Watch
	// implementation and instead assumes that watches are set up
	// before further changes are made.
	//
	// If a test waits for caches to be synced and then immediately
	// adds an object, that new object will never be seen by event handlers
	// if the race goes wrong and the Watch call hadn't completed yet
	// (can be triggered by adding a sleep before https://github.com/kubernetes/kubernetes/blob/b53b9fb5573323484af9a19cf3f5bfe80760abba/staging/src/k8s.io/client-go/tools/cache/reflector.go#L431).
	//
	// To work around that, we wait here for the goroutines which
	// are involved in setting up the watch *before* creating
	// DeviceClasses.
	tCtx.Wait()

	return tCtx, client, ec
}
