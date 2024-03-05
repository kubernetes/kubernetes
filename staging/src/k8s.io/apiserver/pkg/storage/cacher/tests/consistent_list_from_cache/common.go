package consistent_list_from_cache

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher"
	cachertests "k8s.io/apiserver/pkg/storage/cacher/tests"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/pointer"
)

func RunFeatureGateTest(t *testing.T, expectEnabled bool) {
	t.Parallel()
	time.Sleep(time.Second)
	gotEnabled := utilfeature.DefaultFeatureGate.Enabled(features.ConsistentListFromCache)
	if gotEnabled != expectEnabled {
		t.Errorf("unexpected feature gate state, got: %v, want: %v", gotEnabled, expectEnabled)
	}
}

func RunTestList(t *testing.T) {
	ctx, cacher, server, terminate := cachertests.TestSetupWithEtcdServer(t)
	t.Cleanup(terminate)
	storagetesting.RunTestList(ctx, t, cacher, cachertests.CompactStorage(cacher, server.V3Client), true)
}

type BypassTestCase struct {
	Opts         storage.ListOptions
	ExpectBypass bool
}

var CommonBypassTestCases = []BypassTestCase{
	{Opts: storage.ListOptions{ResourceVersion: "0"}, ExpectBypass: false},
	{Opts: storage.ListOptions{ResourceVersion: "1"}, ExpectBypass: false},

	{Opts: storage.ListOptions{ResourceVersion: "", Predicate: storage.SelectionPredicate{Continue: "a"}}, ExpectBypass: true},
	{Opts: storage.ListOptions{ResourceVersion: "0", Predicate: storage.SelectionPredicate{Continue: "a"}}, ExpectBypass: true},
	{Opts: storage.ListOptions{ResourceVersion: "1", Predicate: storage.SelectionPredicate{Continue: "a"}}, ExpectBypass: true},

	{Opts: storage.ListOptions{ResourceVersion: "", Predicate: storage.SelectionPredicate{Limit: 500}}, ExpectBypass: true},
	{Opts: storage.ListOptions{ResourceVersion: "0", Predicate: storage.SelectionPredicate{Limit: 500}}, ExpectBypass: false},
	{Opts: storage.ListOptions{ResourceVersion: "1", Predicate: storage.SelectionPredicate{Limit: 500}}, ExpectBypass: true},

	{Opts: storage.ListOptions{ResourceVersion: "", ResourceVersionMatch: metav1.ResourceVersionMatchExact}, ExpectBypass: true},
	{Opts: storage.ListOptions{ResourceVersion: "0", ResourceVersionMatch: metav1.ResourceVersionMatchExact}, ExpectBypass: true},
	{Opts: storage.ListOptions{ResourceVersion: "1", ResourceVersionMatch: metav1.ResourceVersionMatchExact}, ExpectBypass: true},
}

func RunGetListCacheBypassTest(t *testing.T, options storage.ListOptions, expectBypass bool) {
	errDummy := errors.New("dummy")
	backingStorage := &cacher.DummyStorage{}
	cacher, err := cachertests.NewTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	result := &example.PodList{}

	// Wait until cacher is initialized.
	if err := cacher.WaitReady(context.Background()); err != nil {
		t.Fatalf("unexpected error waiting for the cache to be ready")
	}
	// Inject error to underlying layer and check if cacher is not bypassed.
	backingStorage.GetListFn = func(_ context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
		currentResourceVersion := "42"
		switch {
		// request made by getCurrentResourceVersionFromStorage by checking Limit
		case key == cacher.ResourcePrefix():
			podList := listObj.(*example.PodList)
			podList.ResourceVersion = currentResourceVersion
			return nil
		// request made by storage.GetList with revision from original request or
		// returned by getCurrentResourceVersionFromStorage
		case opts.ResourceVersion == options.ResourceVersion || opts.ResourceVersion == currentResourceVersion:
			return errDummy
		default:
			t.Fatalf("Unexpected request %+v", opts)
			return nil
		}
	}
	err = cacher.GetList(context.TODO(), "pods/ns", options, result)
	if err != nil && err != errDummy {
		t.Fatalf("Unexpected error for List request with options: %v, err: %v", options, err)
	}
	gotBypass := err == errDummy
	if gotBypass != expectBypass {
		t.Errorf("Unexpected bypass result for List request with options %+v, bypass expected: %v, got: %v", options, expectBypass, gotBypass)
	}
}

func RunWaitUntilFreshAndListFromCacheTest(t *testing.T) {
	ctx := context.Background()
	store := cacher.NewTestWatchCache(3, &cache.Indexers{})
	defer store.Stop()
	// In background, update the store.
	go func() {
		store.Add(cacher.MakeTestPod("pod1", 2))
		store.BookmarkRevision <- 3
	}()

	// list from future revision. Requires watch cache to request bookmark to get it.
	list, resourceVersion, indexUsed, err := store.WaitUntilFreshAndList(ctx, 3, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resourceVersion != 3 {
		t.Errorf("unexpected resourceVersion: %v, expected: 6", resourceVersion)
	}
	if len(list) != 1 {
		t.Errorf("unexpected list returned: %#v", list)
	}
	if indexUsed != "" {
		t.Errorf("Used index %q but expected none to be used", indexUsed)
	}
}

func RunWaitUntilFreshAndListTimeoutTest(t *testing.T) {
	ctx := context.Background()
	store := cacher.NewTestWatchCache(3, &cache.Indexers{})
	defer store.Stop()
	fc := store.FakeClock()

	// In background, step clock after the below call starts the timer.
	go func() {
		for !fc.HasWaiters() {
			time.Sleep(time.Millisecond)
		}
		store.Add(cacher.MakeTestPod("foo", 2))
		store.BookmarkRevision <- 3
		fc.Step(cacher.BlockTimeout)

		// Add an object to make sure the test would
		// eventually fail instead of just waiting
		// forever.
		time.Sleep(30 * time.Second)
		store.Add(cacher.MakeTestPod("bar", 4))
	}()

	_, _, _, err := store.WaitUntilFreshAndList(ctx, 4, nil)
	if !apierrors.IsTimeout(err) {
		t.Errorf("expected timeout error but got: %v", err)
	}
	if !storage.IsTooLargeResourceVersion(err) {
		t.Errorf("expected 'Too large resource version' cause in error but got: %v", err)
	}
}

func RunWaitUntilWatchCacheFreshAndForceAllEventsTest(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)()

	scenarios := []struct {
		name               string
		opts               storage.ListOptions
		backingStorage     *cacher.DummyStorage
		verifyBackingStore func(t *testing.T, s *cacher.DummyStorage)
	}{
		{
			name: "allowWatchBookmarks=true, sendInitialEvents=true, RV=105",
			opts: storage.ListOptions{
				Predicate: func() storage.SelectionPredicate {
					p := storage.Everything
					p.AllowWatchBookmarks = true
					return p
				}(),
				SendInitialEvents: pointer.Bool(true),
				ResourceVersion:   "105",
			},
			verifyBackingStore: func(t *testing.T, s *cacher.DummyStorage) {
				require.NotEqual(t, 0, s.RequestWatchProgressCounter, "expected store.RequestWatchProgressCounter to be > 0. It looks like watch progress wasn't requested!")
			},
		},

		{
			name: "legacy: allowWatchBookmarks=false, sendInitialEvents=true, RV=unset",
			opts: storage.ListOptions{
				Predicate: func() storage.SelectionPredicate {
					p := storage.Everything
					p.AllowWatchBookmarks = false
					return p
				}(),
				SendInitialEvents: pointer.Bool(true),
			},
			backingStorage: func() *cacher.DummyStorage {
				hasBeenPrimed := false
				s := &cacher.DummyStorage{}
				s.GetListFn = func(_ context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
					listAccessor, err := meta.ListAccessor(listObj)
					if err != nil {
						return err
					}
					// the first call to this function
					// primes the cacher
					if !hasBeenPrimed {
						listAccessor.SetResourceVersion("100")
						hasBeenPrimed = true
						return nil
					}
					listAccessor.SetResourceVersion("105")
					return nil
				}
				return s
			}(),
			verifyBackingStore: func(t *testing.T, s *cacher.DummyStorage) {
				require.NotEqual(t, 0, s.GetRequestWatchProgressCounter(), "expected store.RequestWatchProgressCounter to be > 0. It looks like watch progress wasn't requested!")
			},
		},

		{
			name: "allowWatchBookmarks=true, sendInitialEvents=true, RV=unset",
			opts: storage.ListOptions{
				Predicate: func() storage.SelectionPredicate {
					p := storage.Everything
					p.AllowWatchBookmarks = true
					return p
				}(),
				SendInitialEvents: pointer.Bool(true),
			},
			backingStorage: func() *cacher.DummyStorage {
				hasBeenPrimed := false
				s := &cacher.DummyStorage{}
				s.GetListFn = func(_ context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
					listAccessor, err := meta.ListAccessor(listObj)
					if err != nil {
						return err
					}
					// the first call to this function
					// primes the cacher
					if !hasBeenPrimed {
						listAccessor.SetResourceVersion("100")
						hasBeenPrimed = true
						return nil
					}
					listAccessor.SetResourceVersion("105")
					return nil
				}
				return s
			}(),
			verifyBackingStore: func(t *testing.T, s *cacher.DummyStorage) {
				require.NotEqual(t, 0, s.GetRequestWatchProgressCounter(), "expected store.RequestWatchProgressCounter to be > 0. It looks like watch progress wasn't requested!")
			},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			var backingStorage *cacher.DummyStorage
			if scenario.backingStorage != nil {
				backingStorage = scenario.backingStorage
			} else {
				backingStorage = &cacher.DummyStorage{}
			}
			c, err := cachertests.NewTestCacher(backingStorage)
			if err != nil {
				t.Fatalf("Couldn't create cacher: %v", err)
			}
			defer c.Stop()
			if err := c.WaitReady(context.Background()); err != nil {
				t.Fatalf("unexpected error waiting for the cache to be ready")
			}

			w, err := c.Watch(context.Background(), "pods/ns", scenario.opts)
			require.NoError(t, err, "failed to create watch: %v")
			defer w.Stop()
			var expectedErr *apierrors.StatusError
			if !errors.As(storage.NewTooLargeResourceVersionError(105, 100, cacher.ResourceVersionTooHighRetrySeconds), &expectedErr) {
				t.Fatalf("Unable to convert NewTooLargeResourceVersionError to apierrors.StatusError")
			}
			storagetesting.VerifyEvents(t, w, []watch.Event{
				{
					Type: watch.Error,
					Object: &metav1.Status{
						Status:  metav1.StatusFailure,
						Message: expectedErr.Error(),
						Details: expectedErr.ErrStatus.Details,
						Reason:  metav1.StatusReasonTimeout,
						Code:    504,
					},
				},
			}, true)

			go func(t *testing.T) {
				err := c.AddObject(cacher.MakeTestPodDetails("pod1", 105, "node1", map[string]string{"label": "value1"}))
				require.NoError(t, err, "failed adding a pod to the watchCache")
			}(t)
			w, err = c.Watch(context.Background(), "pods/ns", scenario.opts)
			require.NoError(t, err, "failed to create watch: %v")
			defer w.Stop()
			storagetesting.VerifyEvents(t, w, []watch.Event{
				{
					Type:   watch.Added,
					Object: cacher.MakeTestPodDetails("pod1", 105, "node1", map[string]string{"label": "value1"}),
				},
			}, true)
			if scenario.verifyBackingStore != nil {
				scenario.verifyBackingStore(t, backingStorage)
			}
		})
	}
}
