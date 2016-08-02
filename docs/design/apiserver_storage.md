# New Storage Interface

## Motivation

The current storage interface was evolved from the legacy etcd2 implementation. It has a few issues that prevents us from further optimizing the storage path efficiency while keeping the code maintainable and readable.

We have several goals for the new storage design. Once we reach these goals, the storage layer should be a completely decoupled component from other core logic parts of Kubernetes. That way we can make it easier to maintain without affecting any external code.

This list of goals we are trying to achieve with this proposal:

- maintainability: The storage code should be straightforward and easy to understand. The storage layer should have very clear responsibility boundary. It should not require huge amount of effort for people to understand other parts of Kubernetes in order to start coding and fixing issues in storage layer.

- testability: The storage code should be >90% (now it is ~70%) test covered. Storage developers should not rely on external packages to test the functionality of the storage layer. Users should feel confident in the correctness of the implementation as long as the unit/integration tests pass.

- clear API: The storage API should be minimalistic and with clear guarantees.  Users of storage pkg should see the guarantees immediately from the methods naming and related docs. For example, we should have guarantees between Read and Write. Should the user expect to read the value at least as new as the previous local write? Or should the user expect to read the value at least as new as the current global write?

- CPU and memory efficient: The storage layer should be designed to support object indexing and watch indexing. Storage layer should be designed to hide all storage related optimization details. The storage should be the single place to optimize server storage. If an index is built, users should expect to select objects in reasonable amount of time (for example < O(logN) for list). To optimize caching size of pod objects, we should do all the work in storage system without touching the pod object defined at API layer.

- easy to troubleshoot: Users should be able to identify easily if the storage layer is the current bottleneck of their cluster and why. We should expose enough metrics and write understandable docs to tell users how to do the troubleshooting.

- easy to support multiple generic storage backends: The storage layer should have a clear separation between operations on Kubernetes objects and generic key-value pairs. To implement a new key-value or database backend, developers should not have to write duplicated code to handle Kubernetes logic. Ideally, they do not even need to know anything about Kubernetes at all.

- backward compatible: Any changes should not introduce backward compatibility issues across at least last two minor leases from the end-user perspective. Using the improved storage system with existing etcd3 backend should just work.

## Direction

We could evolve the current storage layer towards the goal we listed above. It is doable, but it might involve significant interface and implement changes. Also it requires continuous changes of the consumers of the current storage API. During this process, we also need to ensure everything works perfectly. There is no alpha, or beta since we will be changing the single critical path. This is risky, time consuming, and hard to revert.

The solution we proposed here is to start a storage layer from scratch. As Daniel Smith [mentioned](https://github.com/kubernetes/kubernetes/pull/28508#issuecomment-232508836)

> Just to (probably unnecessarily) reiterate my position, I think we're going to need side-by-side implementations of both old and new storage styles, so we shouldn't end up locked into whatever we do now.

We also believe we should start a side-by-side storage implementation to reach the goal we listed. That way, we won’t be affected by any existing issues, and it gives us the opportunities to rethink the design from the ground up, to achieve the goals more clearly. The only risk is that the effort will be completely wasted, if we cannot reach the goals.

## Proposed Storage API

```go

// Storage stores and persists objects.
// Storage provides read-write access to the objects based on string keys.
type Storage interface {
	// Wait waits until the storage has seen any version >= given version.
	Wait(ctx context.Context, version uint64)

	// Put puts an object with a key based on the given conditions.
	// See docs on Conditions for more details.
	Put(ctx context.Context, key string, obj runtime.Object, conditions *Conditions) (cur runtime.Object, err error)

	// Delete a key and its object based on the given conditions.
	// See docs on Conditions for more details.
	Delete(ctx context.Context, key string, conditions *Conditions) (old runtime.Object, err error)

	// Get gets the most recent version of a key.
	// If no object exists on the key, it will return not found error.
	// If version > 0, it will get the current state of the key at the time the given version is committed.
	Get(ctx context.Context, key string, version uint64) (cur runtime.Object, err error)

	// List lists all objects that has given prefix and satisfies selectors.
	// If version > 0, same as Get().
	List(ctx context.Context, prefix string, version uint64, ss ...Selector) (objects []runtime.Object, globalRev uint64, err error)

	// WatchPrefix watches a prefix after given version. If version is 0, we will watch from current state.
	// It returns notifications of any keys that has given prefix.
	// Given selectors, it returns events that contained object of interest, either of current and previous.
	// If there is any problem establishing the watch channel, it will return error.
	// After channel is established, any error that happened will be returned from WatchChan
	// immediately before it's closed.
	WatchPrefix(ctx context.Context, prefix string, version uint64, ss ...Selector) (WatchChan, error)

	// AddIndex adds a new index.
	// An index indicates a field(s) of a type of an object that would be queried or watched frequently and
	// thus require better efficiency and performance.
	// If an index exists for a field, complexity of a query on it should be no more than O(logN)
	// where N is the total number of objects for List or watchers for watch event delivery.
	// See FieldValueGetFunc for more details on field value retrieving.
	AddIndex(indexName, field string, g FieldValueGetFunc) error
	// DeleteIndex deletes an index.
	DeleteIndex(indexName string)
}


// Conditions is used in do atomic conditional operations.
// If compare failed, corresponding method should return storage version conflict error.
type Conditions struct {
	// PrevVersion is used to compare with existing object’s version.
	// Compare would only succeed if versions were matched.
	// Note that PrevVersion=0 means no previous object existed.
	PrevVersion unit64
}

// FieldValueGetFunc returns the value in the object corresponding to given field.
// field is uniquely indicating a field of a type of object.
// It returns (value, true) if field exists in the object. Otherwise it returns ("", false).
type FieldValueGetFunc func(field string, obj runtime.Object) (string, bool)

// Selector is used to express one selection predicate.
// Op is the operator relating corresponding field's value to given values.
// See FieldValueGetFunc for details on Field and FVGetFunc.
type Selector struct {
	Op        Operator
	Field     string
	Values    []string
	FVGetFunc FieldValueGetFunc
}

type WatchChan <-chan WatchResponse

type WatchResponse struct {
	Type       EventType
	Object     runtime.Object
	PrevObject runtime.Object
	Err        error
}
```

We made the following changes based on the current Storage.Interface:

#### 1. More index friendly

The object index is used to find objects with specific attributes (e.g. labels, fields) quickly. For example, we have a redis controller that needs to find pods with labels “app:redis” and “role:slave”. With indexes, storage doesn’t have to scan over all pods, but only get pods with those labels from indexes.

The watcher index is used to find watchers similarly. For example, on receiving an event containing object of labels “app:redis” and “role:slave”, storage doesn’t have to scan over all watchers, but only get those watching on these labels.

In order to make use of indexes, we need select predicate to tell us three things: index key (a field or a label), operator (greater, less, or equal), and value (string, int, etc.). The original design of label and field has the three elements, e.g. “name=nginx”, “spec.nodename=A”. Although we had the correct design on the user side, somehow we lost it on server side in storage layer. The current storage layer can only get a filter or matching func that hides too many details and removes the ability to use indexing. Recent effort like MatcherIndex is not an indexing solution either, and making the matching and filter logic more convoluted. To fix this, we are proposing to have the labels and fields parsed as storage selectors -- a new design that provides the three elements, and pass them down to storage layer.

Indexing is useless if we cannot handle generic use cases. One of the design goals is to support generic indexing. To achieve this, we are introducing a mechanism to extract field values from generic k8s objects. Because each k8s resource has its own type of schema, we are introducing a flexible object parsing function: FieldValueGetFunc. Each resource registry defines its own FieldValueGetFunc to parse the field of an object and return corresponding value.

Now that we have the three basic elements to use the indexing, and also the method to build generic indexing, it’s easy to know how to index k8s objects. To look further, the above changes also enabled us to index watchers, enabling us to find out related watchers for each event quickly by using index.

#### 2. Remove unnecessary helper functions from the interface itself

We remove GuaranteedUpdate() from the Interface. It only uses the existing APIs and does not need to be part of the core storage API. One of the goals is to make storage API minimum and isolated to be more maintainable. So we decide to move it out of the Storage API and implement the same functionality as helper function.

``` go
GuaranteedUpdate(Storage, ...)
```

We have split List into List and Wait. The previous List function will block until the result is as fresh as given version. This can be split into two functions, where one function does only one thing. The previous List behavior could be implemented in the following way:

```go
func WaitAndList(storage, version, ...) ... {
	storage.Wait(version)
	return storage.List(...)
}
```

#### 3. Simplify conditional check

We replace the UID filed in the precondition with version. UID and other fields all depends on the version of the objects. By comparing the versions between objects, we effectively compare its UID. We might add more preconditions if needed.



## What we have done to prove the proposed API works

We have implemented a POC version of the new storage

- Object selection is MUCH faster when index is built.
- Watch deliver is MUCH faster when index is built.
- The complexity reduced from O(N) to O(logN) with naive B-tree based index, or even O(1) with map-based index.

We made all registries working with the new side-by-side storage implementation

- The additional code to support the new storage API is small and easy to maintain
- Successfully run Kubernetes cluster with the new storage implementation
- Index working end to end from API down to storage layer
