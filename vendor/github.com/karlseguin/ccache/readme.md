# CCache
CCache is an LRU Cache, written in Go, focused on supporting high concurrency.

Lock contention on the list is reduced by:

* Introducing a window which limits the frequency that an item can get promoted
* Using a buffered channel to queue promotions for a single worker
* Garbage collecting within the same thread as the worker

## Setup

First, download the project:

    go get github.com/karlseguin/ccache

## Configuration
Next, import and create a `Cache` instance:


```go
import (
  "github.com/karlseguin/ccache"
)

var cache = ccache.New(ccache.Configure())
```

`Configure` exposes a chainable API:

```go
var cache = ccache.New(ccache.Configure().MaxSize(1000).ItemsToPrune(100))
```

The most likely configuration options to tweak are:

* `MaxSize(int)` - the maximum number size  to store in the cache (default: 5000)
* `GetsPerPromote(int)` - the number of times an item is fetched before we promote it. For large caches with long TTLs, it normally isn't necessary to promote an item after every fetch (default: 3)
* `ItemsToPrune(int)` - the number of items to prune when we hit `MaxSize`. Freeing up more than 1 slot at a time improved performance (default: 500)

Configurations that change the internals of the cache, which aren't as likely to need tweaking:

* `Buckets` - ccache shards its internal map to provide a greater amount of concurrency. Must be a power of 2 (default: 16).
* `PromoteBuffer(int)` - the size of the buffer to use to queue promotions (default: 1024)
* `DeleteBuffer(int)` the size of the buffer to use to queue deletions (default: 1024)

## Usage

Once the cache is setup, you can  `Get`, `Set` and `Delete` items from it. A `Get` returns an `*Item`:

### Get
```go
item := cache.Get("user:4")
if item == nil {
  //handle
} else {
  user := item.Value().(*User)
}
```
The returned `*Item` exposes a number of methods:

* `Value() interface{}` - the value cached
* `Expired() bool` - whether the item is expired or not
* `TTL() time.Duration` - the duration before the item expires (will be a negative value for expired items)
* `Expires() time.Time` - the time the item will expire

By returning expired items, CCache lets you decide if you want to serve stale content or not. For example, you might decide to serve up slightly stale content (< 30 seconds old) while re-fetching newer data in the background. You might also decide to serve up infinitely stale content if you're unable to get new data from your source.

### Set
`Set` expects the key, value and ttl:

```go
cache.Set("user:4", user, time.Minute * 10)
```

### Fetch
There's also a `Fetch` which mixes a `Get` and a `Set`:

```go
item, err := cache.Fetch("user:4", time.Minute * 10, func() (interface{}, error) {
  //code to fetch the data incase of a miss
  //should return the data to cache and the error, if any
})
```

### Delete
`Delete` expects the key to delete. It's ok to call `Delete` on a non-existant key:

```go
cache.Delete("user:4")
```

### Extend
The life of an item can be changed via the `Extend` method. This will change the expiry of the item by the specified duration relative to the current time.

### Replace
The value of an item can be updated to a new value without renewing the item's TTL or it's position in the LRU:

```go
cache.Replace("user:4", user)
```

`Replace` returns true if the item existed (and thus was replaced). In the case where the key was not in the cache, the value *is not* inserted and false is returned.

### Stop
The cache's background worker can be stopped by calling `Stop`. Once `Stop` is called
the cache should not be used (calls are likely to panic). Stop must be called in order to allow the garbage collector to reap the cache.

## Tracking
CCache supports a special tracking mode which is meant to be used in conjunction with other pieces of your code that maintains a long-lived reference to data.

When you configure your cache with `Track()`:

```go
cache = ccache.New(ccache.Configure().Track())
```

The items retrieved via `TrackingGet` will not be eligible for purge until `Release` is called on them:

```go
item := cache.TrackingGet("user:4")
user := item.Value()   //will be nil if "user:4" didn't exist in the cache
item.Release()  //can be called even if item.Value() returned nil
```

In practive, `Release` wouldn't be called until later, at some other place in your code.

There's a couple reason to use the tracking mode if other parts of your code also hold references to objects. First, if you're already going to hold a reference to these objects, there's really no reason not to have them in the cache - the memory is used up anyways.

More important, it helps ensure that you're code returns consistent data. With tracking, "user:4" might be purged, and a subsequent `Fetch` would reload the data. This can result in different versions of "user:4" being returned by different parts of your system.

## LayeredCache

CCache's `LayeredCache` stores and retrieves values by both a primary and secondary key. Deletion can happen against either the primary and secondary key, or the primary key only (removing all values that share the same primary key).

`LayeredCache` is useful for HTTP caching, when you want to purge all variations of a request.

`LayeredCache` takes the same configuration object as the main cache, exposes the same optional tracking capabilities, but exposes a slightly different API:

```go
cache := ccache.Layered(ccache.Configure())

cache.Set("/users/goku", "type:json", "{value_to_cache}", time.Minute * 5)
cache.Set("/users/goku", "type:xml", "<value_to_cache>", time.Minute * 5)

json := cache.Get("/users/goku", "type:json")
xml := cache.Get("/users/goku", "type:xml")

cache.Delete("/users/goku", "type:json")
cache.Delete("/users/goku", "type:xml")
// OR
cache.DeleteAll("/users/goku")
```

# SecondaryCache

In some cases, when using a `LayeredCache`, it may be desirable to always be acting on the secondary portion of the cache entry. This could be the case where the primary key is used as a key elsewhere in your code. The `SecondaryCache` is retrieved with:

```go
cache := ccache.Layered(ccache.Configure())
sCache := cache.GetOrCreateSecondaryCache("/users/goku")
sCache.Set("type:json", "{value_to_cache}", time.Minute * 5)
```

The semantics for interacting with the `SecondaryCache` are exactly the same as for a regular `Cache`. However, one difference is that `Get` will not return nil, but will return an empty 'cache' for a non-existent primary key.

## Size
By default, items added to a cache have a size of 1. This means that if you configure `MaxSize(10000)`, you'll be able to store 10000 items in the cache.

However, if the values you set into the cache have a method `Size() int64`, this size will be used. Note that ccache has an overhead of ~350 bytes per entry, which isn't taken into account. In other words, given a filled up cache, with `MaxSize(4096000)` and items that return a `Size() int64` of 2048, we can expect to find 2000 items (4096000/2048) taking a total space of 4796000 bytes.

## Want Something Simpler?
For a simpler cache, checkout out [rcache](https://github.com/karlseguin/rcache)
