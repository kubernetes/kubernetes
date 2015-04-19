# bower-registry-client [![Build Status](https://travis-ci.org/bower/registry-client.png?branch=master)](https://travis-ci.org/bower/registry-client)

> Provides easy interaction with the Bower registry


## Install

```
$ npm install --save bower-registry-client
```


## Usage

```js
var RegistryClient = require('bower-registry-client');
var registry = new RegistryClient(options, logger);
```

The `logger` is optional and is expected to be an instance of the bower [logger](https://github.com/bower/logger).   
Available constructor options:

- `cache`: the cache folder to use for some operations; using null will disable persistent cache (defaults to bower registry cache folder)
- `registry.search`: an array of registry search endpoints (defaults to the Bower server)
- `registry.register`: the endpoint to use when registering packages (defaults to the Bower server)
- `registry.publish`: the endpoint to use when publishing packages (defaults to the Bower server)
- `ca.search`: an array of CA certificates for each registry.search (defaults to null).
- `ca.register`: the CA certificate for registry.register
- `ca.publish`: the CA certificate for registry.publish
- `proxy`: the proxy to use for http requests (defaults to null)
- `httpsProxy`: the proxy to use for https requests (defaults to null)
- `strictSsl`: whether or not to do SSL key validation when making requests via https (defaults to true).
- `userAgent`: the user agent to use for the requests (defaults to null)
- `timeout`: the timeout for the requests to finish (defaults to 60000)
- `force`: If set to true, cache will be bypassed and remotes will always be hit (defaults to false).
- `offline`: If set to true, only the cache will be used (defaults to false).


Note that `force` and `offline` are mutually exclusive.
The cache will speedup operations such as `list`, `lookup` and `search`.
Different operations may have different cache expiration times.


#### .lookup(name, callback)

Looks the registry for the package `name`,

```js
registry.lookup('jquery', function (err, entry) {
    if (err) {
        console.error(err.message);
        return;
    }

    // For now resp.type is always 'alias'
    console.log('type', entry.type);
    console.log('url', entry.url);
});
```

#### .register(name, url, callback)

Registers a package in the registry.

```js
registry.register('my-package', 'git://github.com/my-org/my-package.git', function (err, pkg) {
    if (err) {
        console.error(err.message);
        return;
    }

    console.log('name', pkg.name);
    console.log('url: ', pkg.url);
});
```

#### .search(str, callback)

Searches the registry.

```js
registry.search('jquery', function (err, results) {
    if (err) {
        console.error(err.message);
        return;
    }

    results.forEach(function (pkg) {
        console.log('name', pkg.name);
        console.log('url', pkg.url);
    });
});
```

#### .clearCache(name, callback)

Clears the persistent and runtime cache associated with the `name` package.   
If `name` is null, clears the cache for every package.

Note that in most cases, you don't need to clear the cache since it has
self expiration times.

```js
// Clear jquery cache
registry.clearCache('jquery', function (err) {
    if (err) {
        console.error(err.message);
        return;
    }

    console.log('Done');
});

// Clear all cache
registry.clearCache(function (err) {
    if (err) {
        console.error(err.message);
        return;
    }

    console.log('Done');
});
```


#### .resetCache()

Clears the in-memory cache used to speed up the instance.

Note that in most cases, you don't need to clear the runtime cache since it has
self expiration times.
Might be useful if you use this module in long-living programs.

```js
registry.resetCache();
```

#### #clearRuntimeCache()

Clears the in-memory cache used to speed up the whole module.
This clears the static in-memory cache as well as in-memory cache used by instances.

Note that in edge cases, some instance's in-memory cache might be skipped.
If that's a problem, you should create fresh instances instead.

```js
RegistryClient.clearRuntimeCache();
```


## License

Released under the [MIT License](http://www.opensource.org/licenses/mit-license.php).
