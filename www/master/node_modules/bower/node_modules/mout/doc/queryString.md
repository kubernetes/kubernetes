# queryString #

Utilities for query string manipulation.



## contains(url, paramName):Boolen

Checks if query string contains parameter.

### Arguments:

 1. `url` (String)     : URL or query string.
 2. `paramName` (String) : Parameter name.

### Example:

```js
var url = 'example.com/?lorem=ipsum';
contains(url, 'lorem'); // true
contains(url, 'foo');   //false
```



## decode(queryStr[, shouldTypecast]):Object

Parses query string and creates an object of keys => vals.

Will typecast value with [`string/typecast`](string.html#typecast) by default
and decode string parameters using `decodeURIComponent()`.

```js
var query = '?foo=bar&lorem=123';
decode(query);        // {foo: "bar", lorem: 123}
decode(query, false); // {foo: "bar", lorem: "123"}
```


## encode(obj):String

Encode object into a query string.

Will encode parameters with `encodeURIComponent()`.

```js
encode({foo: "bar", lorem: 123}); // "?foo=bar&lorem=123"
```


## getParam(url, param[, shouldTypecast]):*

Get query parameter value.

Will typecast value with [`string/typecast`](string.html#typecast) by default.

See: [`setParam()`](#setParam)

### Arguments:

 1. `url` (String) : Url.
 2. `param` (String) : Parameter name.
 3. `[shouldTypecast]` (Boolean) : If it should typecast value.

### Example:

```js
var url = 'example.com/?foo=bar&lorem=123&ipsum=false';
getParam(url, 'dolor');        // "amet"
getParam(url, 'lorem');        // 123
getParam(url, 'lorem', false); // "123"
```


## parse(url[, shouldTypecast]):Object

Parses URL, extracts query string and decodes it.

It will typecast all properties of the query object unless second argument is
`false`.

Alias to: `decode(getQuery(url))`.

```js
var url = 'example.com/?lorem=ipsum&a=123';
parse(url);        // {lorem: "ipsum", a: 123}
parse(url, false); // {lorem: "ipsum", a: "123"}
```


## getQuery(url):String

Gets full query as string with all special chars decoded.

```js
getQuery('example.com/?lorem=ipsum'); // "?lorem=ipsum"
```


## setParam(url, paramName, value):String

Add new query string parameter to URL or update existing value.

See: [`getParam()`](#getParam)

```js
setParam('?foo=bar&lorem=0', 'lorem', 'ipsum'); // '?foo=bar&lorem=ipsum'
setParam('?lorem=1', 'foo', 123); // '?lorem=1&foo=123'
```


-------------------------------------------------------------------------------

For more usage examples check specs inside `/tests` folder. Unit tests are the
best documentation you can get...
