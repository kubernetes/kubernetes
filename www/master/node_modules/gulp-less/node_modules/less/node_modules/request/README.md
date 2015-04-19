# Request — Simplified HTTP client

[![NPM](https://nodei.co/npm/request.png)](https://nodei.co/npm/request/)

## Super simple to use

Request is designed to be the simplest way possible to make http calls. It supports HTTPS and follows redirects by default.

```javascript
var request = require('request');
request('http://www.google.com', function (error, response, body) {
  if (!error && response.statusCode == 200) {
    console.log(body) // Print the google web page.
  }
})
```

## Streaming

You can stream any response to a file stream.

```javascript
request('http://google.com/doodle.png').pipe(fs.createWriteStream('doodle.png'))
```

You can also stream a file to a PUT or POST request. This method will also check the file extension against a mapping of file extensions to content-types (in this case `application/json`) and use the proper `content-type` in the PUT request (if the headers don’t already provide one).

```javascript
fs.createReadStream('file.json').pipe(request.put('http://mysite.com/obj.json'))
```

Request can also `pipe` to itself. When doing so, `content-type` and `content-length` are preserved in the PUT headers.

```javascript
request.get('http://google.com/img.png').pipe(request.put('http://mysite.com/img.png'))
```

Now let’s get fancy.

```javascript
http.createServer(function (req, resp) {
  if (req.url === '/doodle.png') {
    if (req.method === 'PUT') {
      req.pipe(request.put('http://mysite.com/doodle.png'))
    } else if (req.method === 'GET' || req.method === 'HEAD') {
      request.get('http://mysite.com/doodle.png').pipe(resp)
    }
  }
})
```

You can also `pipe()` from `http.ServerRequest` instances, as well as to `http.ServerResponse` instances. The HTTP method, headers, and entity-body data will be sent. Which means that, if you don't really care about security, you can do:

```javascript
http.createServer(function (req, resp) {
  if (req.url === '/doodle.png') {
    var x = request('http://mysite.com/doodle.png')
    req.pipe(x)
    x.pipe(resp)
  }
})
```

And since `pipe()` returns the destination stream in ≥ Node 0.5.x you can do one line proxying. :)

```javascript
req.pipe(request('http://mysite.com/doodle.png')).pipe(resp)
```

Also, none of this new functionality conflicts with requests previous features, it just expands them.

```javascript
var r = request.defaults({'proxy':'http://localproxy.com'})

http.createServer(function (req, resp) {
  if (req.url === '/doodle.png') {
    r.get('http://google.com/doodle.png').pipe(resp)
  }
})
```

You can still use intermediate proxies, the requests will still follow HTTP forwards, etc.

## UNIX Socket 

`request` supports the `unix://` protocol for all requests. The path is assumed to be absolute to the root of the host file system. 

HTTP paths are extracted from the supplied URL by testing each level of the full URL against net.connect for a socket response.

Thus the following request will GET `/httppath` from the HTTP server listening on `/tmp/unix.socket`

```javascript
request.get('unix://tmp/unix.socket/httppath')
```

## Forms

`request` supports `application/x-www-form-urlencoded` and `multipart/form-data` form uploads. For `multipart/related` refer to the `multipart` API.

URL-encoded forms are simple.

```javascript
request.post('http://service.com/upload', {form:{key:'value'}})
// or
request.post('http://service.com/upload').form({key:'value'})
```

For `multipart/form-data` we use the [form-data](https://github.com/felixge/node-form-data) library by [@felixge](https://github.com/felixge). You don’t need to worry about piping the form object or setting the headers, `request` will handle that for you.

```javascript
var r = request.post('http://service.com/upload', function optionalCallback (err, httpResponse, body) {
  if (err) {
    return console.error('upload failed:', err);
  }
  console.log('Upload successful!  Server responded with:', body);
})
var form = r.form()
form.append('my_field', 'my_value')
form.append('my_buffer', new Buffer([1, 2, 3]))
form.append('my_file', fs.createReadStream(path.join(__dirname, 'doodle.png')))
form.append('remote_file', request('http://google.com/doodle.png'))

// Just like always, `r` is a writable stream, and can be used as such (you have until nextTick to pipe it, etc.)
// Alternatively, you can provide a callback (that's what this example does — see `optionalCallback` above).
```

## HTTP Authentication

```javascript
request.get('http://some.server.com/').auth('username', 'password', false);
// or
request.get('http://some.server.com/', {
  'auth': {
    'user': 'username',
    'pass': 'password',
    'sendImmediately': false
  }
});
// or
request.get('http://some.server.com/').auth(null, null, true, 'bearerToken');
// or
request.get('http://some.server.com/', {
  'auth': {
    'bearer': 'bearerToken'
  }
});
```

If passed as an option, `auth` should be a hash containing values `user` || `username`, `pass` || `password`, and `sendImmediately` (optional).  The method form takes parameters `auth(username, password, sendImmediately)`.

`sendImmediately` defaults to `true`, which causes a basic authentication header to be sent.  If `sendImmediately` is `false`, then `request` will retry with a proper authentication header after receiving a `401` response from the server (which must contain a `WWW-Authenticate` header indicating the required authentication method).

Note that you can also use for basic authentication a trick using the URL itself, as specified in [RFC 1738](http://www.ietf.org/rfc/rfc1738.txt). 
Simply pass the `user:password` before the host with an `@` sign.

```javascript
var username = 'username',
    password = 'password',
    url = 'http://' + username + ':' + password + '@some.server.com';

request({url: url}, function (error, response, body) {
   // Do more stuff with 'body' here
});
```

Digest authentication is supported, but it only works with `sendImmediately` set to `false`; otherwise `request` will send basic authentication on the initial request, which will probably cause the request to fail.

Bearer authentication is supported, and is activated when the `bearer` value is available. The value may be either a `String` or a `Function` returning a `String`. Using a function to supply the bearer token is particularly useful if used in conjuction with `defaults` to allow a single function to supply the last known token at the time or sending a request or to compute one on the fly.

## OAuth Signing

```javascript
// Twitter OAuth
var qs = require('querystring')
  , oauth =
    { callback: 'http://mysite.com/callback/'
    , consumer_key: CONSUMER_KEY
    , consumer_secret: CONSUMER_SECRET
    }
  , url = 'https://api.twitter.com/oauth/request_token'
  ;
request.post({url:url, oauth:oauth}, function (e, r, body) {
  // Ideally, you would take the body in the response
  // and construct a URL that a user clicks on (like a sign in button).
  // The verifier is only available in the response after a user has
  // verified with twitter that they are authorizing your app.
  var access_token = qs.parse(body)
    , oauth =
      { consumer_key: CONSUMER_KEY
      , consumer_secret: CONSUMER_SECRET
      , token: access_token.oauth_token
      , verifier: access_token.oauth_verifier
      }
    , url = 'https://api.twitter.com/oauth/access_token'
    ;
  request.post({url:url, oauth:oauth}, function (e, r, body) {
    var perm_token = qs.parse(body)
      , oauth =
        { consumer_key: CONSUMER_KEY
        , consumer_secret: CONSUMER_SECRET
        , token: perm_token.oauth_token
        , token_secret: perm_token.oauth_token_secret
        }
      , url = 'https://api.twitter.com/1.1/users/show.json?'
      , params =
        { screen_name: perm_token.screen_name
        , user_id: perm_token.user_id
        }
      ;
    url += qs.stringify(params)
    request.get({url:url, oauth:oauth, json:true}, function (e, r, user) {
      console.log(user)
    })
  })
})
```

### Custom HTTP Headers

HTTP Headers, such as `User-Agent`, can be set in the `options` object.
In the example below, we call the github API to find out the number
of stars and forks for the request repository. This requires a
custom `User-Agent` header as well as https.

```javascript
var request = require('request');

var options = {
	url: 'https://api.github.com/repos/mikeal/request',
	headers: {
		'User-Agent': 'request'
	}
};

function callback(error, response, body) {
	if (!error && response.statusCode == 200) {
		var info = JSON.parse(body);
		console.log(info.stargazers_count + " Stars");
		console.log(info.forks_count + " Forks");
	}
}

request(options, callback);
```

### request(options, callback)

The first argument can be either a `url` or an `options` object. The only required option is `uri`; all others are optional.

* `uri` || `url` - fully qualified uri or a parsed url object from `url.parse()`
* `qs` - object containing querystring values to be appended to the `uri`
* `method` - http method (default: `"GET"`)
* `headers` - http headers (default: `{}`)
* `body` - entity body for PATCH, POST and PUT requests. Must be a `Buffer` or `String`.
* `form` - when passed an object or a querystring, this sets `body` to a querystring representation of value, and adds `Content-type: application/x-www-form-urlencoded; charset=utf-8` header. When passed no options, a `FormData` instance is returned (and is piped to request).
* `auth` - A hash containing values `user` || `username`, `pass` || `password`, and `sendImmediately` (optional).  See documentation above.
* `json` - sets `body` but to JSON representation of value and adds `Content-type: application/json` header.  Additionally, parses the response body as JSON.
* `multipart` - (experimental) array of objects which contains their own headers and `body` attribute. Sends `multipart/related` request. See example below.
* `followRedirect` - follow HTTP 3xx responses as redirects (default: `true`)
* `followAllRedirects` - follow non-GET HTTP 3xx responses as redirects (default: `false`)
* `maxRedirects` - the maximum number of redirects to follow (default: `10`)
* `encoding` - Encoding to be used on `setEncoding` of response data. If `null`, the `body` is returned as a `Buffer`.
* `pool` - A hash object containing the agents for these requests. If omitted, the request will use the global pool (which is set to node's default `maxSockets`)
* `pool.maxSockets` - Integer containing the maximum amount of sockets in the pool.
* `timeout` - Integer containing the number of milliseconds to wait for a request to respond before aborting the request
* `proxy` - An HTTP proxy to be used. Supports proxy Auth with Basic Auth, identical to support for the `url` parameter (by embedding the auth info in the `uri`)
* `oauth` - Options for OAuth HMAC-SHA1 signing. See documentation above.
* `hawk` - Options for [Hawk signing](https://github.com/hueniverse/hawk). The `credentials` key must contain the necessary signing info, [see hawk docs for details](https://github.com/hueniverse/hawk#usage-example).
* `strictSSL` - If `true`, requires SSL certificates be valid. **Note:** to use your own certificate authority, you need to specify an agent that was created with that CA as an option.
* `jar` - If `true` and `tough-cookie` is installed, remember cookies for future use (or define your custom cookie jar; see examples section)
* `aws` - `object` containing AWS signing information. Should have the properties `key`, `secret`. Also requires the property `bucket`, unless you’re specifying your `bucket` as part of the path, or the request doesn’t use a bucket (i.e. GET Services)
* `httpSignature` - Options for the [HTTP Signature Scheme](https://github.com/joyent/node-http-signature/blob/master/http_signing.md) using [Joyent's library](https://github.com/joyent/node-http-signature). The `keyId` and `key` properties must be specified. See the docs for other options.
* `localAddress` - Local interface to bind for network connections.
* `gzip` - If `true`, add an `Accept-Encoding` header to request compressed content encodings from the server (if not already present) and decode supported content encodings in the response.


The callback argument gets 3 arguments: 

1. An `error` when applicable (usually from [`http.ClientRequest`](http://nodejs.org/api/http.html#http_class_http_clientrequest) object)
2. An [`http.IncomingMessage`](http://nodejs.org/api/http.html#http_http_incomingmessage) object
3. The third is the `response` body (`String` or `Buffer`, or JSON object if the `json` option is supplied)

## Convenience methods

There are also shorthand methods for different HTTP METHODs and some other conveniences.

### request.defaults(options)

This method returns a wrapper around the normal request API that defaults to whatever options you pass in to it.

### request.put

Same as `request()`, but defaults to `method: "PUT"`.

```javascript
request.put(url)
```

### request.patch

Same as `request()`, but defaults to `method: "PATCH"`.

```javascript
request.patch(url)
```

### request.post

Same as `request()`, but defaults to `method: "POST"`.

```javascript
request.post(url)
```

### request.head

Same as request() but defaults to `method: "HEAD"`.

```javascript
request.head(url)
```

### request.del

Same as `request()`, but defaults to `method: "DELETE"`.

```javascript
request.del(url)
```

### request.get

Same as `request()` (for uniformity).

```javascript
request.get(url)
```
### request.cookie

Function that creates a new cookie.

```javascript
request.cookie('cookie_string_here')
```
### request.jar

Function that creates a new cookie jar.

```javascript
request.jar()
```


## Examples:

```javascript
  var request = require('request')
    , rand = Math.floor(Math.random()*100000000).toString()
    ;
  request(
    { method: 'PUT'
    , uri: 'http://mikeal.iriscouch.com/testjs/' + rand
    , multipart:
      [ { 'content-type': 'application/json'
        ,  body: JSON.stringify({foo: 'bar', _attachments: {'message.txt': {follows: true, length: 18, 'content_type': 'text/plain' }}})
        }
      , { body: 'I am an attachment' }
      ]
    }
  , function (error, response, body) {
      if(response.statusCode == 201){
        console.log('document saved as: http://mikeal.iriscouch.com/testjs/'+ rand)
      } else {
        console.log('error: '+ response.statusCode)
        console.log(body)
      }
    }
  )
```

Cookies are disabled by default (else, they would be used in subsequent requests). To enable cookies, set `jar` to `true` (either in `defaults` or `options`) and install `tough-cookie`.

```javascript
var request = request.defaults({jar: true})
request('http://www.google.com', function () {
  request('http://images.google.com')
})
```

To use a custom cookie jar (instead of `request`’s global cookie jar), set `jar` to an instance of `request.jar()` (either in `defaults` or `options`)

```javascript
var j = request.jar()
var request = request.defaults({jar:j})
request('http://www.google.com', function () {
  request('http://images.google.com')
})
```

OR

```javascript
// `npm install --save tough-cookie` before this works
var j = request.jar()
var cookie = request.cookie('your_cookie_here')
j.setCookie(cookie, uri);
request({url: 'http://www.google.com', jar: j}, function () {
  request('http://images.google.com')
})
```

To inspect your cookie jar after a request

```javascript
var j = request.jar() 
request({url: 'http://www.google.com', jar: j}, function () {
  var cookie_string = j.getCookieString(uri); // "key1=value1; key2=value2; ..."
  var cookies = j.getCookies(uri); 
  // [{key: 'key1', value: 'value1', domain: "www.google.com", ...}, ...]
})
```
