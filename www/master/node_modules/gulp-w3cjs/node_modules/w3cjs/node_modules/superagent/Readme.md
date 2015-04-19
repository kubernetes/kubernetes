# SuperAgent

  SuperAgent is a small progressive __client-side__ HTTP request library, and __Node.js__ module with the same API, sporting many high-level HTTP client features. View the [docs](http://visionmedia.github.com/superagent/).

![super agent](http://f.cl.ly/items/3d282n3A0h0Z0K2w0q2a/Screenshot.png)

## Motivation

  This library spawned from my frustration with jQuery's weak & inconsistent Ajax support. jQuery's API while having recently added some promise-like support, is largely static, forcing you to build up big objects containing all the header fields and options, not to mention most of the options are awkwardly named "type" instead of "method", etc. Onto examples!

  The following is what you might typically do for a simple __GET__ with jQuery:

```js
$.get('/user/1', function(data, textStatus, xhr){

});
```

great, it's ok, but it's kinda lame having 3 arguments just to access something on the `xhr`. Our equivalent would be:

```js
request.get('/user/1', function(res){

});
```

the response object is an instanceof `request.Response`, encapsulating all of this information instead of throwing a bunch of arguments at you. For example we can check `res.status`, `res.header` for header fields, `res.text`, `res.body` etc.

An example of a JSON POST with jQuery typically might use `$.post()`, however once you need to start defining header fields you have to then re-write it using `$.ajax()`... so that might look like:

```js
$.ajax({
  url: '/api/pet',
  type: 'POST',
  data: { name: 'Manny', species: 'cat' },
  headers: { 'X-API-Key': 'foobar' }
}).success(function(res){

}).error(function(){

});
```

 Not only is it ugly it's pretty opinionated, jQuery likes to special-case {4,5}xx, for example you cannot (easily at least) receive a parsed JSON response for say "400 Bad Request". This same request would look like this:

```js
request
  .post('/api/pet')
  .send({ name: 'Manny', species: 'cat' })
  .set('X-API-Key', 'foobar')
  .set('Accept', 'application/json')
  .end(function(error, res){

  });
```

building on the existing API internally we also provide something similar to `$.post()` for those times in life where your interactions are very basic:

```js
request.post('/api/pet', cat, function(error, res){

});
```

## Running node tests

  Install dependencies:

     $ npm install

  Run em!

    $ make test

## Running browser tests

 Install the test server deps (nodejs / express):

    $ npm install

 Start the test server:

    $ make test-server

 Visit `localhost:4000/` in the browser.

## Browser build

  The browser build of superagent is located in ./superagent.js

## Examples:

- [agency tests](https://github.com/visionmedia/superagent/blob/master/test/node/agency.js)
- [express demo app](https://github.com/hunterloftis/component-test/blob/master/lib/users/test/controller.test.js)

## Wiki

  For superagent extensions such as couchdb and oauth visit the [wiki](https://github.com/visionmedia/superagent/wiki).

## License

  MIT
