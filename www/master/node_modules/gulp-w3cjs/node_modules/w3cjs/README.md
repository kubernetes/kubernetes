
[![Build Status](https://secure.travis-ci.org/thomasdavis/w3cjs.png?branch=master)](http://travis-ci.org/thomasdavis/w3cjs)

# w3cjs

A node.js library for testing files or url's against the w3c html validator.

## Installation

```js
npm install w3cjs
```

## Usage

```js
var w3cjs = require('w3cjs');

var results = w3cjs.validate({
	file: 'demo.html', // file can either be a local file or a remote file
	//file: 'http://html5boilerplate.com/',
	//input: '<html>...</html>',
	//input: myBuffer,
	output: 'json', // Defaults to 'json', other option includes html
	doctype: 'HTML5', // Defaults false for autodetect
	charset: 'utf-8', // Defaults false for autodetect
	proxy: 'http://proxy:8080', // Default to null
	callback: function (res) {
		console.log(res);
		// depending on the output type, res will either be a json object or a html string
	}
});
```

## Example async testing with Mocha 

```js
var w3cjs = require('w3cjs');
describe('html validation', function(){
	it('index page should have no html errors', function(done){
		w3cjs.validate({
			file: 'index.html',
			callback: function (res) {
					console.log(res);
				if (res.messages.length > 0 ) {
					throw {error: 'html errors have been found', results: res};
				};
				done();
			}
		})
	})
})

```



<img alt="Clicky" width="1" height="1" src="//in.getclicky.com/66606907ns.gif" />
