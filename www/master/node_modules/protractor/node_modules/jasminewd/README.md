jasminewd [![Build Status](https://travis-ci.org/angular/jasminewd.png?branch=master)](https://travis-ci.org/angular/jasminewd)
=========

Adapter for Jasmine-to-WebDriverJS. Used by [Protractor](http://www.github.com/angular/protractor).


Features
--------

 - Automatically makes tests asynchronously wait until the WebDriverJS control flow is empty.

 - If a `done` function is passed to the test, waits for both the control flow and until done is called.

 - Enhances `expect` so that it automatically unwraps promises before performing the assertion.

Installation
------------
```
npm install jasminewd
```

Usage
-----

Assumes selenium-webdriver as a peer dependency.

```js
// In your setup.
var minijn = require('minijasminenode');
require('jasminewd');

global.driver = new webdriver.Builder().
    usingServer('http://localhost:4444/wd/hub').
    withCapabilities({browserName: 'chrome'}).
    build();

minijn.executeSpecs(/* ... */);

// In your tests

describe('tests with webdriver', function() {
  it('will wait until webdriver is done', function() {
    // This will be an asynchronous test. It will finish once webdriver has
    // loaded the page, found the element, and gotten its text.
    driver.get('http://www.example.com');

    var myElement = driver.findElement(webdriver.By.id('hello'));

    // Here, expect understands that myElement.getText() is a promise,
    // and resolves it before asserting.
    expect(myElement.getText()).toEqual('hello world');
  });
})
```
