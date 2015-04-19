Setting Up the Browser
=======================

Protractor works with [Selenium WebDriver](http://docs.seleniumhq.org/docs/03_webdriver.jsp), a browser automation framework. Selenium WebDriver supports several browser implementations or [drivers](http://docs.seleniumhq.org/docs/03_webdriver.jsp#selenium-webdriver-s-drivers) which are discussed below.

Browser Support
---------------
Protractor supports the two latest major versions of Chrome, Firefox, Safari, and IE.

Please see [Browser Support](/docs/browser-support.md) for a full list of
supported browsers and known issues.


Configuring Browsers
--------------------

In your Protractor config file (see [referenceConf.js](/docs/referenceConf.js)), all browser setup is done within the `capabilities` object. This object is passed directly to the WebDriver builder ([builder.js](https://code.google.com/p/selenium/source/browse/javascript/webdriver/builder.js)). 


See [DesiredCapabilities](https://code.google.com/p/selenium/wiki/DesiredCapabilities) for full information on which properties are available.


Using Mobile Browsers
---------------------

Please see the [Mobile Setup](/docs/mobile-setup.md) documentation for information on mobile browsers.


Using Browsers Other Than Chrome
--------------------------------

To use a browser other than Chrome, simply set a different browser name in the capabilities object.

```javascript
capabilities: {
  'browserName': 'firefox'
}
```

You may need to install a separate binary to run another browser, such as IE or Android. For more information, see [SeleniumHQ Downloads](http://docs.seleniumhq.org/download/).


Adding Chrome-Specific Options
------------------------------

Chrome options are nested in the `chromeOptions` object. A full list of options is at the [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/capabilities) site. For example, to show an FPS counter in the upper right, your configuration would look like this:

```javascript
capabilities: {
  'browserName': 'chrome',
  'chromeOptions': {
    'args': ['show-fps-counter=true']
  }
},
```


Testing Against Multiple Browsers
---------------------------------

If you would like to test against multiple browsers, use the `multiCapabilities` configuration option.

```javascript
multiCapabilities: [{
  'browserName': 'firefox'
}, {
  'browserName': 'chrome'
}]
```

Protractor will run tests in parallel against each set of capabilities. Please note that if `multiCapabilities` is defined, the runner will ignore the `capabilities` configuration.


Using Multiple Browsers in the Same Test
----------------------------------------
If you are testing apps where two browsers need to interact with each other (e.g. chat systems), you can do that with protractor by dynamically creating browsers on the go in your test. Protractor exposes a function in the `browser` object to help you achieve this: `browser.forkNewDriverInstance(opt_useSameUrl, opt_copyMockModules)`. 
Calling this will return a new independent browser object. The first parameter in the function denotes whether you want the new browser to start with the same url as the browser you forked from. The second parameter denotes whether you want the new browser to copy the mock modules from the browser you forked from.

```javascript
browser.get('http://www.angularjs.org');
browser.addMockModule('moduleA', "angular.module('moduleA', []).value('version', '3');");

// To create a new browser.
var browser2 = browser.forkNewDriverInstance();

// To create a new browser with url as 'http://www.angularjs.org':
var browser3 = browser.forkNewDriverInstance(true);

// To create a new browser with mock modules injected:
var browser4 = browser.forkNewDriverInstance(false, true);

// To create a new browser with url as 'http://www.angularjs.org' and mock modules injected:
var browser4 = browser.forkNewDriverInstance(true, true);
```

Now you can interact with the browsers. However, note that the globals `element`, `$`, `$$` and `browser` are all associated with the original browser. In order to interact with the new browsers, you must specifically tell protractor to do so like the following:

```javascript
var element2 = browser2.element;
var $2 = browser2.$;
var $$2 = browser2.$$;
element2(by.model(...)).click();
$2('.css').click();
$$2('.css').click();
```

Protractor will ensure that commands will automatically run in sync. For example, in the following code, `element(by.model(...)).click()` will run before `browser2.$('.css').click()`:

```javascript
browser.get('http://www.angularjs.org');
browser2.get('http://localhost:1234');

browser.sleep(5000);
element(by.model(...)).click();
browser2.$('.css').click();
```


Setting up PhantomJS
--------------------
_Note: We recommend against using PhantomJS for tests with Protractor. There are many reported issues with PhantomJS crashing and behaving differently from real browsers._

In order to test locally with [PhantomJS](http://phantomjs.org/), you'll need to either have it installed globally, or relative to your project. For global install see the [PhantomJS download page](http://phantomjs.org/download.html). For local install run: `npm install phantomjs`.

Add phantomjs to the driver capabilities, and include a path to the binary if using local installation:
```javascript
capabilities: {
  'browserName': 'phantomjs',

  /* 
   * Can be used to specify the phantomjs binary path.
   * This can generally be ommitted if you installed phantomjs globally.
   */
  'phantomjs.binary.path': require('phantomjs').path,
  
  /*
   * Command line args to pass to ghostdriver, phantomjs's browser driver.
   * See https://github.com/detro/ghostdriver#faq
   */
  'phantomjs.ghostdriver.cli.args': ['--loglevel=DEBUG']
}
```
