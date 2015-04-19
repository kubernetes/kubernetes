## v2.44.0

* Added the `until` module, which defines common explicit wait conditions.
    Sample usage:

        var firefox = require('selenium-webdriver/firefox'),
            until = require('selenium-webdriver/until');

        var driver = new firefox.Driver();
        driver.get('http://www.google.com/ncr');
        driver.wait(until.titleIs('Google Search'), 1000);

* FIXED: 8000: `Builder.forBrowser()` now accepts an empty string since some
    WebDriver implementations ignore the value. A value must still be specified,
    however, since it is a required field in WebDriver's wire protocol.
* FIXED: 7994: The `stacktrace` module will not modify stack traces if the
    initial parse fails (e.g. the user defined `Error.prepareStackTrace`)
* FIXED: 5855: Added a module (`until`) that defines several common conditions 
    for use with explicit waits. See updated examples for usage.

## v2.43.5

* FIXED: 7905: `Builder.usingServer(url)` once again returns `this` for
    chaining.

## v2.43.2-4

* No changes; version bumps while attempting to work around an issue with
    publishing to npm (a version string may only be used once).

## v2.43.1

* Fixed an issue with flakiness when setting up the Firefox profile that could
    prevent the driver from initializing properly.

## v2.43.0

* Added native support for Firefox - the Java Selenium server is no longer
    required.
* Added support for generator functions to `ControlFlow#execute` and
    `ControlFlow#wait`. For more information, see documentation on
    `webdriver.promise.consume`. Requires harmony support (run with
    `node --harmony-generators` in `v0.11.x`).
* Various improvements to the `Builder` API. Notably, the `build()` function
    will no longer default to attempting to use a server at
    `http://localhost:4444/wd/hub` if it cannot start a browser directly -
    you must specify the WebDriver server with `usingServer(url)`. You can
    also set the target browser and WebDriver server through a pair of
    environment variables. See the documentation on the `Builder` constructor
    for more information.
* For consistency with the other language bindings, added browser specific
    classes that can be used to start a browser without the builder.

        var webdriver = require('selenium-webdriver')
            chrome = require('selenium-webdriver/chrome');

        // The following are equivalent.
        var driver1 = new webdriver.Builder().forBrowser('chrome').build();
        var driver2 = new chrome.Driver();

* Promise A+ compliance: a promise may no longer resolve to itself.
* For consistency with other language bindings, deprecated
    `UnhandledAlertError#getAlert` and added `#getAlertText`.
    `getAlert` will be removed in `2.45.0`.
* FIXED: 7641: Deprecated `ErrorCode.NO_MODAL_DIALOG_OPEN` and
    `ErrorCode.MODAL_DIALOG_OPENED` in favor of the new
    `ErrorCode.NO_SUCH_ALERT` and `ErrorCode.UNEXPECTED_ALERT_OPEN`,
    respecitvely.
* FIXED: 7563: Mocha integration no longer disables timeouts. Default Mocha
    timeouts apply (2000 ms) and may be changed using `this.timeout(ms)`.
* FIXED: 7470: Make it easier to create WebDriver instances in custom flows for
    parallel execution.

## v2.42.1

* FIXED: 7465: Fixed `net.getLoopbackAddress` on Windows
* FIXED: 7277: Support `done` callback in Mocha's BDD interface
* FIXED: 7156: `Promise#thenFinally` should not suppress original error

## v2.42.0

* Removed deprecated functions `Promise#addCallback()`,
    `Promise#addCallbacks()`, `Promise#addErrback()`, and `Promise#addBoth()`.
* Fail with a more descriptive error if the server returns a malformed redirect
* FIXED: 7300: Connect to ChromeDriver using the loopback address since
    ChromeDriver 2.10.267517 binds to localhost by default.
* FIXED: 7339: Preserve wrapped test function's string representation for
    Mocha's BDD interface.

## v2.41.0

* FIXED: 7138: export logging API from webdriver module.
* FIXED: 7105: beforeEach/it/afterEach properly bind `this` for Mocha tests.

## v2.40.0

* API documentation is now included in the docs directory.
* Added utility functions for working with an array of promises:
    `promise.all`, `promise.map`, and `promise.filter`
* Introduced `Promise#thenCatch()` and `Promise#thenFinally()`.
* Deprecated `Promise#addCallback()`, `Promise#addCallbacks()`,
    `Promise#addErrback()`, and `Promise#addBoth()`.
* Removed deprecated function `webdriver.WebDriver#getCapability`.
* FIXED: 6826: Added support for custom locators.

## v2.39.0

* Version bump to stay in sync with the Selenium project.

## v2.38.1

* FIXED: 6686: Changed `webdriver.promise.Deferred#cancel()` to silently no-op
    if the deferred has already been resolved.

## v2.38.0

* When a promise is rejected, always annotate the stacktrace with the parent
    flow state so users can identify the source of an error.
* Updated tests to reflect features not working correctly in the SafariDriver
    (cookie management and proxy support; see issues 5051, 5212, and 5503)
* FIXED: 6284: For mouse moves, correctly omit the x/y offsets if not
    specified as a function argument (instead of passing (0,0)).
* FIXED: 6471: Updated documentation on `webdriver.WebElement#getAttribute`
* FIXED: 6612: On Unix, use the default IANA ephemeral port range if unable to
    retrieve the current system's port range.
* FIXED: 6617: Avoid triggering the node debugger when initializing the
    stacktrace module.
* FIXED: 6627: Safely rebuild chrome.Options from a partial JSON spec.

## v2.37.0

* FIXED: 6346: The remote.SeleniumServer class now accepts JVM arguments using
    the `jvmArgs` option.

## v2.36.0

* _Release skipped to stay in sync with main Selenium project._

## v2.35.2

* FIXED: 6200: Pass arguments to the Selenium server instead of to the JVM.

## v2.35.1

* FIXED: 6090: Changed example scripts to use chromedriver.

## v2.35.0

* Version bump to stay in sync with the Selenium project.

## v2.34.1

* FIXED: 6079: The parent process should not wait for spawn driver service
    processes (chromedriver, phantomjs, etc.)

## v2.34.0

* Added the `selenium-webdriver/testing/assert` module. This module
    simplifies writing assertions against promised values (see
    example in module documentation).
* Added the `webdriver.Capabilities` class.
* Added native support for the ChromeDriver. When using the `Builder`,
    requesting chrome without specifying a remote server URL will default to
    the native ChromeDriver implementation.  The
    [ChromeDriver server](https://code.google.com/p/chromedriver/downloads/list)
    must be downloaded separately.

        // Will start ChromeDriver locally.
        var driver = new webdriver.Builder().
            withCapabilities(webdriver.Capabilities.chrome()).
            build();

        // Will start ChromeDriver using the remote server.
        var driver = new webdriver.Builder().
            withCapabilities(webdriver.Capabilities.chrome()).
            usingServer('http://server:1234/wd/hub').
            build();

* Added support for configuring proxies through the builder. For examples, see
    `selenium-webdriver/test/proxy_test`.
* Added native support for PhantomJS.
* Changed signature of `SeleniumServer` to `SeleniumServer(jar, options)`.
* Tests are now included in the npm published package. See `README.md` for
    execution instructions
* Removed the deprecated `webdriver.Deferred#resolve` and
    `webdriver.promise.resolved` functions.
* Removed the ability to connect to an existing session from the Builder. This
    feature is intended for use with the browser-based client.

## v2.33.0

* Added support for WebDriver's logging API
* FIXED: 5511: Added webdriver.manage().timeouts().pageLoadTimeout(ms)

## v2.32.1

* FIXED: 5541: Added missing return statement for windows in
    `portprober.findFreePort()`

## v2.32.0

* Added the `selenium-webdriver/testing` package, which provides a basic
    framework for writing tests using Mocha. See
    `selenium-webdriver/example/google_search_test.js` for usage.
* For Promises/A+ compatibility, backing out the change in 2.30.0 that ensured
    rejections were always Error objects. Rejection reasons are now left as is.
* Removed deprecated functions originally scheduled for removal in 2.31.0
    * promise.Application.getInstance()
    * promise.ControlFlow#schedule()
    * promise.ControlFlow#scheduleTimeout()
    * promise.ControlFlow#scheduleWait()
* Renamed some functions for consistency with Promises/A+ terminology. The
    original functions have been deprecated and will be removed in 2.34.0:
    * promise.resolved() -> promise.fulfilled()
    * promise.Deferred#resolve() -> promise.Deferred#fulfill()
* FIXED: remote.SeleniumServer#stop now shuts down within the active control
    flow, allowing scripts to finish. Use #kill to shutdown immediately.
* FIXED: 5321: cookie deletion commands

## v2.31.0

* Added an example script.
* Added a class for controlling the standalone Selenium server (server
available separately)
* Added a portprober for finding free ports
* FIXED: WebElements now belong to the same flow as their parent driver.

## v2.30.0

* Ensures promise rejections are always Error values.
* Version bump to keep in sync with the Selenium project.

## v2.29.1

* Fixed a bug that could lead to an infinite loop.
* Added a README.md

## v2.29.0

* Initial release for npm:

        npm install selenium-webdriver
