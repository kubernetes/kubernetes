Working with Spec and Config Files
==================================

Protractor needs two files to run, the test or spec file, and the configuration file.

Spec files
==========

Protractor tests  are written using the syntax of your test framework, for example [Jasmine](http://jasmine.github.io/), and the [Protractor API](/docs/api.md).

Example Spec File
-----------------
This simple script ([example_spec.js](/example/example_spec.js)) tests the 'The Basics' example on the [angularjs.org](http://www.angularjs.org) homepage.

```js
describe('angularjs homepage', function() {
  it('should greet the named user', function() {
    // Load the AngularJS homepage.
    browser.get('http://www.angularjs.org');

    // Find the element with ng-model matching 'yourName' - this will
    // find the <input type="text" ng-model="yourName"/> element - and then
    // type 'Julie' into it.
    element(by.model('yourName')).sendKeys('Julie');

    // Find the element with binding matching 'yourName' - this will
    // find the <h1>Hello {{yourName}}!</h1> element.
    var greeting = element(by.binding('yourName'));

    // Assert that the text element has the expected value.
    // Protractor patches 'expect' to understand promises.

    expect(greeting.getText()).toEqual('Hello Julie!');
  });
});
```

Global Variables
----------------

Protractor exports these global variables to your spec (test) file:

 - `browser` - A wrapper around an instance of WebDriver, used for navigation and page-wide information. The `browser.get` method loads a page. Protractor expects Angular to be present on a page, so it will throw an error if the page it is attempting to load does not contain the Angular library. (If you need to interact with a non-Angular page, you may access the wrapped webdriver instance directly with `browser.driver`).

 - `element` - A helper function for finding and interacting with DOM elements on the page you are testing. The `element` function searches for an element on the page. It requires one parameter, a locator strategy for locating the element. See [Using Locators](/docs/locators.md) for more information. See Protractor's findelements test suite ([elements_spec.js](/spec/basic/elements_spec.js)) for more examples.

 - `by` - A collection of element locator strategies. For example, elements can be found by CSS selector, by ID, or by the attribute they are bound to with ng-model. See [Using Locators](/docs/locators.md).

 - `protractor` - The Protractor namespace which wraps the WebDriver namespace. Contains static variables and classes, such as `protractor.Key` which enumerates the codes for special keyboard signals.


Config Files
============

The configuration file tells Protractor how to set up the Selenium Server, which tests to run, how to set up the browsers, and which test framework to use. The configuration file can also include one or more global settings.

Example Config File
-------------------

A simple configuration ([conf.js](https://github.com/angular/protractor/tree/master/example)) is shown below.
```js
// An example configuration file
exports.config = {
  // The address of a running selenium server.
  seleniumAddress: 'http://localhost:4444/wd/hub',

  // Capabilities to be passed to the webdriver instance.
  capabilities: {
    'browserName': 'chrome'
  },

  // Spec patterns are relative to the configuration file location passed
  // to proractor (in this example conf.js).
  // They may include glob patterns.
  specs: ['example-spec.js'],

  // Options to be passed to Jasmine-node.
  jasmineNodeOpts: {
    showColors: true, // Use colors in the command line report.
  }
};
```

Reference Config File
---------------------

The [reference config file](/docs/referenceConf.js) file provides explanations for all of the Protractor configuration options. Default settings include the standalone Selenium Server, the Chrome browser, and the Jasmine test framework. Additional information about various configuration options is available here:

 - [Setting Up the Selenium Server](/docs/server-setup.md)
 - [Setting Up the Browser](/docs/browser-setup.md)
 - [Choosing a Framework](/docs/frameworks.md)
 - [Using Page Objects to Organize Tests](/docs/page-objects.md)
