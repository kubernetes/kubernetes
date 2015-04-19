Using Page Objects to Organize Tests
====================================

When writing end-to-end tests, a common pattern is to use [Page Objects](https://code.google.com/p/selenium/wiki/PageObjects). Page Objects help you write cleaner tests by encapsulating information about the elements on your application page. A Page Object can be reused across multiple tests, and if the template of your application changes, you only need to update the Page Object.

Without Page Objects
--------------------

Here’s a simple test script ([example_spec.js](/example/example_spec.js)) for ‘The Basics’ example on the [angularjs.org](http://www.angularjs.org) homepage.

```js
describe('angularjs homepage', function() {
  it('should greet the named user', function() {
    browser.get('http://www.angularjs.org');
    element(by.model('yourName')).sendKeys('Julie');
    var greeting = element(by.binding('yourName'));
    expect(greeting.getText()).toEqual('Hello Julie!');
  });
});
```

With PageObjects
----------------

To switch to Page Objects, the first thing you need to do is create a Page Object. A Page Object for ‘The Basics’ example on the angularjs.org homepage could look like this:

```js
var AngularHomepage = function() {
  this.nameInput = element(by.model('yourName'));
  this.greeting = element(by.binding('yourName'));

  this.get = function() {
    browser.get('http://www.angularjs.org');
  };

  this.setName = function(name) {
    this.nameInput.sendKeys(name);
  };
};
```
The next thing you need to do is modify the test script to use the PageObject and its properties. Note that the _functionality_ of the test script itself does not change (nothing is added or deleted).

```js
describe('angularjs homepage', function() {
  it('should greet the named user', function() {
    var angularHomepage = new AngularHomepage();
    angularHomepage.get();

    angularHomepage.setName('Julie');

    expect(angularHomepage.greeting.getText()).toEqual('Hello Julie!');
  });
});
```

Configuring Test Suites
-----------------------

It is possible to separate your tests into various test suites. In your config file, you could setup  the suites option as shown below. 

```js
exports.config = {
  // The address of a running selenium server.
  seleniumAddress: 'http://localhost:4444/wd/hub',

  // Capabilities to be passed to the webdriver instance.
  capabilities: {
    'browserName': 'chrome'
  },

  // Spec patterns are relative to the location of the spec file. They may
  // include glob patterns.
  suites: {
    homepage: 'tests/e2e/homepage/**/*Spec.js',
    search: ['tests/e2e/contact_search/**/*Spec.js',
      'tests/e2e/venue_search/**/*Spec.js']
  },

  // Options to be passed to Jasmine-node.
  jasmineNodeOpts: {
    showColors: true, // Use colors in the command line report.
  }
};
```

From the command line, you can then easily switch between running one or the other suite of tests. This command will run only the homepage section of the tests:

    protractor protractor.conf.js --suite homepage
