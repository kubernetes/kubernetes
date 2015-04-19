# 1.8.0

## Dependency Version Upgrades

- ([1159612](https://github.com/angular/protractor/commit/1159612ed76bb65612dbb2cc648e45928a251b10)) 
  fix(webdriver): bump selenium to 2.45.0

  Bump the selenium standalone binary to 2.45.0.

  See https://code.google.com/p/selenium/source/browse/java/CHANGELOG for a full list of changes to
  the selenium server.

  Closes #1734

## Features

- ([54163dc](https://github.com/angular/protractor/commit/54163dcd22cee27cf16685fbb4d53a2712233d26)) 
  feat(a11yPlugin): plugin for integrating with Chrome Accessibility Developer Tools

  Also includes missing Angular map files. See plugins/accessibility/index.js for usage.

- ([658902b](https://github.com/angular/protractor/commit/658902bd04bf809bde2751db79e93ae00de2f810)) 
  feat(plugins): add postTest hook for plugins

  Additionally, add some tests to make sure that plugins can fail properly.

  Closes #1842

- ([13d34c9](https://github.com/angular/protractor/commit/13d34c9192a06634827d89bf356bea33fea75747)) 
  feat(a11yPlugin): add support for Tenon.io

- ([5f8cffd](https://github.com/angular/protractor/commit/5f8cffd95c50ab4e7949376425f10e13747eb922)) 
  feat(plugins): allow plugins to export a name for use in reporting

## Bug Fixes

- ([aabdd56](https://github.com/angular/protractor/commit/aabdd567ee62d0d48fad499ee5decbb5d7d6b939)) 
  fix(debugger): breakpoint isn't set properly for windows

- ([361ae21](https://github.com/angular/protractor/commit/361ae21ee761eb78d1e2c9b2b7d270873a28ef81)) 
  fix(plugins): add a 'test' or 'fail' string to plugins

  Closes #1843

- ([847e739](https://github.com/angular/protractor/commit/847e73961e52caa1537df269589d9cfe6373b986)) 
  fix(webdriver-manager): unzipping ie driver should overwrite old version

# 1.7.0

## Dependency Version Upgrades

- ([2658865](https://github.com/angular/protractor/commit/2658865640d82617e69208cdb2263a2073a20156)) 
  feat(webdriver): bump chromedriver to 2.14

  Chromedriver 2.14 contains support for accessing elements inside the shadow DOM.

## Features

- ([d220ecf](https://github.com/angular/protractor/commit/d220ecf5ebc7ba023eab728d4a684e978ff77c83)) 
  feat(locators): add by.deepCss selector for finding elements in the shadow dom

  Usage:

  ```
  element(by.deepCss('.foo'))
  equivalent to 'element(by.css('* /deep/ .foo'))
  ```

- ([324f69d](https://github.com/angular/protractor/commit/324f69d6aa7c23ad77f1d50e26e0a56bade40132)) 
  feat(locators): add by.exactRepeater

- ([eb9d567](https://github.com/angular/protractor/commit/eb9d56755fa93401502e7608c7c3d0f16927c082)) 
  feat(frameworks): add support for custom frameworks

  Usage:

  ```js
  exports.config = {
    framework: 'custom',
    frameworkPath: '/path/to/your/framework/index.js'
  }
  ```

- ([9bc1c53](https://github.com/angular/protractor/commit/9bc1c53e40161521b0c125a810f86235c974f100)) 
  feat(expectedConditions): add helper library for syncing with non-angular apps

  Usage:

  ```javascript
  var EC = protractor.ExpectedConditions;
  var button = $('#xyz');
  var isClickable = EC.elementToBeClickable(button);

  browser.get(URL); browser.wait(isClickable, 5000); //wait for an element to become clickable 
  button.click();
  ```

  You can also customize the conditions:

  ```javascript
  var urlChanged = function() {
    return browser.getCurrentUrl().then(function(url) {
      return url != 'http://www.angularjs.org';
    });
  };

  // condition to wait for url to change, title to contain 'foo', and $('abc') element to contain text 'bar'
  var condition = EC.and(urlChanged, EC.titleContains('foo'),
      EC.textToBePresentInElement($('abc'), 'bar'));
  $('navButton').click(); browser.wait(condition, 5000); //wait for condition to be true.
  // do other things
  ```

- ([fb099de](https://github.com/angular/protractor/commit/fb099dedf92a64732d88401dd1b0d4d30b22650d)) 
  feat(elementExplorer): Combine browser.pause with elementExplorer

   * reuse logic for browser.pause for elementExplorer
   * introduce browser.enterRepl
   * allow customization of driver for elementExplorer
   * fix bug where repl cannot return an ElementFinder (related #1600)

    Closes #1314, #1315

- ([9def5e0](https://github.com/angular/protractor/commit/9def5e0e67e031949010fed4ed47178a534c99e8)) 
  feat(runner): add browser.getProcessedConfig method

  Now, instances of the `browser` object have a `getProcessedConfig` method which returns a promise
  that resolves to the current Protractor configuration object for the current runner instance. This
  means that if multiCapabilities are being used or tests are sharded, `getProcessedConfig` will
  return an object with the `capabilities` and `specs` property specific to the current instance.

  Closes #1724

## Bug Fixes

- ([ccb165d](https://github.com/angular/protractor/commit/ccb165d99b69e1ae66e4c1badd2f4e04f1481e75)) 
  fix(webdriver-manager): unzipping chromedriver should override old version

  See #1813

# 1.6.1

## Bug Fixes

- ([92c5d17](https://github.com/angular/protractor/commit/92c5d17844a2b4dc56c483ab4a65e2bf631175f9)) 
  fix(element): test crashes when using certain locators with `fromWebElement_`

  Protractor crashes when one uses locators with findElementsOverride (i.e. any custom protractor
  locator like by.binding/repeater/etc) in map/filter/then/each/reduce

# 1.6.0

## Features

- ([1e60a95](https://github.com/angular/protractor/commit/1e60a9504c883a95f3500eafa38e1fc11dc28c9b)) 
  feat(frameworks): add jasmine2 framework

  Jasmine2.x may now be used by setting `framework: jasmine2` in your config.
  See https://github.com/angular/protractor/blob/master/docs/jasmine-upgrade.md

- ([0b93003](https://github.com/angular/protractor/commit/0b930035905d1868225667de358222e51394f3ac)) 
  feat(jasmine2): add 'grep' option to jasmine2

  Allow users to filter the specs that they want to run using simple string match. To use this
  feature, either: 1) specify jasmineNodeOpts.grep in your conf.js file
   or 2) via commandline like "protractor conf.js --grep='pattern to match'"

- ([4368842](https://github.com/angular/protractor/commit/4368842da73d4ed501df21b61daf71951e59524b)) 
  feat(wddebugger): enable repl (with autocomplete) for browser.pause

  See https://github.com/angular/protractor/blob/master/docs/debugging.md for
  usage.

- ([9c9ed31](https://github.com/angular/protractor/commit/9c9ed31591f5a3c552222ad7feb1ecd650973902)) 
  feat(launcher): allow multicapabilities to take array of promises

  Enables adding `getMultiCapabilities: function(){}` to your configuration file. The function
  returns either multiCapabilities or a promise of a multiCapabilities that is resolved after
  `afterLaunch` and before driver set up. If this is specified, both capabilities and
  multiCapabilities will be ignored.

  Also allows specifying `seleniumAddress` in the capabilities/multiCapabilities object, which will
  override the global `seleniumAddress`. This allows you to use a different `seleniumAddress` per
  capabilities.

  Breaking Changes:
  `capabilities` can no longer be a promise. Use getMultiCapabilities if you need to return a
  promise.
  `seleniumAddress` can no longer be a promise. Likewise, use getMultiCapabilities.

- ([1670384](https://github.com/angular/protractor/commit/167038499aacfd5def03472f9f548529b273e1e0)) 
  feat(runner): allow protractor to restart browser between tests

  Enables adding `restartBrowserBetweenTests: true` to your configuration file. Note that this will
  slow down test suites considerably. Closes #1435

- ([56beb24](https://github.com/angular/protractor/commit/56beb24b9473ceedc491f3ca00fbce1bb9a18f29)) 
  feat(protractor): add browser.getRegisteredMockModules()

  Now `browser.getRegisteredMockModules()` returns a list of the functions or strings that have
  been registered as mock modules. For troubleshooting.

  Closes #1434.

- ([5a404c2](https://github.com/angular/protractor/commit/5a404c27326fdb130e5d4ac5c4704b4013c78853)) 
  feat(timeline): add timeline plugin

  This plugin gathers test timeline information from the protractor test process, the selenium
  client logs (if available), and sauce labs (if available), and presents the output visually. This
  improves understanding of where latency issues are in tests. See #674

  Usage:

  Add the plugin to your configuration file:

  ```js
  exports.config = {
   plugins: [{
     path: 'node_modules/protractor/plugins/timeline/index.js',

      // Output json and html will go in this folder.
     outdir: 'timelines',

      // Optional - if sauceUser and sauceKey are specified, logs from
     // SauceLabs will also be parsed after test invocation.
       sauceUser: 'Jane',
       sauceKey: 'abcdefg'
     }],
   // other configuration settings
  };
  ```

- ([a9d83f7](https://github.com/angular/protractor/commit/a9d83f7ebbce1be7f7f8c2986d1bfebccff1d6f3)) 
  feat(plugins): add postResults hook for plugins

  Allows plugins to include a postResults function, which will be called after webdriver has been
  quit and the environment has been torn down. This step may not modify the contents of the test
  results object.

## Dependency Version Upgrades

- ([2b4ac07](https://github.com/angular/protractor/commit/2b4ac07eaccafec2ad88c05747a75268a3529759)) 
  feat(webdriver): version bumps for chromedriver and supported browsers

  Chromedriver to 2.13. CI browser version bumps for Chrome 39 and Firefox 34.


## Bug Fixes

- ([adf30ba](https://github.com/angular/protractor/commit/adf30ba701d2a1ec992912001723de19366bea57)) 
  fix(test): use a platform agnostic way to run minijasminenode

- ([50ee0b4](https://github.com/angular/protractor/commit/50ee0b4d1a1b93cedf3d099d349b937b25ee9e79)) 
  fix(test): allow to run 'npm start' or 'npm test' from windows too

- ([b28355d](https://github.com/angular/protractor/commit/b28355dabde4c507ac620b973104e98e96279f2a)) 
  fix(cucumber): emit on cucumber scenario instead of step

- ([33dcd77](https://github.com/angular/protractor/commit/33dcd777fe34c6682b64bda0adc4f3595b03394b)) 
  fix(util): webdriver could deadlock

  when prepare scripts containing promises are wrapped in a flow.execute

- ([a877268](https://github.com/angular/protractor/commit/a877268f35cb0df8f34f60b71ad7201fef58d189)) 
  fix(locators): ng-repeat-start should not return extra null element

- ([d505249](https://github.com/angular/protractor/commit/d505249fff773d0eaee8b17435ab751be8fbefa6)) 
  fix(waitforangular): improve error messages when waitForAngular fails

  Previously, caught errors were being interpreted as an empty object, causing lots of errors such
  as
  'Uncaught exception: Error while waiting for Protractor to sync with the page: {}' Now the error
  message will be displayed, and a more useful custom message will be thrown if the variable
  'angular' is not present or the root element is not part of the ng-app.

  See #1474

## Breaking Changes

- Due to ([9c9ed31](https://github.com/angular/protractor/commit/9c9ed31591f5a3c552222ad7feb1ecd650973902)) 
  feat(launcher): allow multicapabilities to take array of promises

  Breaking Changes:
  `capabilities` can no longer be a promise. Use getMultiCapabilities if you need to return a
  promise.
  `seleniumAddress` can no longer be a promise. Likewise, use getMultiCapabilities.

  Why is this breaking change not causing a major version bump? This feature was
  not fully supported previously and we worked with all known users when making
  the change.


# 1.5.0

## Features

- ([55a91ea](https://github.com/angular/protractor/commit/55a91ea137395891248db148df75dd6408c3b3a2)) 
  feat(launcher): reorganize launcher + add option to store test results as JSON

  You may now use `config.resultJsonOutputFile` to specify a location for
  output. See docs/referenceConf.js for more usage.

- ([6a88642](https://github.com/angular/protractor/commit/6a886425a11b28fce83b6eec1f52296c4f78b7f0)) 
  feat(plugins): basic tools for adding plugins

- ([2572feb](https://github.com/angular/protractor/commit/2572febe2c607d459a21e2ba99a1dcece2083d2d)) 
  feat(plugin): ngHint plugin

  For information on usage, see `plugins/ngHintPlugin.js`. More documentation
  on plugins will be added soon.

- ([0bbfd2b](https://github.com/angular/protractor/commit/0bbfd2b6d38392938781d846ad37b5a0fd964004)) 
  feat(protractor/runner): allow multiple browser in test

  Closes https://github.com/angular/protractor/issues/381
  Usage: `browser.forkNewDriverInstance`.

- ([8b5ae8b](https://github.com/angular/protractor/commit/8b5ae8ba3d2b3f1de75c0add91694e39e9c591a8)) 
  feat(troubleshoot): Add more information when the --troubleshoot flag is used

  Improve error messages and add debug info when
  - the configuration file cannot be parsed
  - a webdriver session cannot be started
  - more than one element is found using `element`

  Unify format used for warnings and errors.

## Bug Fixes

- ([30023f2](https://github.com/angular/protractor/commit/30023f2689171bc4f51a173d9cfd62a18fe276c5)) 
  fix(runner): setTestPreparer does not work

  setTestPreparer would always set the testPrepare to config.onprepare during
  `runner.run()`. This is breaking for code that relies on setTestPreparer directly.

- ([47f12ba](https://github.com/angular/protractor/commit/47f12ba31754346062a1e1d20380346a1c7a0659)) 
  fix(clientsidescripts): make findByCssContainingText tolerate elements with no
  textContent/innerText

- ([6a9b87c](https://github.com/angular/protractor/commit/6a9b87cac9b85cde6ae464eafe4abbba27e4fe4f)) 
  fix(elementexplorer): eval always treat result as promise

- ([289dbb9](https://github.com/angular/protractor/commit/289dbb91a0676add40c12bb85d134904c57dcefd)) 
  fix(util): properly handle exceptions from onPrepare and onExit

- ([a132fac](https://github.com/angular/protractor/commit/a132fac0afed5dc5fe8e2663e5aa1c1a90586920)) 
  fix(jasmine): fix errors when iit was used

  Errors were due to Jasmine not calling reportSpecStarting when iit was used, but calling
  reportSpecResults.

  Closes #1602

## Breaking Changes

- ([0bbfd2b](https://github.com/angular/protractor/commit/0bbfd2b6d38392938781d846ad37b5a0fd964004)) 
  feat(protractor/runner): allow multiple browser in test

  `protractor.getInstance()` had been unused (replaced by global `browser` in v0.12.0)
  and is now removed.

  Before:
  ```js
  var myBrowser2 = protractor.getInstance();
  ```

  After:
  ```js
  // In normal tests, just use the exported global browser
  var myBrowser2 = browser;
  ```

  If you are creating your own instance of the Protractor class, you may still
  use `protractor.wrapDriver` as before.


# 1.4.0

## Features

- ([adef9b2](https://github.com/angular/protractor/commit/adef9b208fcba2a9d60347bda38a3fe3fac6bf50)) 
  feat(runner): add a new method of getting browser drivers - directConnect

  directConnect as an option on the configuration will replace chromeOnly. Now, WebDriverJS allows
  Firefox to be used directly as well, so directConnect will work for Chrome and Firefox, and throw
  an error if another browser is used.

  This change deprecates but does not remove the chromeOnly option.

- ([0626963](https://github.com/angular/protractor/commit/06269636f52f9b3a9c73beb6191ae89a7a376cfb)) 
  feat(config): Option to exclude test for specific capability

  Add the option to exclude spec files for a specific capability. This way you can ignore spec
  files for one capability only. For example if the test is known to fail in the capability.

  Closes #1230

- ([710cad7](https://github.com/angular/protractor/commit/710cad7c5a2d838a0c4184defa1b7d4240f577f6)) 
  feat(runner/frameworks): Change interface contract of the protractor runner instance so that it
  returns a promise instead of calling a callback function

- ([50f44f4](https://github.com/angular/protractor/commit/50f44f430851cbd76dbb3a41d6071198f6f479a4)) 
  feat(protractor): add clone methods for ElementFinder and ElementArrayFinder

- ([eedf50b](https://github.com/angular/protractor/commit/eedf50b48ca55f18e8555ce5aa64ad92b03887c8)) 
  feat(launcher): add beforeLaunch and afterLaunch

- ([8dd60b7](https://github.com/angular/protractor/commit/8dd60b73a3013bd29213c8d281819da6e545c7ff)) 
  feat(protractor): wrap negative indices for ElementArrayFinder.get(i)

  Closes #1213

- ([be236e7](https://github.com/angular/protractor/commit/be236e7f44c5306df36b62bb21bb3ba940c86944)) 
  feat(debugging): use custom messages when executing scripts to improve stack traces

  Now, instead of asynchronous events during executeScript all being described as
  `WebDriver.executeScript`, they have their own custom messages. The schedule shown when debugging
  will be more informative.

## Dependency Version Upgrades

- ([889a5a7](https://github.com/angular/protractor/commit/889a5a70c1f980d09a615cf1e8ceaea33272ba8e)) 
  feat(webdriver): version bumps for webdriver, chromedriver, webdriverJS

  Upgrade to WebDriver 2.44.0 and ChromeDriver 2.12.

## Bug Fixes

- ([2fbaf52](https://github.com/angular/protractor/commit/2fbaf52fd59f03929e173ebf760a97de34bf91d4)) 
  fix(element): use the root element only to find the testability API, not scope searches

  In 9a8f45a a change was introduced which made Protractor's custom locators (by.binding, by.model,
  etc) use config.rootElement as the root for all their searches. This meant that
  config.rootElement was used both to specify how to get hold of Angular's injector as well as
  where to begin searching for elements. This does not work for all cases, for example if a dialog 
  should be searched for elements but is a sibling, not a child, of ng-app.

  This reverts that change, and uses document as the parent for all searches. This is consistent
  with the behavior of the native locators by.id, by.css, and friends, which do not scope their
  search based on config.rootElement.

- ([9db5327](https://github.com/angular/protractor/commit/9db5327e4ada7eb3caa271b394bcda0ba5e8fd62)) 
  fix(ElementFinder): ElementFinder should allow null as success handler. Passes the value to the
  next in the chain.

- ([0858280](https://github.com/angular/protractor/commit/0858280db156f924ef126c3aaeae6973b8d44067)) 
  fix(locators): by.cssContainingText now operates on pre-transformed text

  Previously, the implementation depended on the browser. Now, it will always operate on the text
  before text-transform is applied. Closes #1217

- ([1a4eea4](https://github.com/angular/protractor/commit/1a4eea4eb89362822dc86be6904c1ddfba95661e)) 
  fix(elementexplorer): elementexplorer hangs when returning ElementFinder

- ([f4e6b40](https://github.com/angular/protractor/commit/f4e6b40c597dc1c59dc7eccfe236abcc336a46a9)) 
  fix(runner): webdriver could get into lock when there is async promise

- ([cf284b9](https://github.com/angular/protractor/commit/cf284b994fb6766c8ab34d0af9b4ccf8fd866bd1)) 
  fix(clientsidescripts): by.exactBinding not working because of regex typo

  Closes #1441

- ([9cc0f63](https://github.com/angular/protractor/commit/9cc0f6398146ed9bfc757c1efc05d1806bab1e16)) 
  fix(runner): gracefully shutdown browsers after test

- ([86ead2c](https://github.com/angular/protractor/commit/86ead2c5a20d474e59c3b9796b5438dc2090a6ed)) 
  fix(webdriver-manager): Avoid incompatibility between request with callback and pipe.

- ([7283fdf](https://github.com/angular/protractor/commit/7283fdfa1e4c69bcab6af8f28f8f1b77634a50fd)) 
  fix(launcher): exit code is always 100 for sharded and 1 for nonsharded tests

# 1.3.1

## Bug Fixes

- ([714e4e2](https://github.com/angular/protractor/commit/714e4e28ab90fb5dfeca4375a68469ef609e2722)) 
  fix(locators): fix regression passing root element to locator scripts

  Closes #1378

# 1.3.0

## Features

- ([4f1fe68](https://github.com/angular/protractor/commit/4f1fe68882dedba662752e722b9e7b76bfed19b6)) 
  feat(runner): Allow onCleanup to accept a file

- ([548f0c0](https://github.com/angular/protractor/commit/548f0c09748502cb6ae87e602db09e6df78df348)) 
  feat(webdriver): bump WebDriver to version 2.43

- ([466b383](https://github.com/angular/protractor/commit/466b3831569dc28c5fc2be31fbdf96574e57c3f0)) 
  feat(protractor): allow advanced features for ElementArrayFinder

  changed ElementFinder as a subset of an ElementArrayFinder.

  This enables actions on ElementArrayFinders, such as:
  `element.all(by.css('.foo')).click()`

  The function `filter` now returns an ElementArrayFinder, so you may also do:
  `element.all(by.css('.foo')).filter(filterFn).click()`

  or

  `element.all(by.css('.foo')).filter(filterFn).last().click()`

- ([7bd2dde](https://github.com/angular/protractor/commit/7bd2dde0a6fca8c8481ad68d0683b4f411d611b9)) 
  chore(angular): upgrade angular to version 1.3.

  This change updates Protractor's test application from 1.2.9 to 1.3.0-r0.

  There is a significant behind-the-scenes change in the implementation of locating elements and
  waiting for the page to be stable. If you are updating your application to Angular 1.3, you may
  run into some changes you will need to make in your tests:

   - `by.binding` no longer allows using the surrounding `{{}}`. Previously, these
     were optional.
     Before: `var el = element(by.binding('{{foo}}'))`
     After: `var el = element(by.binding('foo'))`

   - Prefixes `ng_` and `x-ng-` are no longer allowed for models. Use `ng-model`.

   - `by.repeater` cannot find elements by row and column which are not children
    of the row. For example, if your template is
    `<div ng-repeat="foo in foos">{{foo.name}}</div>`
    Before: `var el = element(by.repeater('foo in foos').row(2).column('foo.name'))`
    After: You may either enclose `{{foo.name}}` in a child element or simply use:
    `var el = element(by.repeater('foo in foos').row(2))`

- ([ee82f9e](https://github.com/angular/protractor/commit/ee82f9e3d0656b3c88f041f0115743352bc08941)) 
  feat(webdriver-manager): ignore ssl checks with --ignore_ssl option

  Allow ability to ignore SSL checks when downloading webdriver binaries. Usage: `webdriver-manager
  update --ignore_ssl`

## Bug Fixes

- ([838f5a2](https://github.com/angular/protractor/commit/838f5a2b248b1539b7ece13a8ccb921eda08ee45)) 
  fix(element): isPresent should not throw on chained finders

  Now, `$('nonexistant').$('foo').isPresent()` will return false instead of throwing an error. This
  change also adds tests that ensure that catching errors from promises works as expected.

## Breaking Changes

- ([f7c3c37](https://github.com/angular/protractor/commit/f7c3c370a239218f6143a4992b1fc4763f4cdd3d)) 
  feat(webdriver): update to WebDriverJS 2.43.5

  Breaking Changes WebDriverJS has introduced changes in the way that Promises are handled in
  version 2.43. See
  https://github.com/SeleniumHQ/selenium/blob/master/javascript/node/selenium-webdriver/CHANGES.md
  - `webdriver.WebElement` has now been split into `webdriver.WebElementPromise`
    and `webdriver.WebElement` so that it does not resolve to itself. This change
    should be largely transparent to users.
  - `WebElement.toWireValue` has been removed.


# 1.2.0

## Features

- ([830f511](https://github.com/angular/protractor/commit/830f51128d1ca6c8858c99617b2752172044a752)) 
  feat(protractor): allow file:// baseUrls

  Modified protractor to support testing node-webkit by using string concatenation vs url.resolve()
  when the baseUrl begins with file://

  Closes #1266.

- ([71b9c97](https://github.com/angular/protractor/commit/71b9c97432316a8409c7c83e28a3b1eba2d83f25)) 
  feat(cucumber): process the Cucumber 'coffee' param

## Bug Fixes

- ([ade9a92](https://github.com/angular/protractor/commit/ade9a9277558a564e15e46266a82aeb43261d958)) 
  fix(webdriver-manager): always use https for downloading webdriver binaries

  This fixes issues with unzipping - see #1259

- ([9a8f45a](https://github.com/angular/protractor/commit/9a8f45af49633f1637c88960ba079d7d425ca72c)) 
  fix(locators): locators should use the root element provided in config

  Previously, locators used 'document' as the root for their search. After this change, they will
  use the root element provided in the config file -
  `config.rootElement`. This will make sure behavior is correct if there are multiple angular apps
  on one page, and also enables the getTestability path, because that requires a root element under
  an ng-app.

# 1.1.1
This is a minor release with no functional changes. It contains a couple
implementation switches that new versions of Angular will use.

# 1.1.0

## Features

- ([316961c](https://github.com/angular/protractor/commit/316961c6a5d7410d73a2784a9622b106008b0930)) 
  feat(runner/hosted): add support for promises for seleniumAddress and capabilities

  Change driverProviders/hosted to resolve promise values in configuration to allow async jobs in
  setup. Specifically, seleniumAddress, capabilities, and multiCapabilities may be promises.
  Primarily, this would be for a network call to acquire a selenium host or to start a proxy
  server.

- ([953faf7](https://github.com/angular/protractor/commit/953faf7ebee345f686bfedff61ebcb29c5170083)) 
  feat(runner): allow onPrepare functions to return a promise

  If onPrepare is a function which returns a promise (or a file which exports a promise), the test
  runner will now wait for that promise to be fulfilled before starting tests.

- ([6de2e32](https://github.com/angular/protractor/commit/6de2e32328fc30b43428973457db08f34b7c1839)) 
  feat(runner): Add support for async onCleanUp functions

  If the onCleanUp function returns a promise, the process will allow it to resolve before exiting.
  This is useful for performing async operations like writing to a file or calling an API at the
  end of a test run.

- ([cd575ee](https://github.com/angular/protractor/commit/cd575ee3a4d8c0930db23ad66649bf0d665ce2d6)) 
  feat(sauce provider): allow for custom server addresses when running against SauceLabs.

  Use `config.sauceSeleniumAddress` to connect to a custom URL for Sauce Labs.

- ([1b16c26](https://github.com/angular/protractor/commit/1b16c26ac143910d3f3e92a3db4ac6ab168a8544)) 
  feat(suites): allow more than one suite from the command line

  Allow a comma-separated list of suites be provided on the command line, like
  `--suite=suite1,suite2`

- ([25cf88c](https://github.com/angular/protractor/commit/25cf88c29449cef6b925d19ec9cd17671f1befc9)) 
  feat(ElementArrayFinder): keep a reference to the original locator

## Bug Fixes

- ([d15d35a](https://github.com/angular/protractor/commit/d15d35a82a5a267bb7ae9c675017f091901c019f)) 
  fix issue where ElementFinder.then does not return a promise

  See https://github.com/angular/protractor/issues/1152

- ([9e36584](https://github.com/angular/protractor/commit/9e365848820a9a56547e884592e5ea13ef8460ea)) 
  fix(webdriver-manager): removed ssl on chromedriver url for consistency

  Other URLs use http, make chromedriver use this as well.

- ([0da1e0c](https://github.com/angular/protractor/commit/0da1e0c65ba7a2b55ad2f5a4582e229dd876f940)) 
  fix(protractor): add dummy isPending function

  See https://github.com/angular/protractor/issues/1021

- ([9814af1](https://github.com/angular/protractor/commit/9814af11f35973f0b4a3325fcd0d9e0d91233e61)) 
  fix issue where color formatting text is leaking

  See https://github.com/angular/protractor/issues/1131

- ([8f1b447](https://github.com/angular/protractor/commit/8f1b4472430ec2d24f102d284e807b073d17ad81)) 
  fix(launcher): fix issue where test passes on unexpected failures


# 1.0.0

No changes from rc6.

# 1.0.0-rc6

## Dependency Version Upgrades

- ([b6ab644](https://github.com/angular/protractor/commit/b6ab644dd8105d3f64e347342a0ae2ad2f0100fc)) 
  chore(jasminewd): update to version 1.0.4

  This version contains a fix for too many timeout messages.

## Bug Fixes

- ([0c4a70e](https://github.com/angular/protractor/commit/0c4a70e0ffbbf4373dbd9f1ab29daabe9338d57b)) 
  fix(protractor) fix stack traces for WebElement errors

  When angular/protractor@3c0e727136ab3d397c1a9a2bb02692d0aeb9be40 refactored `element()` into the
  ElementFinder object, the function lost some of its error handling.  This removed references to
  frames inside tests (`it()` blocks), making it hard to tell where the error was actually
  occurring.

  This commit fixes these problems, showing full stack traces for WebElement errors.

# 1.0.0-rc5

## Features

- ([51a5e89](https://github.com/angular/protractor/commit/51a5e89f7dace45e61d8eab70e1ea6e9354d4de6)) 
  feat(config): allow setting the get page timeout globally from the config

  To change the timeout for how long a page is allowed to stall on `browser.get`, change
  `getPageTimeout: timeout_in_millis` in the configuration. As before, you may also change the
  timeout for one particular `get` call by using a second parameter:
  `browser.get(url, timeout_in_sec)`

## Bug Fixes

- ([985ff27](https://github.com/angular/protractor/commit/985ff27c9a94cca83af8db5bf7e570d826b23838)) 
  fix(configParser): load new functions from configs

  Closes #1043

## Breaking Changes

- ([51a5e89](https://github.com/angular/protractor/commit/51a5e89f7dace45e61d8eab70e1ea6e9354d4de6)) 
  feat(config): allow setting the get page timeout globally from the config

  This change contains a small breaking change for consistency. Previously, the second parameter to
  `get` changed the timeout in seconds. Now, the units are milliseconds. This is consistent with
  all the other timeouts, as well as base JavaScript functions like setTimeout.

   - before: `browser.get(url, 4)`
  - after: `browser.get(url, 4000)`

# 1.0.0-rc4

## Bug Fixes

- ([ab1d0be](https://github.com/angular/protractor/commit/ab1d0be8cd83b37906b9b8750dd9d85d72))
  fix(navigation): fix using browser.get with safari driver

  SafariDriver fails with data urls - see #1049. Reverting to use about:blank for now.

# 1.0.0-rc3

## Features

- ([f0e7984](https://github.com/angular/protractor/commit/f0e7984cdd169df947142c1cff0bd1bc33ac995b)) 
  feat(launcher): append capability tag for all output

## Bug Fixes

- ([1198dde](https://github.com/angular/protractor/commit/1198ddef9e353383819fca3a40bdaba0db22f96f)) 
  fix(navigation): use empty html data urls for page resets instead of about:blank

  Except on internet explorer, which does not allow data urls.

  Closes #1023.

# 1.0.0-rc2

## Dependency Version Updates

- ([e10e1a4](https://github.com/angular/protractor/commit/e10e1a4a8ae5013982f00d209e6fab1ff2b1d2a1)) 
  chore(minijasminenode): update minijasminenode dependency to v1.1.0

  This adds several options for the reporter, which can be included in protractor's
  `config.jasmineNodeOpts`
  ```js
  // If true, output nothing to the terminal. Overrides other printing options.
  silent: false,
  // If true, print timestamps for failures
  showTiming: true,
  // Print failures in real time.
  realtimeFailure: false
  ```

- ([be0bb00](https://github.com/angular/protractor/commit/be0bb00db6f51e381e31e80c6808a202270ecb20)) 
  chore(jasminewd): update jasminewd to v1.0.3

  This fixes extra logging when a timeout occurs.

## Features

- ([82c1d47](https://github.com/angular/protractor/commit/82c1d47462779688bb8c9ac74ba3a6ecfefb7775)) 
  feat(protractor): add iteration index to ElementArrayFinder.each

- ([62bcf7e](https://github.com/angular/protractor/commit/62bcf7e1c84ed720bc17435c40e1f78c50ba194c)) 
  feat(webdriver-manager): minor proxy enhancements

  Added error handling for request - previously, any errors coming from the request module were
  silently swallowed.

  Fixed error handling to remove empty files when a download fails for some reason.

  Also evaluating both uppercase and lowercase proxy variables. Many tools use proxy variables in
  the form https_proxy, others use HTTPS_PROXY.

## Bug Fixes

- ([dbf7ab5](https://github.com/angular/protractor/commit/dbf7ab5fdf7832676f37328e2ad96b9097db3f62)) 
  fix(mocha): mocha globals should be re-wrapped for every new suite

  Closes #523, closes #962


# 1.0.0-rc1

## Dependency Version Updates

- ([0dc0421](https://github.com/angular/protractor/commit/0dc04217a6a5b772d42b1463c91d89beca7df258)) 
  chore(selenium): version bumps to selenium 2.42.2

## Features

- ([6906c93](https://github.com/angular/protractor/commit/6906c9326a4a83d81a0d09bdc1446cccb579d699)) 
  feat(webdriver-manager): use proxy for webdriver-manager

- ([7d90880](https://github.com/angular/protractor/commit/7d9088025c5a1c37428ea3f1cee3dc8d3793f79e)) 
  feat(locators): implement by.options

- ([4e1cfe5](https://github.com/angular/protractor/commit/4e1cfe5ad0f22947d21b4ebecd7cd05e0319af1a)) 
  feature(launcher): aggregate failures at the end and output message from the launcher

- ([ff3d5eb](https://github.com/angular/protractor/commit/ff3d5ebc071a8806259f5da20018f2d937409455)) 
  feat(locators): add toString() wrapper for this.message

- ([c892c2a](https://github.com/angular/protractor/commit/c892c2a1a773cc24cc6565efe2db892776143104)) 
  feat(protractor): implement reduce and filter for ElementArrayFinder

  See https://github.com/angular/protractor/issues/877

- ([8920028](https://github.com/angular/protractor/commit/8920028f42e683dc45e18a6dd9386bd862548010)) 
  feat(pause): allow the user to specify a port to be used for debugging

  Using browser.pause(portNumber) will now start the debugger on the specified port number.

  Closes #956

## Bug Fixes

- ([f9082d0](https://github.com/angular/protractor/commit/f9082d0460c7b6465d53c37f326a0f66412c21ce)) 
  fix(clientsidescripts): make exactBinding more exact

  See https://github.com/angular/protractor/issues/925

- ([6641c81](https://github.com/angular/protractor/commit/6641c8168d74914d4826c5968771a2aec8299833)) 
  fix(launcher): report summary when specs fail

- ([36e0e0a](https://github.com/angular/protractor/commit/36e0e0aaf090b0c9b5450fa59ba2da4c4237442a)) 
  fix(protractor): allow exceptions from actions to be catchable

  See https://github.com/angular/protractor/issues/959

- ([e86eb72](https://github.com/angular/protractor/commit/e86eb726ad20737d463341afdb4c56b4d19ef414)) 
  fix(protractor): removing a mock module that was never added now is a noop

  It used to remove the last module - now is a noop.

  Closes #764

- ([bf26f76](https://github.com/angular/protractor/commit/bf26f76ba5dc99d02ea4cd7fc198dce410a9f58c)) 
  fix(locators): findind elements by text should trim whitespace

  WebDriver always trims whitespace from around the text of an element, so to be consistent we
  should trim the text from button elements before doing a by.buttonText.

  Closes #903, Closes #904.

- ([48798b0](https://github.com/angular/protractor/commit/48798b0a8ac1fc56d0cdd80e177d67fdf592069c)) 
  fix(elementexplorer): element.all hangs in interactive mode

# 0.24.2
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Bug Fixes

- ([a43f983](https://github.com/angular/protractor/commit/a43f98391d36cead7378d1dd26f54248f39300b7)) 
  fix(protractor): make ElementFinder.then resolve to itself instead of null

- ([31d42a3](https://github.com/angular/protractor/commit/31d42a3875c5b95893d8a20d00dc5365c289ff98)) 
  fix(protractor): throw index-out-of-bounds

  See https://github.com/angular/protractor/issues/915
  - to make error more specific instead of propagate later

# 0.24.1
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Bug Fixes

- ([59af936](https://github.com/angular/protractor/commit/59af936e1ef6e21432f7876144554db145083d46)) 
  fix(locators): Missing information in warning/error messages

  Webdriver's built-in locators (such as `by.css()`) appeared as
  'undefined' in protractor's messages.

  For instance, if a locator matched multiple elements, protractor would print the following
  message: 'warning: more than one element found for locator undefined- you may need to be more
  specific'.

- ([13373f5](https://github.com/angular/protractor/commit/13373f5de18690e1994b32e092105cfe3ad1753d)) 
  fix(launcher): output error messages when child processes exit with error

  Version 0.24.0 introduced a bug where child processes would error without outputting the error
  message. Fix. See #902.

- ([72668fe](https://github.com/angular/protractor/commit/72668fe5ebbdc8126ff16887814f763198128ab5)) 
  fix(cssShortcut): fix $$ global throwing error

# 0.24.0
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([7299155](https://github.com/angular/protractor/commit/729915554cfa440bda0eec8a1c4bf423f4089481)) 
  feat(sauceprovider): append spec filename to capabilities.name

- ([f22456d](https://github.com/angular/protractor/commit/f22456d3cf0768a577371776d716b8888a74397d)) 
  refactor(jasminewd): use jasminewd from its own node module

  The Jasmine Webdriver Adapter is now its own npm module. The code has been moved to
  http://www.github.com/angular/jasminewd.

  Remove the code from Protractor, and add a dependency on jasminewd@1.0.0.

- ([f23565d](https://github.com/angular/protractor/commit/f23565d5db4fbb102cfec8311ce9dfaee52e9113)) 
  feat(protractor): new API allowAnimations(bool) on protractor elements.

- ([876a3c0](https://github.com/angular/protractor/commit/876a3c04c07a9f8d97e1edca3ec1f76e51e1a310)) 
  feat(runner): support running dart2js spec files

  This commit supports running Dart2JS output in NodeJS.  Officially, Dart2JS in supposed to only
  generate code for running in a real webbrowser.  With this patch, the dart2js code can also be
  executed in NodeJS.

  Ref:
  https://code.google.com/p/dart/source/browse/branches/bleeding_edge/dart/sdk/lib/js/dart2js/js_dart2js.dart?spec=svn32943&r=32943#487

- ([8d46e21](https://github.com/angular/protractor/commit/8d46e210b91ed1521f6692a2cf35f60740c0ace6)) 
  feat(runner): support sourcemaps in spec files

  This feature allows folks who are generating their spec files from a different language to see
  stack traces that use the line numbers from their sources before translation.

  This commit introduces a dependency on the `source-map-support` library.

  For general information about sourcemaps, refer:
  -  http://www.html5rocks.com/en/tutorials/developertools/sourcemaps/
  -  https://github.com/evanw/node-source-map-support
  -  https://docs.google.com/document/d/1U1RGAehQwRypUTovF1KRlpiOFze0b-_2gc6fAH0KY0k/view

## Bug Fixes

- ([56daa54](https://github.com/angular/protractor/commit/56daa54e2e269064bd44bc05ed0bbf2c44657ca8)) 
  fix(clientsidescripts): convert non-Error exceptions to Errors

  If any functions called by clientSideScripts throws a an exception that doesn't inherit from
  `Error`, the stack trace is completely unhelpful and the message is just "unknown error."  This
  commit wraps such errors into
  `Error` instances so that we have meaningful stack traces and the correct exception message. 
  (e.g. This is the common case when running dart2js code.  This commit gives us the Dart stack
  trace and exception message.)

  In addition, I've pushed the construction of the string to install into the browser into
  clientsidescripts.js.

- ([00c6abe](https://github.com/angular/protractor/commit/00c6abef16c47868974eed8ad1a4c38494b2a504)) 
  fix(element): fix WebElement.$ using the incorrect By object

  Closes #852

- ([0500b2c](https://github.com/angular/protractor/commit/0500b2c3b2698fe41bedf694b92aad884f3b0d0e)) 
  fix(navigation): ignore unknown JS errors when looking for the URL

  This should address #841

  Ignoring the error and trying again has worked for all of my test cases, and the error has never
  occurred more than once in a row.

- ([c8c85e0](https://github.com/angular/protractor/commit/c8c85e0d94d7a7211b000650f01af714663611ad)) 
  fix(locators): fix by.repeater finding all rows for IE

  Previously, element.all(by.repeater('foo in foos')) would find non-element nodes for
  ng-repeat-start elements, which could cause IEDriver to fall over if the test tried to get text
  from those nodes.

## Breaking Changes

- ([3c0e727](https://github.com/angular/protractor/commit/3c0e727136ab3d397c1a9a2bb02692d0aeb9be40)) 
  refactor(protractor): reorganize internal structure of elementFinder/webelement

  - Allow chaining of actions (i.e. `element(By.x).clear().sendKeys('abc')`)
  - first(), last(), and get(index) are not executed immediately, allowing
      them to be placed in page objects
  - Rework the way that elementFinder and wrappedWebElement is represented
  - Breaking changes:
    - element.all is chained differently
        ```
        Before: element(By.x).element.all(By.y)
        Now:    element(By.x).all(By.y)
        
        However, using element.all without chaining did not change,
          i.e. `element.all(By.x)`
        ```

    - Changed the way for retrieving underlying webElements
        ```
        Before: element(By.x).find(), element(By.x).findElement(By.y),
                  and element(By.x).findElements(By.y)
        Now:    element(By.x).getWebElement(),
                  element(By.x).element(By.y).getWebElement(),
                  and element(By.x).element(By.y).getWebElements(),
                  respectively
        ```
    - browser.findElement returns a raw WebElement so $, $$, and
        evaluate will no longer be available

- ([fbfc72b](https://github.com/angular/protractor/commit/fbfc72bad15667990232bb9ff1da503e03d16230)) 
  feat(launcher): Add support for maxSession

  - add support for maxSession and capability-specific specs
  - cleaned up launcher (refactored out taskScheduler.js)
  - Breaking change:
    - changed the config to shard test files; also sharding is specific to
  capabilities now
      ```
      Before: config.splitTestsBetweenCapabilities
      Now: config.capabilities.shardTestFiles or config.multiCapabilities[index].shardTestFiles
      ```

- ([9e5d9e4](https://github.com/angular/protractor/commit/9e5d9e4abb7d0928e6092a711fda527554994be7)) 
  feat(locators): remove deprecated locator APIs

  This is a **breaking change**. The following deprecated Locator APIs have been removed.

  - `by.input`
  - `by.select`
  - `by.selectedOption`
  - `by.textarea`

  `input`, `select`, and `textarea` can be replaced by `by.model`.

  `element(by.selectedOption('foo'))` can be replaced by
  `element(by.model('foo')).$('option:checked')`

# 0.23.1
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Bug Fixes

- ([59533d9](https://github.com/angular/protractor/commit/59533d95219796ce18f796434f8c3396ada7402c))
  fix(navigation): revert changes to the page reset

  Navigating to an empty data URL won't work for internet explorer, sadly.

  Reverting to about:blank. Will watch for flakes and explore other options.


# 0.23.0
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([b7afa87](https://github.com/angular/protractor/commit/b7afa8791ba91b83fd6613cdd9ad4c4c26d04f7e)) 
  feat(addMockModule): allow additional parameters

  Allow Protractor’s 'addMockModule' method to pass context to its mocks, providing an argument to
  the script which overrides a module. Rely on the WebDriver’s 'executeScript' method.

  Closes #695

- ([546d41f](https://github.com/angular/protractor/commit/546d41faeb75342c875e0f9bb7702309c1aa186d)) 
  feat(sauceprovider): runner now prints a link to saucelabs test URL

- ([fd7fe4a](https://github.com/angular/protractor/commit/fd7fe4a8c2c6fab6678d0c1f4d5619f7a2376990)) 
  feat(launcher): Add support for splitTestsBetweenCapabilities.

- ([b93bf18](https://github.com/angular/protractor/commit/b93bf18feaf3c44b406a41bf87d70c95e7a900e0)) 
  feat(elementFinder): keep a reference to the original locator

- ([98f4ba5](https://github.com/angular/protractor/commit/98f4ba590207e3f468b3cb2a30ff6ab6ae10fea1)) 
  feat(locators): add by.exactBinding

## Bug Fixes

- ([43ff9e5](https://github.com/angular/protractor/commit/43ff9e5e2a05b4e51d04133122d763ef4ed3f2d1)) 
  fix(jasminewd): allow asynchronous callbacks for jasmine tests

  Closes #728, Closes #704

- ([6249efe](https://github.com/angular/protractor/commit/6249efe57109d238044394636d623e0bd93dd4ad)) 
  fix(webdriver-manager): use request module instead of http

  Google changed selenium-server-standalone.jar's location and is returning 302 http module does
  not follow redirects

  Closes #826

- ([95093c3](https://github.com/angular/protractor/commit/95093c3011431d1a1bdd6ec4d6139a6ff1c3e491)) 
  fix(configParser): don't run suite if specs are supplied

- ([27a5706](https://github.com/angular/protractor/commit/27a5706a23e33bc898a5a9c7b301e79f962e3a7b)) 
  fix(loading): fix timeouts with about:blank removal

  As documented at https://github.com/jnicklas/capybara/pull/1215 there are sometimes issues with
  webdriver and about:blank pages.

  Switching instead to try a data url.

- ([cbcdb48](https://github.com/angular/protractor/commit/cbcdb483002e51bc3cc4061fd5162627bbac7699)) 
  fix(runner): add -r for each cucumber require

- ([e36c32a](https://github.com/angular/protractor/commit/e36c32a975739a99f6d434e1c9844d37382bda3a)) 
  fix(jasminewd): Use promise.all to combine promises and done

  - Make the flow promise explicit and use promise.all to wait for both
   promises to be fulfilled before calling the done callback

- ([b5c18db](https://github.com/angular/protractor/commit/b5c18dbb746e63496809460d6ed6e2100909659e)) 
  fix(drivers): prevent Sauce Labs login credentials from showing up in logs

  Closes #754

- ([b85af50](https://github.com/angular/protractor/commit/b85af5031241d424e2952db0eb8d7d0c8ce4475b)) 
  fix(protractor): change angular-bootstrap wrapper for navigation

- ([8abea3c](https://github.com/angular/protractor/commit/8abea3cbb6f054c20e4f5abcbf61813d5b671239)) 
  fix(jasminewd): fix timeout for beforeEach and afterEach

# 0.22.0
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([8b088fd](https://github.com/angular/protractor/commit/8b088fd6bf83696fd2ad294d8818e20894332693)) 
  feat(locators): Added a By.cssContainingText locator.

  This new locator find elements by css selector and inner text in response to the lack of
  ':contains' selector.

  Example: By.cssContainingText('ul .pet', 'Dog') will find all ul children with class 'pet'
  containing the text 'Dog'.

  Closes #488, Closes #709

- ([54060b7](https://github.com/angular/protractor/commit/54060b7cef4eb2f4c184c360cef7c2eb25c0ff6a)) 
  feat(protractor): add the browser.setLocation method to perform in-page navigation

  Allow a faster way to navigate within the app. The current browser.get method forces the entire
  app to load every time you navigate to a new page. The proposed browser.setLocation method uses
  the same format as $location.url().

  Closes #368

- ([74761e8](https://github.com/angular/protractor/commit/74761e8b25395dd78e1c301ee23a7730fef36db9)) 
  feat(cli): use protractor.conf.js as a default config file if none is passed

  Closes #615

## Chores and updates

- ([b81cf5a](https://github.com/angular/protractor/commit/b81cf5a949dee25c9070491edd1eb9e9feee556f)) 
  chore(webdriver): update WebDriverJS version to 2.41.0

- ([a96df4d](https://github.com/angular/protractor/commit/a96df4d60a1f2e09de865bf7ca9c5c780f945239)) 
  chore(minijasminenode): update to version 0.4.0.

  This allows the use of `because('message')` before expectations, to give additional information
  when a failure occurs.

  It also removes warnings for Node 0.11.* users about util.print being deprecated.

  Closes #377

- ([6f31b56](https://github.com/angular/protractor/commit/6f31b5619de4fdb9b1b6e9a29a62dac09b781c6b)) 
  chore(package): npm start now brings up the testapp

  Closes #712

## Bug Fixes

- ([1137d12](https://github.com/angular/protractor/commit/1137d12b95435438d2b84448796f9fe32d2f87b2)) 
  fix(mocha): fix it.only so that it does not double-wrap

  Closes #469

- ([bde56a0](https://github.com/angular/protractor/commit/bde56a0d92a79570f377490929dd1d05107f4e25)) 
  fix(cli): fix --exclude command line flag

  Accidentally got changed to 'excludes'. As discussed earlier, should be single to be consistent
  with Karma.

  Closes #637

- ([9e426df](https://github.com/angular/protractor/commit/9e426dfd300a11f513c5d7202bbb632f4b1c41d8)) 
  fix(locators): using $().$$() should return an ElementArrayFinder

  Prior, $(foo).$$(bar) would return a promise which resolved to an array of WebElements. This is
  unexpected, since $(foo).$(bar) returns an ElementFinder, and
  element(by.css(foo)).element.all(by.css(bar)) returns an ElementArrayFinder. Fixed so things are
  more consistent.

  Closes #640

- ([b67810a](https://github.com/angular/protractor/commit/b67810a08d19940cd144fea25f08af4478166231)) 
  fix(webdriver-manager): do not download files if HTTP response is not 200

  Closes #656

- ([28912f0](https://github.com/angular/protractor/commit/28912f0a77b44cce19ef5367c92b023388f7ff10)) 
  fix(webdriver-manager): fix download paths

# 0.21.0
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([0c4ef69](https://github.com/angular/protractor/commit/0c4ef69c1f61a2fb41829fa6d0afae3493148eee)) 
  feat(launcher): launcher outputs a final summary of how the browsers did

- ([f1db8b4](https://github.com/angular/protractor/commit/f1db8b438fd154cef241895c01ed902b3f343315)) 
  feat(runner): make runner an event emitter and log passes or failures from the launcher

  Now, for runs with multiple capabilities, the launcher will output '.' or 'F' for each pass or
  fail instead of just '.' for every chunk of data. TODO - complete the event emitter API for the
  Cucumber runner.

- ([f9c4391](https://github.com/angular/protractor/commit/f9c43910021095e1bee1d1074e8788f4b0aee145)) 
  feat(cli+config): allow defining multiple test suites in the config and running them separately
  from the command line.

- ([06bd573](https://github.com/angular/protractor/commit/06bd573cbc2471c719a8504f906468fb672d4097)) 
  feat(pause): add the browser.pause method to enter a webdriver-specific debugger

  Warning: this is still beta, there may be issues. Usage: In test code, insert a `browser.pause()`
  statement. This will stop the test at that point in the webdriver control flow. No need to change
  the command line you use to start the test. Once paused, you can step forward, pausing before
  each webdriver command, and interact with the browser. Exit the debugger to continue the tests.

## Bug Fixes

- ([43aff83](https://github.com/angular/protractor/commit/43aff830bb74aa97fc4704f3aea9ef38feaee1b6)) 
  fix(pageload): Changing how `about:blank` unload waits Also changing `executeScript` script
  comment from `//` to `/**/` format. These two small changes should not affect functionality but 
  make Protractor work with Selendroid.

- ([1334662](https://github.com/angular/protractor/commit/1334662905d8d6b642a294fbf1e97ec3bc371084)) 
  fix(locators): Improve custom locators message

  Increase readability of custom locator message by displaying each argument instead of the
  arguments object.

- ([c9dbbaa](https://github.com/angular/protractor/commit/c9dbbaa94e2b4378bcc2db580dcad637b609a868)) 
  refactor(launcher): skip the child process if only one capability is requested

  Closes #603

- ([26d67a2](https://github.com/angular/protractor/commit/26d67a29a8a12aa52331a1ec4ae8013cf63257f2)) 
  fix(launcher): launcher should report a failure when only one capability is running

- ([9530a0c](https://github.com/angular/protractor/commit/9530a0cab2791cb0350f81eae3f619d68fb620c3)) 
  (fix): Convert test.sh to test.js

  This would enable the tests to be run on both Linux and Windows.

- ([6d85ab4](https://github.com/angular/protractor/commit/6d85ab4b9f3b5824db3307df5aca77a0720dc2e6)) 
  fix(jasminewd): display stack traces in correct order and with WebElement method failure details

- ([8964ac9](https://github.com/angular/protractor/commit/8964ac97cb994eb6cf7cf7ce77b7eb40882e852b)) 
  fix(test): Fixed path of configuration file to pass on windows

- ([99bda1a](https://github.com/angular/protractor/commit/99bda1aa732288f74126c9a77c48dd7cff63531a)) 
  fix(waitForAngular): when timeout overflows, at least pass the negative to error messages

  Closes #622

- ([4fd060a](https://github.com/angular/protractor/commit/4fd060a38faa1f938f880fa52746e1a481a9122d)) 
  fix (element): Allow ElementFinder to be passed to actions directly.

  Previously, do to an action such as drag and drop, one would have to use
  `element(by.foo).find()`. Now, just passing `element(by.foo)` works. For example:

  ```javascript
  browser.actions().doubleClick(element(by.id('mybutton'))).perform();
  ```

- ([b2a4ffc](https://github.com/angular/protractor/commit/b2a4ffced58964826125ea00705e6e257cdb588b)) 
  fix(configParser): always return "this" from addFileConfig

# 0.20.1
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Bug Fixes
- ([17de697](https://github.com/angular/protractor/commit/17de697fe9f64e238a8df0fbc6358b8e578e45f2
  fix(debug): make new debug work on windows too

  Closes #580

# 0.20.0
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([220d793](https://github.com/angular/protractor/commit/220d79372fb93d3b58c5131188b24e48be8176ab)), ([6603a7e](https://github.com/angular/protractor/commit/6603a7e964c8f1632db4790081a71648360cf1f9)) 
  chore(webdriver): update selenium version to 2.40.0 and download location

- ([ad5f3aa](https://github.com/angular/protractor/commit/ad5f3aa77fc3429fcf83f825a14fdb43fd7cc8a7)) 
  feat(jasminewd): allow custom matchers to return promises

  Allow custom jasmine matchers to return a promise which resolves to a boolean and match against
  the resolution of the promise

- ([41feaca](https://github.com/angular/protractor/commit/41feaca58c81fbd578c77424abf745acaf26f84f)) 
  feat(framework.cucumber): Allow multiple tags on cucumber tests.

  Motivation: Support for multiple tags on the cucumber test execution, to be able to filter with
  more complex expressions the scenarios to run.

  How to use:
  ```
  cucumberOpts: {
     tags: '@dev'
  }
  ```

  or

  ```
  cucumberOpts: {
     tags: ['@dev', '~@ignore']
  }
  ```

  More information on tags: https://github.com/cucumber/cucumber/wiki/Tags

## Bug Fixes

- ([2ca6541](https://github.com/angular/protractor/commit/2ca654114a2bf937313ff027583308f87e909892)) 
  fix(debug): make protractor debug work in the new runner/launcher world

  Closes #552

- ([a68627b](https://github.com/angular/protractor/commit/a68627b3581c0551e04460682cfc13f8f91be366)) 
  fix(launcher): command line args should be passed as-is to the runner

  This allows users to continue to use optimist (or other process.argv) processing within their
  tests and grab values from the command line.

  Closes #571.

- ([767c306](https://github.com/angular/protractor/commit/767c306102956ba6015cfe3998affb7e8430f259)), ([02defe3](https://github.com/angular/protractor/commit/02defe360dce41ee6841df9012166d249acfeca0)) 
  fix(jasminewd): include full pre-async-call stack trace in expectation failure message

- ([b6df2cf](https://github.com/angular/protractor/commit/b6df2cfcfd35b31e2e473604b6df9add744c6c2d)) 
  fix(configParser): load coffee and typescript for child processes

  Without loading coffee in configParser.js, child processes which try and load a coffeescript
  config file do not have coffee registered with node's required, and child tests fail.

  Fixes an issue with using coffeescript config files.

- ([64bee25](https://github.com/angular/protractor/commit/64bee252f6df52f9243c0f5d7e40f39bf5407134)) 
  fix(locators): add locator with multiple arguments

  When using a custom locator with multiple arguments, only the first argument was used when 
  calling `webdriver.findElements`.


- ([87b0c7f](https://github.com/angular/protractor/commit/87b0c7f2ecc8befa4fa1ebd5d8238c811a869aff)) 
  fix(debug): display error message when runner fails

# 0.19.0
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features
- ([77393d0](https://github.com/angular/protractor/commit/77393d08343ef16ddc2b8042e187c9d68fe7bf2f)), ([6848180](https://github.com/angular/protractor/commit/68481801d506941ebf00fab71f87be510c7a87ba)), ([cca82ca](https://github.com/angular/protractor/commit/cca82caab6ae444b368eebe040a69967d774737e))
  feat(runner/launcher): major runner updates to allow multiple capabilities

  Adding simultaneous runner capability (grid-style), refactoring launch/runner init system, and
  abstracting out configParser module.

- ([642de06](https://github.com/angular/protractor/commit/642de06e8bbabf82c7b8e0a64a280df5c4daf01c)) 
  feat(protractor): add removeMockModule method

- ([88c339f](https://github.com/angular/protractor/commit/88c339fc1d392717a0a5b8265806934b40158c5f)) 
  feat(runner): add adapter for cucumber.js

  Conflicts:
  lib/runner.js

## Bug Fixes
- ([8924bbc](https://github.com/angular/protractor/commit/8924bbca9e8f04073a29534bf16b0867a1ede7a0)) 
  fix(cli): convert capabilities arguments to dot-notation for WebDriver compatibility

- ([a96d32f](https://github.com/angular/protractor/commit/a96d32f44a92ba9447fc843bc0aca7b91b777635)) 
  fix(webdriver-manager): upcase in IE download url

  The url for the Win32 version of the IEDriverServer is apparently case sensitive: _win32_ vs
  _Win32_

## Breaking Changes
- ([05eb42b](https://github.com/angular/protractor/commit/05eb42bb482c7cb36b48af1a86210afc442aa112)) 
  refactor(locators): moves scope in locators to last argument

  scope defaults to document, and is an optional argument so now be moved to the end. Came up from
  debugging and trying to use window.clientSideScripts.findInputs('username'); which failed.
  Refactored to match original intent.

  BREAKING CHANGE: anything relying on clientsidescripts should no longer pass
     element scope as first argument.

      Before:

      window.clientSideScripts.findInputs(document, 'username');

      After:

      window.clientSideScripts.findInputs('username', document);
      // or simply
      window.clientSideScripts.findInputs('username');

    Also, any custom locators using addLocator will now break since the
    arguments order has changed. To migrate the code follow the example below:

      Before:

      var findMenuItem = function() {
        var domScope = arguments[0];
        var myArg = arguments[1];
        // balh blah blah
      };
      by.addLocator('menuItem', findMenuItem);

      After:

      var findMenuItem = function() {
        var myArg = arguments[0];
        var domScope = arguments[1];
        // balh blah blah
      };
      by.addLocator('menuItem', findMenuItem);

  Closes #497


# 0.18.1
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Bug Fixes

- ([a79aa73](https://github.com/angular/protractor/commit/a79aa73df5df598ccad695af882d23ddaac2c2d9))
  fix(cli): specs was being processed as a string, not a list

  Fixes #495

# 0.18.0
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([e3b1e7c](https://github.com/angular/protractor/commit/e3b1e7cec7af35f2e245ca64e4f94227ecaa1c57)) 
  feat(config): add option to exclude specs based on file patterns

  The config now accepts `exclude`, an array of patterns to exclude.

- ([88a1e58](https://github.com/angular/protractor/commit/88a1e587a40f0e6d978b20fe55160a18e2855493)) 
  Feat(clientSideScripts): Add by.buttonText, by.partialButtonText

  Adds client side JS implementations of by.buttonText and by.partialButtonText, enabling element
  lookup based on innerText.

  Closes #452

- ([8d29c93](https://github.com/angular/protractor/commit/8d29c939766f044d910401e60834769cf8e5e44b)) 
  feat(config): allow LiveScript configuration files

## Bug Fixes

- ([d06d931](https://github.com/angular/protractor/commit/d06d931e1cb2c2bd38c2c50965a6f78690bdc336)) 
  fix(timeouts): fix an obscure cause of firefox timeouts

  Fixes #493

- ([de39e50](https://github.com/angular/protractor/commit/de39e5077d09daaeb885767e968a5cef78c9cac7)) 
  fix(jasminewd): support multi-argument matchers

  Implement support for multi-argument matchers in promise wrapper.

  Closes #477

- ([11c4210](https://github.com/angular/protractor/commit/11c4210fe740771707d5421a4940bdce43d3d33e)) 
  fix(testForAngular): add a message when page load does not complete in time

- ([6ae6261](https://github.com/angular/protractor/commit/6ae626158ee0610b70501af5d57ad4ff379c5ead)) 
  refactor(waitForAngular): improve error messages when timeouts occur

- ([5dd93c2](https://github.com/angular/protractor/commit/5dd93c2397a401011e16271f6472c72037c871b6)) 
  fix(config): allow CoffeeScript 1.7 to be used

  CoffeeScript now requires a register call to be made.

- ([10aec0f](https://github.com/angular/protractor/commit/10aec0ff212987bfdb9ab4011e6cb2f9c646fca2)) 
  fix(pageload): increase wait timeout

  The 300 ms wait caused problems when testing IE on Sauce Labs. It seems way too short.
  "browser.get()" invariably timed out. Increasing it solved our problem.


# 0.17.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([cc4f7b5](https://github.com/angular/protractor/commit/cc4f7b55e1fe46fcef1b8c3ca39d702a32ee6d82)), ([8348803](https://github.com/angular/protractor/commit/834880368115ecade154b3a090e06159667c0c2d)) 
  feat(element): allow chaining of element finders with element().element()...

  Chaining calls to element will now build a scoped element finder. No webdriver functions will be
  called until a method (such as getText) is called on the final element. Example:

      var elem = element(by.id('outer')).element(by.css('inner'));
     browser.get('myPage');
     elem.click();

  Closes #340.

- ([088a581](https://github.com/angular/protractor/commit/088a58150f992a6520da983fc461fec4eac1a0ed)) 
  feat(runner): add a callback for when the tests are done

  Add an onCleanUp callback to be able to hook into when all the tests have been run.

  Conflicts:
  referenceConf.js

- ([66c4774](https://github.com/angular/protractor/commit/66c4774aa18d94d4da81c101b82db4a748cf69a4)) 
  feat(runner): add mocha options to config file

  change lib/runner to allow setting mocha options from config.

- ([092fe1f](https://github.com/angular/protractor/commit/092fe1fc1e7d1b58b786870ff1ce33f95e652d78)), ([3151ca7](https://github.com/angular/protractor/commit/3151ca7daaeeec9f537561b31c6dfd42c678f7bb)) 
  feat(locators): Add map() function to element.all

  Added a map function to element.all to apply a function to each element and return the result of
  the transformation.

  Resolve promises if there is an object that contains multiple promises. Added index as a second
  argument to the map function callback.

  Closes #392

- ([7259614](https://github.com/angular/protractor/commit/7259614a326802b8e7a906346bd9830b92e1514d)), ([0257b5f](https://github.com/angular/protractor/commit/0257b5f225052ab0a075d96811dd56961f9278ae)) 
  feat(config): allow CoffeeScript configuration files

  Require CoffeeScript in the cli file to enable CS configuration and spec files.

  Possibly fixes #38

- ([e7d9e08](https://github.com/angular/protractor/commit/e7d9e081cdc7fcf100e0346b1dcf0f7fdad7d889)) 
  feat(global): export By (== by) on the global for use with coffeescript (or others who prefer it)

## Bug Fixes

- ([a0bd84b](https://github.com/angular/protractor/commit/a0bd84b9a28ec92eccd2784f8b849388985a4480)) 
  fix(pageload): add a wait during protractor.get() to solve unload issues

  Some systems would not wait for the browser unload event to finish before beginning the
  asynchronous script execution.

  Closes #406. Closes #85.

- ([4b053eb](https://github.com/angular/protractor/commit/4b053ebe587d51562d77ca512848be28195ae0cc)) 
  fix(runner): only run selenium with spec files

  Only setup Selenium if there are actual spec files passed in

- ([8e096b9](https://github.com/angular/protractor/commit/8e096b9a91af9c37ab4bf84e100568544351efc8)) 
  fix(Protractor.prototype.get): resolve `baseUrl` before ignoring synchronization

  Fixes issues where setting `ignoreSynchronization = true` ignores the value of `baseUrl`
  entirely.

# 0.16.1
_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

_Note: 0.16.0 was released as a canary - no changelog for it._

## Features

- ([a75fa04](https://github.com/angular/protractor/commit/a75fa04a70f64e0da29b9a0a9100bd60d9ebf93f)) 
  docs(readme): add the travis status widget

- ([478c00a](https://github.com/angular/protractor/commit/478c00a01dc9b93de68983b6ef2dfa55f0b42649)) 
  feat(runner): add beta support for using mocha as your test framework

  This change adds limited support for using mocha as the test framework instead of jasmine. Make
  the switch by using `--framework=mocha` on the command line or adding `framework: 'mocha'` to the
  config. Tests will be run using the BDD interface. The interface is adapted so that tests run
  asynchronously without needing to call `done()`.

  Note that there is currently no support for an assertion framework, so you will need to require
  whichever assertion framework you prefer. This means there is no adapter to make the assertions
  unwrap promises, so you will need to resolve promises yourself and run the assertions afterwards.

- ([3731abf](https://github.com/angular/protractor/commit/3731abf901c4278b4470336c3a58765161b08bcc)) 
  feat(webdriver-manager): add seleniumPort command line option

  Added seleniumPort command line option so that the standalone selenium server can be started with
  the supplied port number as opposed to the default port 4444.
  ```
  $ webdriver-manager start --seleniumPort 4443
  ```

## Bug Fixes

- ([bc18c42](https://github.com/angular/protractor/commit/bc18c42dab6207d111f88ea1f1deefb9bcc28f23)) 
  chore(config): saucelabs requires tunnel identifier to be a string

## Breaking Changes

 - ([478c00a](https://github.com/angular/protractor/commit/478c00a01dc9b93de68983b6ef2dfa55f0b42649)) 
  feat(runner): add beta support for using mocha as your test framework

  To allow the user to customize their framework, the protractor runner will now wait
  until just before `onPrepare` to load the framework. This means that `jasmine` will
  not be available in global until `onPrepare`. For example, this means that requiring
  the jasmine-reporters module must be done inside onPrepare, since that module expects
  jasmine to be available at the time it is loaded.


# 0.15.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([f8d0291](https://github.com/angular/protractor/commit/f8d02910340f54af92a8ed5fdd067fa03ca2cef8)) chore(version): update the version of dependency minijasminenode

  This is notable because in the newer 0.2.6 version of minijasminenode,
  ddescribe and iit are supported. These should be available after running
  an 'npm update'.

- ([6165023](https://github.com/angular/protractor/commit/6165023a9593f4f69fe342761b8b2d75923baf7a)) feat(runner): return a promise from runner.runOnce

  In some cases knowing when the runner has finished is a requirement (e.g. an async grunt task).

- ([d44ef01](https://github.com/angular/protractor/commit/d44ef01c64023b4e3a24a9959740676b691f6074)) feat(debugging): remove webdriver lines from stacktraces by default to improve readability

- ([33fa4a4](https://github.com/angular/protractor/commit/33fa4a43acfbe87f3a4d4c84fa93c5c20b3cca0c)) feat(locators): by model works for anything with a model, not just input

  Notably, by.model will now find selects and textareas.

  Closes #321.

- ([238bb74](https://github.com/angular/protractor/commit/238bb7429572f9a9f6620bf1317690f1ac825960)) feat(ignoresync): ignoreSynchronization now affects the behavior of browser.get

  Now, when ignoring synchronization, calls to browser.get are equivalent to calling
  browser.driver.get.

  Closes #306

- ([30c0ceb](https://github.com/angular/protractor/commit/30c0ceb3e2745d3bcc549f4d4963d9fade132e71)) feat(element) element.all exports an 'each' method

  Usage:
  ```
  element.all(by.model('foo')).each(function(webElement) {
    // Do stuff with webElement.
  });
  ```
  Closes #298

- ([6a73a25](https://github.com/angular/protractor/commit/6a73a25c61a72ef991a604eadae010c90a157266)) feat(by.repeat) by.repeat support for multi ng-repeat

  Make by.repeat (and its column and row friends) work with ng-repeat-start
  and ng-repeat-end elements.

  Closes #366. Closes #182.

## Bug Fixes

- ([50d6fde](https://github.com/angular/protractor/commit/50d6fde25148e24d7ef22be371b04333cdf61e50)) fix(clientSideScripts): bind-template directive shouldn't break bind locators

  Fix "UnknownError: angular.element(...).data(...).$binding[0] is
  undefined" error raised when trying to use "by.binding" locator in any
  element of a page that contains at least one "bind-template" directive.

- ([f8c606b](https://github.com/angular/protractor/commit/f8c606bae7b2f414a67b6349f841881132d9cc97)) fix(webdriver-manager): make sure selenium standalone shuts down nicely

  This addresses selenium server shutdown in two ways
   - the node process will stay open until selenium has exited
   - if the user inputs to STDIN (e.g. press space) selenium will shut down gracefully

- ([e98f71e](https://github.com/angular/protractor/commit/e98f71ebd7778d5c77c41bbecc73e31f1aeca177)) fix(webdriver-manager): fix IEDriver install and running via windows

  Changed the binaries.ie.url function to return the correct URL for the IEDriverServer.
  Created the zip object in the win32 section to be able to decompress IEDriverServer.
  Added a function to normalize a command across OS and spawn it. It allows start the webdriver in win32.

  Seen here:
  https://github.com/yeoman/generator/blob/master/lib/actions/spawn_command.js
  

# 0.14.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([c579a1a](https://github.com/angular/protractor/commit/c579a1a01bae6798a87a5ca8915417775e1b6fb2)), ([f54fd5d](https://github.com/angular/protractor/commit/f54fd5d7c3caa8df319a0115086bb4db8443d856)) feat(webdriver-manager): redo the script to run and install selenium/webdriver

  Breaking Change.
  As outlined in Issue #296, redoing the way the selenium/webdriver
  install and run helper scripts work. Now, the 'webdriver-manager' script
  will be available either locally or globally (depending on how protractor
  was installed). It replaced install_selenium_standalone and the 'start' script
  that was provided after install. Run `webdriver-manager update` to download
  new versions of selected webdriver binaries. Run `webdriver-manager start`
  to start the standalone server. In addition, this fixes issues with running
  the server starter in Windows, and allows automated downloading of the IEDriver.

  Thanks to kurthong and vipper for their PRs with windows fixes, which were
  very useful in preparing this.

- ([a69ebc3](https://github.com/angular/protractor/commit/a69ebc3b783fb7bf42877a658498de90d3d196c3)) feat(runner): use selenium and chromedriver from the default location if nothing else is specified

## Bug Fixes

- ([1fa090c](https://github.com/angular/protractor/commit/1fa090c656cbab55bdbfb101b503b53811b50dff)) fix(runner): merge should override entire arrays, not just parts of them

  Closes #304

- ([a2afb4d](https://github.com/angular/protractor/commit/a2afb4d8399ba980674c79138dd98efb683e9ab9)) fix(element): element.all.get and element.all.first/last should wrap web elements

  Closes #307

- ([f3be172](https://github.com/angular/protractor/commit/f3be1727cf95dea50b597d20c6510e62a605dee2)) fix(runner): running with chromeOnly should try to find chromedriver with .exe extension

  Closes #283

## Breaking Changes

- ([c579a1a](https://github.com/angular/protractor/commit/c579a1a01bae6798a87a5ca8915417775e1b6fb2)) feat(webdriver-manager): redo the script to run and install selenium/webdriver

  Breaking Change.
  Your old selenium/start script will continue to work, but install_selenium_standalone no longer exists.
  To do a clean update, remove the selenium folder. Then run
  `webdriver-manager update`

- ([a1c91a2](https://github.com/angular/protractor/commit/a1c91a29af5c1e1f35744462ca16ef4b33ad6c48)) fix(config): Make all file paths in config files relative to the config file itself

  Breaking Change
  Previously, onPrepare and specs were relative to the location of the config,
  but seleniumServerJar and chromeDriver were relative to the cwd when the
  test was called. If you were calling the tests from somewhere other than
  the same directory as the config location, you will need to change the paths of
  seleniumServerJar and/or chromeDriver.  Closes #222.


# 0.13.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([ce5f494](https://github.com/angular/protractor/commit/ce5f494289c3750b84c6783339a14342a1b74f3d)) feat(element): element.all now has 'first' and 'last' methods

- ([ef61662](https://github.com/angular/protractor/commit/ef6166232186b3385769f63430819a722052cc44)) feat(runner): allow bypassing the selenium standalone server if running only chrome

  Using the config option `chromeOnly` now enables running ChromeDriver directly,
  without going through the Selenium Standalone. The chromedriver binary should be
  available in your PATH, or should be specified with the config option
  `chromeDriver`.

- ([76c094a](https://github.com/angular/protractor/commit/76c094a3fa69511b0311011b0ef2c7343b8e655b)) feat(getLocationAbsUrl) - allows current url to be obtained on IE (and Chrome/Firefox)

- ([6a1c918](https://github.com/angular/protractor/commit/6a1c91848858453d0af712588b51c0bdaa0c9445)) feat(runner): add error message for bad jar path

- ([98bce7e](https://github.com/angular/protractor/commit/98bce7e2ac1e659faf2d8727e1fda210b796525e)) feat(locators): add the ability to add custom element locators with by.addLocator

  Custom locators can now be added using by.addLocator(name, script), where
  script is a self-contained snippet to be executed on the browser which returns
  an array of elements. Closes #236.

- ([c7bcc20](https://github.com/angular/protractor/commit/c7bcc20c07416237f69f7934d257b5ba5bfe8c1f)) chore(angular): update to angular 1.2


## Bug Fixes

- ([a24eeee](https://github.com/angular/protractor/commit/a24eeee4f08e973ffcecd107b6610ce1c2c5e3f6)) fix(runner): do not error out if only one spec pattern does not match any files

  Previously, the runner would throw an error if any one of the spec patterns did not
  match any files. Now it logs a warning in that case, and errors out only if there
  are no found files in any spec patterns. Closes #260

- ([f3b3fdb](https://github.com/angular/protractor/commit/f3b3fdbcbc8fe4f3c5915ef0f6eb7c89e339a62e)) fix(element): fix an error where all.then() wasn't calling callbacks.

  Closes #267

- ([137d804](https://github.com/angular/protractor/commit/137d8040778215fd841654d3ca465b71f8719ea5)) fix(jasminewd): patched matcher should understand 'not'

  Closes #139.


# 0.12.1

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Minor features

- ([201b59c](https://github.com/angular/protractor/commit/201b59c2e728c56d2a88a1167ed3007b22ab9034)) feat(jasminewd): better error messaging when expect is called with a WebElement

- ([d383770](https://github.com/angular/protractor/commit/d383770499da4b08b74ad53c20ffa288147f94e9)) feat(clientsidescripts): better error reporting from testForAngular and waitForAngular

## Bug fixes

- ([8580c0c](https://github.com/angular/protractor/commit/8580c0c76c5ccd3c55d053e59d8df37b3c4cf35a)) fix(install-selenium): update to chromedriver 2.6

  Update to the latest version of Chromedriver. This fixes the issue with
  OS X 10.9. Closes #181.

- ([ebc528f](https://github.com/angular/protractor/commit/ebc528fec2c2e88b0f9e32cee0661ecd79da2252)) fix(debugging): switch debugging tests to the new test app urls.

- ([8ff4787](https://github.com/angular/protractor/commit/8ff47875488647513f4199bab36e3b0023dd305d)) fix(runner): exit with proper code when tests fail

  When errors with messages matching /timeout/ were created, Protractor
  clears the control flow so that the remainder of the tasks scheduled
  for that spec don't bleed over into the next spec. This was messing up
  the promises used in the runner, since they are also webdriver promises.
  Long term, the runner should _not_ use webdriver promises. For now, fix by
  having the runner resolve promises directly rather than through chaining,
  and add a TODO to use promises which aren't connected to WebDriver's
  control flow in the runner.

  Closes #214.

- ([81501c5](https://github.com/angular/protractor/commit/81501c5d941cd7edb15439cef7c7a64c0e773e27)) fix(clientsidescripts): workaround for IE 8 "async page reload" init problem

- ([21264fd](https://github.com/angular/protractor/commit/21264fdc2f6cb3345c8f005936c74985ecd811dc)) fix(find): fix error when exposed to ng-options element with a default option

  Protractor will now ignore elements with the ng-bind class that don't have
  a proper binding on their data, instead of blowing up when encoutering them.

  Closes #165, may fix #170

  - ([f672648](https://github.com/angular/protractor/commit/f6726482cd2ce9a7dda9ccdeeb93574d3b9293e3)) fix(findelements): fix isPresent for repeaters by row for real

## Breaking Changes

- ([bf5b076](https://github.com/angular/protractor/commit/bf5b076cb8897d844c25baa91c263a12c61e3ab3)) fix(cli): remove boolean verbose and stack trace options

  Also add better description for what the command line options are.

  Tiny breaking change:
    Rename the 'includeStackTrace' command line option to 'stackTrace' for brevity.

# 0.12.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

This change introduces major syntax updates. Using the new syntax is recommended, but the old version is still supported for now. Note also that the test application, docs, and example tests have been updated.

## Features

- ([a2cd6c8](https://github.com/angular/protractor/commit/a2cd6c8baf242a81c4efea1f55249d597de95329)) feat(syntax): big syntax reboot, expose global $, $$, element, and by

In an effort to make tests more readable and clear, a few more global variables
will now be exported.

`browser` is an instance of protractor. This was previously accessed using
`protractor.getInstance`.

`by` is a collection of element locators. Previously, this was `protractor.By`.

`$` is a shortcut for getting elements by css. `$('.foo')` === `element(by.css('.foo'))`

All changes should be backwards incompatible, as tested with the new 'backwardscompat'
tests.

## Bug fixes

- ([8c87ae6](https://github.com/angular/protractor/commit/8c87ae6b430479445744a2f5c8eaca7f5f03d61d)) fix(onPrepare): onPrepare with a string argument should resolve from the config directory

onPrepare can take a string, which is a filename containing a script to load adn execute
before any tests run. This fixes the string to resolve the filename relative to the
config file, instead of relative to the current working directory where protractor
is called.



# 0.11.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([02cb819](https://github.com/angular/protractor/commit/02cb8199d89c6645d0bc9dbb39e5cb27517bfaf3)) feat(cli): allow passing params directly to your test

Adds a config object 'params' which is passed directly
to instances of protractor. 'params' may contain nested
objects, and can be changed via the command line as:

  --params.login.user 'Joe' --params.login.password 'abc'

This change also switches to using optimist to parse
command line flags for more flexibility and better usage
documentation. Closes #32.

- ([c025ddb](https://github.com/angular/protractor/commit/c025ddbe617b977908db509f365cc882924b196f)) feat(findElements): $ & $$ shortcuts.

Introducing the $ shortcut method for finding a single element by css
without having to call protractor.By.css.  Additionally $$ for finding
all elements by css.

Examples:
- ptor.$('.some .selector')
- ptor.$$('.some .selector')

- ([7d74184](https://github.com/angular/protractor/commit/7d7418411ea4a9d696855f755899161ecb36818d)) feat(explorer): add an interactive element explorer

When debugging or first writing test suites, you may find it helpful to
try out Protractor commands without starting up the entire test suite. You can
do this with the element explorer. This change introduces a first version
of the element explorer. Closes #107

## Bug Fixes

- ([e45ceaa](https://github.com/angular/protractor/commit/e45ceaae825cce0ec69580b8f6e93d102d4d61f1)) fix(repeaters): allow finding all rows of a repeater

Now, finding an element with the strategy 'protractor.By.repeater()' returns
a promise which will resolve to an array of WebElements, where each WebElement
is a row in the repeater. Closes #149.

- ([b501ceb](https://github.com/angular/protractor/commit/b501ceb7b776a5d9f1c2659326577601d0fbce5a)) fix(findElements): Consistently include evaluate.

When using findElements with a css locator, wrap the returned list of
elements with protractor specific functionality.

- ([c17ac12](https://github.com/angular/protractor/commit/c17ac12c2a213a7b6f8c236e81ba5cb2db542fd0)) fix(cli): allow running from command line without a config file

If all necessary fields are specified (e.g. seleniumAddress and at least
one spec), a config file shouldn't be necessary.

## Breaking Changes

- ([421d623](https://github.com/angular/protractor/commit/421d6232fe0b45ca1758afd634997da644f2e1db)) fix(repeat): use 0-based indexing for repeater rows

BREAKING CHANGE: Finding rows with protractor.By.repeater now
indexes from 0 instead of 1. This should be more familiar to most
modern programmers. You will need to edit existing tests. Closes #90.

Before:
```
// The fourth foo
ptor.findElement(protractor.By.repeater('foo in foos').row(4));
```
After:
```
// The fourth foo
ptor.findElement(protractor.By.repeater('foo in foos').row(3));
```

# 0.10.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([881759e](https://github.com/angular/protractor/commit/881759e77462dc8e1001eb77008668ae6dc552cd)) feat(timeouts): add a unique error message when waitForAngular times out

To improve the readability of error messages, when waitForAngular times out
it now produces a custom message. This should help clarify confusion
for pages that continually poll using $interval. This change also adds more
documentation on timeouts. See issue #109.

- ([37e0f1a](https://github.com/angular/protractor/commit/37e0f1af196c3c0bf54dcddf0088a8c16602e5f2)) feat(install selenium): better communication in the install script

Adds better messages in the selenium server install script, and also
makes the script output a 'start' executable which can be used to quickly
start up the selenium standalone. *not yet windows friendly*. Closes #108.

- ([b32f5a5](https://github.com/angular/protractor/commit/b32f5a59169f1324271bd5abc09c17fcd9c4f249)) feat(config): add examples for dealing with log-in

Adds examples for how to log in when the login page is not written
in Angular. New examples are in spec/login.

- ([1b7675a](https://github.com/angular/protractor/commit/1b7675aab7846bee54117876887bfec07ce31745)) feat(cli): add an onPrepare callback to the config

This onPrepare callback is useful when you want to do something with
protractor before running the specs. For example, you might want
to monkey-patch protractor with custom functions used by all the
specs, or add the protractor instance to the globals.
An example usage is shown in the spec/onPrepareConf.js file and its
associated spec.

## Bug fixes

- ([256b21c](https://github.com/angular/protractor/commit/256b21cf8c744a200892e6b7f9172150b2f4fe8d)) fix(cli): allow passing the config file before the options

The cli usage says:
> USAGE: protractor configFile [options]
However, the options passed as argument are merged into the default
configuration as soon as the configFile is met in the args parsing
loop.
This fix merges the options in the default configuration only after
the loop, allowing to pass the options to the cli before or after,
or around the config file.

- ([6223825](https://github.com/angular/protractor/commit/62238252c7fc68c6a5941883f6a272e95997a8ff)) fix(jasminewd): allow use of custom matchers

Using jasmine.Matchers.prototype to generate the chained methods for
expect() calls is flawed because it does not pick up custom matchers
defined using addMatcher.  Instead, use either the matchersClass for
the current spec or from the environment.

- ([c22fc38](https://github.com/angular/protractor/commit/c22fc387bc0ef7a07371e023d39a6ce58dfa56c9)) fix(sync): getCurrentUrl and friends should sync with Angular first

getCurrentUrl, getPageSource, and getTitle should sync with Angular
before executing. Closes #92.

- ([dd06756](https://github.com/angular/protractor/commit/dd067561cf9fe0a765e98605b9ebdd8fbfef04d3)) fix(clientsidescripts): findElements and isElementPresent for protractor.By.select

- ([c607459](https://github.com/angular/protractor/commit/c60745945c6514e25403783eab3de5873e15758b)) fix (navigation): The defer label should appear before other window names,
not after.

- ([806f381](https://github.com/angular/protractor/commit/806f38113c675a26171776a559a20bf3899aa2cc)) Fix: findElements() and isElementPresent() now work for protractor.By.input.
Closes #79.

## Breaking changes

- ([881759e](https://github.com/angular/protractor/commit/881759e77462dc8e1001eb77008668ae6dc552cd)) feat(timeouts): add a unique error message when waitForAngular times out

This changes the default script timeout from 100 seconds down to 11. Tests
which relied on extremely long timeouts will need to change the default script
timeout with `driver.manage().timeouts().setScriptTimeout(<bigNumber>)`.

# 0.9.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([0e8de99](https://github.com/angular/protractor/commit/0e8de99eb0d8a0db4f6d3538dd051c94f35775f5)) Wrap WebElements with Protractor specific features. This change allows
chained findElement calls to work with Protractor locators. It also
adds a function, evaluate, to evaluate an angular expression in the context
of a WebElement's scope.

- ([9f53118](https://github.com/angular/protractor/commit/9f5311812cbae5122ce2c6ebe522132273b0ebcc)) Improving the command line interface (adding more options). This allows
the --spec option to be passed with test files that will be resolved
relative to the current directory. Smarter merging of default config
values. Closes #65.

- ([73821fb](https://github.com/angular/protractor/commit/73821fb6b6d252a93cc15ce990b4ec4738b87b95)) Adding an 'ignoreSynchronization' property to turn off Protractor's attempt to
wait for Angular to be ready on a page. This can be used to test pages that
poll with $timeout or $http.

## Bug fixes

- ([cfc6438](https://github.com/angular/protractor/commit/cfc6438e80e77387afae776f289cd55813e9b2d9)) Adding support for isElementPresent with Protractor locators.
Closes #11.

- ([8329b01](https://github.com/angular/protractor/commit/8329b01865074c32f7a261fe9bbf2c151b704a34)) Adding waitForAngular calls before WebElement functions. Closes #37.

# 0.8.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Docs
- Added documentation to the [docs folder](https://github.com/angular/protractor/tree/master/docs).

- ([08ef244](https://github.com/angular/protractor/commit/08ef244217fb83206c818c84cbe8f07999116ee3)) Adding debugging tests showing different types of timeouts, and fixing
a bug where scheduled tasks from a previous it block would run into
the next in case of a timeout.

## Features

- ([1c7eae0](https://github.com/angular/protractor/commit/1c7eae0c09f13b7068f81324f24967709e264241)) Updating the binary script to understand debug, so that
protractor debug conf.js works.

- ([7a59479](https://github.com/angular/protractor/commit/7a594791b5ac6616de9c98dcd7d44ecaffb0e8a3)) Adding a 'debug' function to protractor. This schedules a debugger pause
within the webdriver control flow.

- ([679c82d](https://github.com/angular/protractor/commit/679c82d510ea016690fba259db50b4afa36154cc)) Mixing in all webdriver exports to protractor. This means that webdriver
classes such as ActionSequence and Keys are accessible on the global
protractor.

- ([3c76246](https://github.com/angular/protractor/commit/3c76246a01e584bc30da645a36e75920b5397251)) Added nested angular app (ng-app on an element other than `<html>` or `<body>`) capability via conf file.

## Bug fixes

- ([1c9b98d](https://github.com/angular/protractor/commit/1c9b98d0464bbe57194cf340c6e5942cbe7c8385)) Fixed Sauce issues: low timeouts, shutdown and init order.

## Breaking Changes

- Now running selenium 2.25. Requires updating WebDriverJS and the selenium standalone binary and chromedriver binary.

- ([a54abfb](https://github.com/angular/protractor/commit/a54abfbbfd3b13be5144e64e52a267c73d409a81)) Spec paths in configuration files are now resolved from the location of the spec file instead of the current working directory when the command line is run.



# 0.7.0

_Note: Major version 0 releases are for initial development, and backwards incompatible changes may be introduced at any time._

## Features

- ([7966912](https://github.com/angular/protractor/commit/796691205795d93fe12c998d20a58c8220ac6fb7)) Updating to Selenium 2.24.

- ([90f0a94](https://github.com/angular/protractor/commit/90f0a942b09faff5924674a20ce7705b6d685eba)) Instead of having tests run with the protractor runner need to require()
the protractor library, publish it to the global namespace. This insures
the instance of protractor used within the tests is the same as the
one used on the command line. Closes #36. Version bump for incompatible
API changes.

- ([cb373c9](https://github.com/angular/protractor/commit/cb373c99a7e33c5514bf1d2728a64f631ec8784c)) Adding glob matching to the spec files from the config. Closes #29.


## Breaking changes

- Now running on selenium 2.24. Requires updating WebDriverJS and the selenium standalone binary.

- The protractor runner now publishes `protractor` to the global namespace and sets up the Jasmine-WebDriver adapter. Tests run with this should no longer include

````javascript
// var protractor = require('protractor'); // No longer needed!
// require('protractor/jasminewd'); // No longer needed!

var ptor = protractor.getInstance(); // This should just work now.
````
