# Accessibility Developer Tools

This is a library of accessibility-related testing and utility code.

Its main component is the accessibility audit: a collection of audit rules checking for common accessibility problems, and an API for running these rules in an HTML page.

There is also a collection of accessibility-related utility code, including but not limited to:
* contrast ratio calculation and color suggestions
* retrieving and validating ARIA attributes and states
* accessible name calculation using the algorithm at [http://www.w3.org/TR/wai-aria/roles#textalternativecomputation](http://www.w3.org/TR/wai-aria/roles#textalternativecomputation)

# Getting the code

To include just the javascript rules, require the following file:

    https://raw.github.com/GoogleChrome/accessibility-developer-tools/stable/dist/js/axs_testing.js

  `git 1.6.5` or later: 

    % git clone --recursive https://github.com/GoogleChrome/accessibility-developer-tools.git
    
  Before `git 1.6.5`:

    % git clone https://github.com/GoogleChrome/accessibility-developer-tools.git
    % cd accessibility-developer-tools
    % git submodule init; git submodule update

# Building

You will need `node` and `grunt-cli` to build.

1. (Once only) Install [Node.js](http://nodejs.org/) and `npm` - useful instructions here: [https://gist.github.com/isaacs/579814](https://gist.github.com/isaacs/579814)

    Make sure you have Node.js v 0.8 or higher.

2. (Once only) Use `npm` to install `grunt-cli`

        % npm install -g grunt-cli  # May need to be run as root

3. (Every time you make a fresh checkout) Install dependencies (including `grunt`) for this project (run from project root)

        % npm install

4. Build using `grunt` (run from project root)

        % grunt

# Using the Audit API

## Including the library

The simplest option is to include the generated `axs_testing.js` library on your page.

Work is underway to include the library in WebDriver and other automated testing frameworks.

## The `axs.Audit.run()` method

Once you have included `axs_testing.js`, you can call call `axs.Audit.run()`. This returns an object in the following form:

    {
      /** @type {axs.constants.AuditResult} */
      result,  // one of PASS, FAIL or NA

      /** @type {Array.<Element>} */
      elements,  // The elements which the rule fails on, if result == axs.constants.AuditResult.FAIL

      /** @type {axs.AuditRule} */
      rule  // The rule which this result is for.
    }

### Command Line Runner

The Accessibility Developer Tools project includes a command line runner for the audit. To use the runner, [install phantomjs](http://phantomjs.org/download.html) then run the following command from the project root directory.

    $ phantomjs tools/runner/audit.js <url-or-filepath>

The runner will load the specified file or URL in a headless browser, inject axs_testing.js, run the audit and output the report text.

## Using the results

### Interpreting the result

The result may be one of three constants:
* `axs.constants.AuditResult.PASS` - This implies that there were elements on the page that may potentially have failed this audit rule, but they passed. Congratulations!
* `axs.constants.AuditResult.NA` - This implies that there were no elements on the page that may potentially have failed this audit rule. For example, an audit rule that checks video elements for subtitles would return this result if there were no video elements on the page.
* `axs.constants.AuditResult.FAIL` - This implies that there were elements on the page that did not pass this audit rule. This is the only result you will probably be interested in.

### Creating a useful error message

The static, global `axs.Audit.createReport(results, opt_url)` may be used to create an error message using the return value of axs.Audit.run(). This will look like the following:

    *** Begin accessibility audit results ***
    An accessibility audit found 4 errors and 4 warnings on this page.
    For more information, please see https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules

    Error: badAriaAttributeValue (AX_ARIA_04) failed on the following elements (1 - 3 of 3):
    DIV:nth-of-type(3) > INPUT
    DIV:nth-of-type(5) > INPUT
    #aria-invalid

    Error: badAriaRole (AX_ARIA_01) failed on the following element:
    DIV:nth-of-type(11) > SPAN

    Error: controlsWithoutLabel (AX_TEXT_01) failed on the following elements (1 - 3 of 3):
    DIV > INPUT
    DIV:nth-of-type(12) > DIV:nth-of-type(3) > INPUT
    LABEL > INPUT

    Error: requiredAriaAttributeMissing (AX_ARIA_03) failed on the following element:
    DIV:nth-of-type(13) > DIV:nth-of-type(11) > DIV

    Warning: focusableElementNotVisibleAndNotAriaHidden (AX_FOCUS_01) failed on the following element:
    #notariahidden

    Warning: imagesWithoutAltText (AX_TEXT_02) failed on the following elements (1 - 2 of 2):
    #deceptive-img
    DIV:nth-of-type(13) > IMG

    Warning: lowContrastElements (AX_COLOR_01) failed on the following elements (1 - 2 of 2):
    DIV:nth-of-type(13) > DIV
    DIV:nth-of-type(13) > DIV:nth-of-type(3)

    Warning: nonExistentAriaLabelledbyElement (AX_ARIA_02) failed on the following elements (1 - 2 of 2):
    DIV:nth-of-type(3) > INPUT
    DIV:nth-of-type(5) > INPUT
    *** End accessibility audit results ***

Each rule will have at most five elements listed as failures, in the form of a unique query selector for each element.

### Configuring the Audit

If you wish to fine-tune the audit, you can create an `axs.AuditConfiguration` object, with the following options:

#### Ignore parts of the page for a particular audit rule

For example, say you have a separate high-contrast version of your page, and there is a CSS rule which causes certain elements (with class `pretty`) on the page to be low-contrast for stylistic reasons. Running the audit unmodified produces results something like

    Warning: lowContrastElements (AX_COLOR_01) failed on the following elements (1 - 5 of 15):
    ...

You can modify the audit to ignore the elements which are known and intended to have low contrast like this:

    var configuration = new axs.AuditConfiguration();
    configuration.ignoreSelectors('lowContrastElements', '.pretty');
    axs.Audit.run(configuration);

The `AuditConfiguration.ignoreSelectors()` method takes a rule name, which you can find in the audit report, and a query selector string representing the parts of the page to be ignored for that audit rule. Multiple calls to `ignoreSelectors()` can be made for each audit rule, if multiple selectors need to be ignored.

#### Restrict the scope of the entire audit to a subsection of the page

You may have a part of the page which varies while other parts of the page stay constant, like a content area vs. a toolbar. In this case, running the audit on the entire page may give you spurious results in the part of the page which doesn't vary, which may drown out regressions in the main part of the page.

You can set a `scope` on the `AuditConfiguration` object like this:

    var configuration = new axs.AuditConfiguration();
    configuration.scope = document.querySelector('main');  // or however you wish to choose your scope element
    axs.Audit.run(configuration);
    
## License

Copyright 2013 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
