#!/usr/bin/env node

/**
 * This is an explorer to help get the right element locators, and test out what
 * Protractor commands will do on your site without running a full test suite.
 *
 * This beta version only uses the Chrome browser.
 *
 * Usage:
 *
 * Expects a selenium standalone server to be running at http://localhost:4444
 * from protractor directory, run with:
 *
 *     ./bin/elementexplorer.js <urL>
 *
 * This will load up the URL on webdriver and put the terminal into a REPL loop.
 * You will see a > prompt. The `browser`, `element` and `protractor` variables
 * will be available. Enter a command such as:
 *
 *     > element(by.id('foobar')).getText()
 *
 * or
 *
 *     > browser.get('http://www.angularjs.org')
 *
 * try just
 *
 *     > browser
 *
 * to get a list of functions you can call.
 *
 * Typing tab at a blank prompt will fill in a suggestion for finding
 * elements.
 */
console.log('Please use "protractor [configFile] [options] --elementExplorer"' +
  ' for full functionality\n');

if (process.argv.length > 3) {
  console.log('usage: elementexplorer.js [urL]');
  process.exit(1);
} else if (process.argv.length === 3) {
  process.argv[2] = ('--baseUrl=' + process.argv[2]);
}

process.argv.push('--elementExplorer');
require('../lib/cli.js');
