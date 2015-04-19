Setting Up Protractor
=====================

Prerequisites
-------------

**Node.js**

Protractor is a [Node.js](http://nodejs.org/) program. To run Protractor, you will need to have Node.js installed. Check the version of node you have by running `node --version`. It should be greater than v0.10.0. 

Node.js comes with the Protractor [npm](https://www.npmjs.org/) package, which you can use to install Protractor.


Installing Protractor
---------------------

Use npm to install Protractor globally (omit the -g if you’d prefer not to install globally):

    npm install -g protractor

Check that Protractor is working by running `protractor --version`.

The Protractor install includes the following:
 - `protractor` command line tool
 - `webdriver-manager` command line tool
 - Protractor API (library)
