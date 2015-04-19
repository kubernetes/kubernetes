Protractor [![Build Status](https://travis-ci.org/angular/protractor.png?branch=master)](https://travis-ci.org/angular/protractor)
==========

[Protractor](http://angular.github.io/protractor) is an end-to-end test framework for [AngularJS](http://angularjs.org/) applications. Protractor is a [Node.js](http://nodejs.org/) program built on top of [WebDriverJS](https://code.google.com/p/selenium/wiki/WebDriverJs). Protractor runs tests against your application running in a real browser, interacting with it as a user would. 


Getting Started
---------------

The Protractor documentation for users is located in the [protractor/docs](https://github.com/angular/protractor/tree/master/docs) folder.

To get set up and running  quickly:
 - The [Protractor Website](http://angular.github.io/protractor)
 - Work through the [Tutorial](http://angular.github.io/protractor/#/tutorial)
 - Take a look at the [Table of Contents](http://angular.github.io/protractor/#/toc)

Once you are familiar with the tutorial, you’re ready to move on. To modify your environment, see the Protractor Setup docs. To start writing tests, see the Protractor Tests docs.

To better understand how Protractor works with the Selenium WebDriver and Selenium Server see the reference materials.


For Contributors
----------------
Clone the github repository:

    git clone https://github.com/angular/protractor.git
    cd protractor
    npm install

Start up a selenium server. By default, the tests expect the selenium server to be running at `http://localhost:4444/wd/hub`.

Protractor's test suite runs against the included test application. Start that up with

    npm start

Then run the tests with

    npm test
