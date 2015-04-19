How It Works
============


Protractor is an end-to-end test framework for AngularJS applications. Protractor is a Node.js program that supports the Jasmine, Mocha, and Cucumber test frameworks.

Selenium is a browser automation framework. Selenium includes the Selenium Server, the WebDriver APIs, and the WebDriver browser drivers.

Protractor works in conjunction with Selenium to provide an automated test infrastructure that can simulate a user’s interaction with an Angular application running in a browser or mobile device.

![Protractor Components Diagram](/docs/components.png)

When working with Protractor, it’s important to keep the following in mind:
 - Protractor is a wrapper around WebDriverJS, the JavaScript bindings for the Selenium WebDriver API (before writing any tests, skim through the [WebDriverJS Users Guide](https://code.google.com/p/selenium/wiki/WebDriverJs)).
 - WebDriver commands are asynchronus. They are scheduled on a control flow and return promises, not primitive values (see [The WebDriver Control Flow](/docs/control-flow.md)).
 - Your test scripts send commands to the Selenium Server, which in turn communicates with the browser driver. Read on for more details.

Process Communication
---------------------

A test using Selenium WebDriver involves three processes - the test script, the server, and the browser. The communication between these processes is shown in the diagram below.

![WebDriver test Processes Diagram](/docs/processes.png)

The Selenium Server takes care of interpreting commands from the test and forwarding them to one or more browsers. Communication between the server and the browser uses the [WebDriver Wire Protocol](https://code.google.com/p/selenium/wiki/JsonWireProtocol), a JSON protocol. The command is interpreted by the Browser Driver.

With Protractor, the test script is run using Node.js. Protractor runs an extra command before performing any action on the browser to ensure that the application being tested has stabilized. For example, let's look at the following snippet of test code.

    element(by.css('button.myclass')).click();

This will result in three commands being sent to the Browser Driver

 - [/session/:sessionId/execute_async](https://code.google.com/p/selenium/wiki/JsonWireProtocol#/session/:sessionId/execute_async) - First, Protractor tells the browser to run a snippet of JavaScript. This is a custom command which asks Angular to respond when the application is done with all timeouts and asynchronous requests, and ready for the test to resume.

 - [/session/:sessionId/element](https://code.google.com/p/selenium/wiki/JsonWireProtocol#POST_/session/:sessionId/element) - Then, the command to find the element is sent.

 - [/session/:sessionId/element/:id/click](https://code.google.com/p/selenium/wiki/JsonWireProtocol#POST_/session/:sessionId/element/:id/click) - Finally the command to perform a click action is sent.



