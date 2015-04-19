Setting Up the Selenium Server
==============================

When working with Protractor, you need to specify how to connect to the browser drivers which will start up and control the browsers you are testing on. You will most likely use the Selenium Server. The server acts as proxy between your test script (written with the WebDriver API) and the browser driver (controlled by the WebDriver protocols).

The server forwards commands from your script to the driver and returns responses from the driver to your script. The server can handle multiple scripts in different languages. The server can startup and manage multiple browsers in different versions and implementations.

         [Test Scripts] < ------------ > [Selenium Server] < ------------ > [Browser Drivers]

The [reference config file](/docs/referenceConf.js) includes several options for the Selenium Server, which are explained in the sections below.


Standalone Selenium Server
--------------------------

To run the Selenium Server on your local machine, use the standalone Selenium Server. 

**JDK**

To run a local Selenium Server, you will need to have the [Java Development Kit (JDK)](http://www.oracle.com/technetwork/java/javase/downloads/index.html) installed.  Check this by running `java -version` from the command line.


**Installing and Starting the Server**

To install and start the standalone Selenium Server manually, use the webdriver-manager command line tool, which comes with Protractor.

1. Run the update command:
    `webdriver-manager update`
     This will install the server and ChromeDriver.

2. Run the start command:
   `webdriver-manager start`
    This will start the server. You will see a lot of output logs, starting with INFO. The last 
    line will  be 'Info - Started org.openqa.jetty.jetty.Server'.

3. Leave the server running while you conduct your test sessions.

4. In your config file, set `seleniumAddress` to the address of the running server. This defaults to
   `http://localhost:4444/wd/hub`.


**Starting the Server from a Test Script**

To start the standalone Selenium Server from within your test script, set these options in your config file:

 - `seleniumServerJar` -  The location of the jar file for the standalone Selenium Server. Specify a file location.

 - `seleniumPort` - The port to use to start the standalone Selenium Server. If not specified, defaults to 4444.

 - `seleniumArgs` -  Array of command line options to pass to the server. For a full list, start the server with the `-help` flag.

**Connecting to a Running Server**

To connect to a running instance of a standalone Selenium Server, set this option:

 - `seleniumAddress` -  Connect to a running instance of a standalone Selenium Server. The address will be a URL.

Please note that if you set seleniumAddress, the settings for `seleniumServerJar`, `seleniumPort` and `seleniumArgs` will be ignored (it will also override the sauce options listed below).


Remote Selenium Server
----------------------

To run your tests against a remote Selenium Server, you will need an account with a service that hosts the server (and the browser drivers). Protractor has built in support for [Sauce Labs](http://www.saucelabs.com).

In your config file, set these options:
 - `sauceUser` -  The username for your Sauce Labs account.
 - `sauceKey` -  The key for your Sauce Labs account.

Please note that if you set `sauceUser` and `sauceKey`, the settings for `seleniumServerJar`, `seleniumPort` and `seleniumArgs` will be ignored.

You can optionally set the [`name` property](referenceConf.js#L113) in a capability in order to give the jobs a name on the server.  Otherwise they will just be called `Unnamed Job`.


Connecting Directly to Browser Drivers
--------------------------------------

Protractor can test directly against Chrome and Firefox without using a Selenium Server. To use this, in your config file set `directConnect: true`.

 - `directConnect: true` -  Your test script communicates directly Chrome Driver or Firefox Driver, bypassing any Selenium Server. If this is true, settings for `seleniumAddress` and `seleniumServerJar` will be ignored. If you attempt to use a browser other than Chrome or Firefox an error will be thrown.

The advantage of directly connecting to browser drivers is that your test scripts may start up and run faster.
