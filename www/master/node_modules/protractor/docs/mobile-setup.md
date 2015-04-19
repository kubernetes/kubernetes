Mobile Setup
============

There are many options for using WebDriver to test on mobile browsers. Protractor
does not yet officially support or run its own tests against a particular configuration, but the following are some notes on various setup options.

Setting Up Protractor with Appium - Android/Chrome
-------------------------------------
###### Setup
*   Install Java SDK (>1.6) and configure JAVA_HOME (Important: make sure it's not pointing to JRE).
*   Follow http://spring.io/guides/gs/android/ to install and set up Android developer environment. Do not set up Android Virtual Device as instructed here.
*   From commandline, ```android avd``` and then install an AVD, taking note of the following:
   * Start with an ARM ABI
   * Enable hardware keyboard: ```hw.keyboard=yes```
   * Enable hardware battery: ```hw.battery=yes```
   * Use the Host GPU
   * Here's an example:

Phone:
```shell
> android list avd
Available Android Virtual Devices:
    Name: LatestAndroid
  Device: Nexus 5 (Google)
    Path: /Users/hankduan/.android/avd/LatestAndroid.avd
  Target: Android 4.4.2 (API level 19)
 Tag/ABI: default/armeabi-v7a
    Skin: HVGA
```

Tablet:
```shell
> android list avd
Available Android Virtual Devices:
    Name: LatestTablet
  Device: Nexus 10 (Google)
    Path: /Users/hankduan/.android/avd/LatestTablet.avd
  Target: Android 4.4.2 (API level 19)
 Tag/ABI: default/armeabi-v7a
    Skin: WXGA800-7in
```
*   Follow http://ant.apache.org/manual/index.html to install ant and set up the environment.
*   Follow http://maven.apache.org/download.cgi to install mvn (Maven) and set up the environment. 
   * NOTE: Appium suggests installing Maven 3.0.5 (I haven't tried later versions, but 3.0.5 works for sure).
*   Install Appium using node ```npm install -g appium```. Make sure you don't install as sudo or else Appium will complain.
   * You can do this either if you installed node without sudo, or you can chown the global node_modules lib and bin directories.
*   Start emulator manually (at least the first time) and unlock screen.

```shell
> emulator -avd LatestAndroid
```
* Your devices should show up under adb now:

```shell
> adb devices
List of devices attached 
emulator-5554 device
```
*   If the AVD does not have chrome (and it probably won't if it just created), you need to install it:
   * You can get v34.0.1847.114 from http://www.apk4fun.com/apk/1192/
   * Once you download the apk, install to your AVD as such:

```shell
> adb install ~/Desktop/chrome-browser-google-34.0.1847.114-www.apk4fun.com.apk 
2323 KB/s (30024100 bytes in 12.617s)
Success
```
* If you check your AVD now, it should have Chrome.

###### Running Tests
*   Ensure app is running if testing local app (Skip if testing public website):

```shell
> npm start # or `./scripts/web-server.js`
Starting express web server in /workspace/protractor/testapp on port 8000
```
*   If your AVD isn't already started from the setup, start it now:

```shell
> emulator -avd LatestAndroid
```
*   Start Appium:

```shell
> appium
info: Welcome to Appium v1.0.0-beta.1 (REV 6fcf54391fb06bb5fb03dfcf1582c84a1d9838b6)
info: Appium REST http interface listener started on 0.0.0.0:4723
info: socket.io started
```
*Note Appium listens to port 4723 instead of 4444*

*   Configure protractor:

```javascript
exports.config = {
  seleniumAddress: 'http://localhost:4723/wd/hub',

  specs: ['basic/*_spec.js'],

  // Reference: https://github.com/appium/sample-code/blob/master/sample-code/examples/node/helpers/caps.js
  capabilities: {
    browserName: 'chrome',
    'appium-version': '1.0',
    platformName: 'Android',
    platformVersion: '4.4.2',
    deviceName: 'Android Emulator',
  },

  baseUrl: 'http://10.0.2.2:8000',
 
  // configuring wd in onPrepare
  // wdBridge helps to bridge wd driver with other selenium clients
  // See https://github.com/sebv/wd-bridge/blob/master/README.md
  onPrepare: function () {
    var wd = require('wd'),
      protractor = require('protractor'),
      wdBridge = require('wd-bridge')(protractor, wd);
    wdBridge.initFromProtractor(exports.config);
  }
};
```
*Note the following:*
 - baseUrl is 10.0.2.2 instead of localhost because it is used to access the localhost of the host machine in the android emulator  
 - selenium address is using port 4723
 
Setting Up Protractor with Appium - iOS/Safari
-------------------------------------
###### Setup
*   Install Java SDK (>1.6) and configure JAVA_HOME (Important: make sure it's not pointing to JRE).
*   Follow http://ant.apache.org/manual/index.html to install ant and set up the environment.
*   Follow http://maven.apache.org/download.cgi to install mvn (Maven) and set up the environment. 
   * NOTE: Appium suggests installing Maven 3.0.5 (I haven't tried later versions, but 3.0.5 works for sure).
*   Install Appium using node ```npm install -g appium```. Make sure you don't install as sudo or else Appium will complain.
   * You can do this either if you installed node without sudo, or you can chown the global node_modules lib and bin directories.
*  Run the following: `appium-doctor` and `authorize_ios` (sudo if necessary)
*  You need XCode >= 4.6.3, 5.1.1 recommended. Note, iOS8 (XCode 6) does not work off the shelf (see https://github.com/appium/appium/pull/3517)

###### Running Tests
*   Ensure app is running if testing local app (Skip if testing public website):

```shell
> npm start # or `./scripts/web-server.js`
Starting express web server in /workspace/protractor/testapp on port 8000
```
*   Start Appium:

```shell
> appium
info: Welcome to Appium v1.0.0-beta.1 (REV 6fcf54391fb06bb5fb03dfcf1582c84a1d9838b6)
info: Appium REST http interface listener started on 0.0.0.0:4723
info: socket.io started
```
*Note: Appium listens to port 4723 instead of 4444.*

*   Configure protractor:

iPhone:
```javascript
exports.config = {
  seleniumAddress: 'http://localhost:4723/wd/hub',

  specs: [
    'basic/*_spec.js'
  ],

  // Reference: https://github.com/appium/sample-code/blob/master/sample-code/examples/node/helpers/caps.js
  capabilities: {
    browserName: 'safari',
    'appium-version': '1.0',
    platformName: 'iOS',
    platformVersion: '7.1',
    deviceName: 'iPhone Simulator',
  },

  baseUrl: 'http://localhost:8000',

  // configuring wd in onPrepare
  // wdBridge helps to bridge wd driver with other selenium clients
  // See https://github.com/sebv/wd-bridge/blob/master/README.md
  onPrepare: function () {
    var wd = require('wd'),
      protractor = require('protractor'),
      wdBridge = require('wd-bridge')(protractor, wd);
    wdBridge.initFromProtractor(exports.config);
  }
};
```

iPad:
```javascript
exports.config = {
  seleniumAddress: 'http://localhost:4723/wd/hub',

  specs: [
    'basic/*_spec.js'
  ],

  // Reference: https://github.com/appium/sample-code/blob/master/sample-code/examples/node/helpers/caps.js
  capabilities: {
    browserName: 'safari',
    'appium-version': '1.0',
    platformName: 'iOS',
    platformVersion: '7.1',
    deviceName: 'IPad Simulator',
  },

  baseUrl: 'http://localhost:8000',

  // configuring wd in onPrepare
  // wdBridge helps to bridge wd driver with other selenium clients
  // See https://github.com/sebv/wd-bridge/blob/master/README.md
  onPrepare: function () {
    var wd = require('wd'),
      protractor = require('protractor'),
      wdBridge = require('wd-bridge')(protractor, wd);
    wdBridge.initFromProtractor(exports.config);
  }
};

```
*Note the following:*
 - note capabilities
 - baseUrl is localhost (not 10.0.2.2)
 - selenium address is using port 4723

Setting Up Protractor with Selendroid
-------------------------------------
###### Setup
*   Install Java SDK (>1.6) and configure JAVA_HOME (Important: make sure it's not pointing to JRE).
*   Follow http://spring.io/guides/gs/android/ to install and set up Android developer environment. Do not set up Android Virtual Device as instructed here.
*   From commandline, 'android avd' and then follow Selendroid's recommendation (http://selendroid.io/setup.html#androidDevices). Take note of the emulator accelerator. Here's an example:

```shell
> android list avd
Available Android Virtual Devices:
    Name: myAvd
  Device: Nexus 5 (Google)
    Path: /Users/hankduan/.android/avd/Hank.avd
  Target: Android 4.4.2 (API level 19)
 Tag/ABI: default/x86
    Skin: WVGA800
```

###### Running Tests
*   Ensure app is running if testing local app (Skip if testing public website):

```shell
> npm start # or `./scripts/web-server.js`
Starting express web server in /workspace/protractor/testapp on port 8000
```

*   Start emulator manually (at least the first time):

```shell
> emulator -avd myAvd
HAX is working and emulator runs in fast virt mode
```

*Note: The last line that tells you the emulator accelerator is running.*
*   Start selendroid:

```shell
> java -jar selendroid-standalone-0.9.0-with-dependencies.jar
...
```

*   Once selendroid is started, you should be able to go to "http://localhost:4444/wd/hub/status" and see your device there:

```javascript
{"value":{"os":{"name":"Mac OS X","arch":"x86_64","version":"10.9.2"},"build":{"browserName":"selendroid","version":"0.9.0"},"supportedDevices":[{"emulator":true,"screenSize":"WVGA800","avdName":"Hank","androidTarget":"ANDROID19"}],"supportedApps":[{"mainActivity":"io.selendroid.androiddriver.WebViewActivity","appId":"io.selendroid.androiddriver:0.9.0","basePackage":"io.selendroid.androiddriver"}]},"status":0}
```

*   Configure protractor:

```javascript
exports.config = {
  seleniumAddress: 'http://localhost:4444/wd/hub',

  specs: [
    'basic/*_spec.js'
  ],

  capabilities: {
    'browserName': 'android'
  },

  baseUrl: 'http://10.0.2.2:8000'
};
```

*Note the following:*
 - browserName is 'android'
 - baseUrl is 10.0.2.2 instead of localhost because it is used to access the localhost of the host machine in the android emulator
