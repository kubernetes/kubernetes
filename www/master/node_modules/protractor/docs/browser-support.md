Browser Support
===============
Protractor supports the two latest major versions of Chrome, Firefox, Safari, and IE. These are used in Protractor's own suite of tests. You can view the current status [on Travis](https://travis-ci.org/angular/protractor).

Note that because Protractor uses WebDriver to drive browsers, any issues with WebDriver implementations (such as FireFoxDriver, ChromeDriver, and IEDriver) will show up in Protractor. The chart below links to major known issues. You can search through all WebDriver issues at [the Selenium issue tracker](https://code.google.com/p/selenium/issues/list).


| Driver                 | Support      | Known Issues    |
|------------------------|--------------|-----------------|
|ChromeDriver            |Yes           |[Link](https://github.com/angular/protractor/labels/browser%3A%20chrome) |
|FirefoxDriver           |Yes           |[Link](https://github.com/angular/protractor/labels/browser%3A%20firefox) |
|SafariDriver            |Yes           |[Link](https://github.com/angular/protractor/labels/browser%3A%20safari) |
|IEDriver                |Yes           |[Link](https://github.com/angular/protractor/labels/browser%3A%20IE) |
|OperaDriver             |No            |                 |
|ios-Driver              |No            |                 |
|Appium - iOS/Safari     |Yes*          |[Link](https://github.com/angular/protractor/labels/browser%3A%20iOS%20safari) |
|Appium - Android/Chrome |Yes*          |[Link](https://github.com/angular/protractor/labels/browser%3A%20android%20chrome) |
|Selendroid              |Yes*          |                 |
|PhantomJS / GhostDriver |**            |[Link](https://github.com/angular/protractor/labels/browser%3A%20phantomjs) | 

(*) These drivers are not yet in the Protractor smoke tests.

(**) We recommend against using PhantomJS for tests with Protractor. There are many reported issues with PhantomJS crashing and behaving differently from real browsers.
