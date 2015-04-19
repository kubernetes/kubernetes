Setting Up the System Under Test
================================

Protractor uses real browsers to run its tests, so it can connect to anything that your browser can connect to. This means you have great flexibility in deciding what you are actually testing. It could be a development server on localhost, a staging server up on your local network, or even production servers on the general internet. All Protractor needs is the URL.
There are a couple of things to watch out for!

**If your page does manual bootstrap** Protractor will not be able to load your page using browser.get. Instead, use the base webdriver instance - `browser.driver.get`. This means that Protractor does not know when your page is fully loaded, and you may need to add a wait statement to make sure your tests avoid race conditions.

**If your page uses `$timeout` for polling** Protractor will not be able to tell when your page is ready. Consider using `$interval` instead of `$timeout`.

If you need to do global preparation for your tests (for example, logging in), you can put this into the config in the `onPrepare` property. This property can be either a function or a filename. If a filename, Protractor will load that file with Node.js and run its contents. See the [login tests](https://github.com/angular/protractor/blob/master/spec/withLoginConf.js) for an example.
