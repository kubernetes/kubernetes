# jQuery.scrollTo

Lightweight, cross-browser and highly customizable animated scrolling with jQuery

[![GitHub version](https://badge.fury.io/gh/flesler%2Fjquery.scrollTo.svg)](http://badge.fury.io/gh/flesler%2Fjquery.scrollTo)
[![libscore](http://img.shields.io/badge/libscore-31656-brightgreen.svg?style=flat-square)](http://libscore.com/#jQuery.fn.scrollTo)

## Installation
The plugin requires jQuery 1.8 or higher.

Via [bower](https://github.com/flesler/jquery.scrollTo/blob/master/bower.json):
```bash
bower install jquery.scrollTo
```
Via [npm](https://www.npmjs.com/package/jquery.scrollto):
```bash
npm install jquery.scrollto
```
Via [packagist](https://packagist.org/packages/flesler/jquery.scrollTo):
```php
php composer.phar require --prefer-dist flesler/jquery.scrollto "*"
```

### Using a public CDN

CDN provided by [jsdelivr](http://www.jsdelivr.com/#!jquery.scrollto)
```html
<script src="//cdn.jsdelivr.net/jquery.scrollto/2.1.0/jquery.scrollTo.min.js"></script>
```
CDN provided by [cdnjs](https://cdnjs.com/libraries/jquery-scrollTo)
```html
<script src="//cdnjs.cloudflare.com/ajax/libs/jquery-scrollTo/2.1.0/jquery.scrollTo.min.js"></script>
```

### Downloading Manually

If you want the latest stable version, get the latest release from the [releases page](https://github.com/flesler/jquery.scrollTo/releases).

## 2.0

Version 2.0 has been recently released. It is mostly backwards compatible, if you have any issue first check [this link](https://github.com/flesler/jquery.scrollTo/wiki/Migrating-to-2.0).
If your problem is not solved then go ahead and [report the issue](https://github.com/flesler/jquery.scrollTo/issues/new).

## Usage

jQuery.scrollTo's signature is designed to resemble [$().animate()](http://api.jquery.com/animate/).

```js
$(element).scrollTo(target[,duration][,settings]);
```

### _element_

This must be a scrollable element, to scroll the whole window use `$(window)`.

### _target_

This defines the position to where `element` must be scrolled. The plugin supports all these formats:
 * A number with a fixed position: `250`
 * A string with a fixed position with px: `"250px"`
 * A string with a percentage (of container's size): `"50%"`
 * A string with a relative step: `"+=50px"`
 * An object with `left` and `top` containining any of the aforementioned: `{left:250, top:"50px"}`
 * The string `"max"` to scroll to the end.
 * A string selector that will be relative to the element to scroll: `".section:eq(2)"`
 * A DOM element, probably a child of the element to scroll: `document.getElementById("top")`
 * A jQuery object with a DOM element: `$("#top")`

### _settings_

The `duration` parameter is a shortcut to the setting with the same name.
These are the supported settings:
 * __axis__: The axes to animate: `xy` (default), `x`, `y`, `yx`
 * __interrupt__: If `true` will cancel the animation if the user scrolls. Default is `false`
 * __limit__: If `true` the plugin will not scroll beyond the container's size. Default is `true`
 * __margin__: If `true`, subtracts the margin and border of the `target` element. Default is `false`
 * __offset__: Added to the final position, can be a number or an object with `left` and `top`
 * __over__: Adds a % of the `target` dimensions: `{left:0.5, top:0.5}`
 * __queue__: If `true` will scroll one `axis` and then the other. Default is `false`
 * __onAfter(target, settings)__: A callback triggered when the animation ends (jQuery's `complete()`)
 * __onAfterFirst(target, settings)__: A callback triggered after the first axis scrolls when queueing

You can add any setting supported by [$().animate()](http://api.jquery.com/animate/#animate-properties-options) as well:

 * __duration__: Duration of the animation, default is `0` which makes it instantaneous
 * __easing__: Name of an easing equation, you must register the easing function: `swing`
 * __fail()__: A callback triggered when the animation is stopped (f.e via `interrupt`)
 * __step()__: A callback triggered for every animated property on every frame
 * __progress()__: A callback triggered on every frame
 * And more, check jQuery's [documentation](http://api.jquery.com/animate/#animate-properties-options)

### window shorthand

You can use `$.scrollTo(...)` as a shorthand for `$(window).scrollTo(...)`.

### Changing the default settings

As with most plugins, the default settings are exposed so they can be changed.
```js
$.extend($.scrollTo.defaults, {
  axis: 'y',
  duration: 800
});
```

### Stopping the animation

jQuery.scrollTo ends up creating ordinary animations which can be stopped by calling [$().stop()](http://api.jquery.com/stop/) or [$().finish()](http://api.jquery.com/finish/) on the same element you called `$().scrollTo()`, including the `window`.
Remember you can pass a `fail()` callback to be called when the animation is stopped.

## Demo

Check the [demo](http://demos.flesler.com/jquery/scrollTo/) to see every option in action.

## Complementary plugins

There are two plugins, also created by me that depend on jQuery.scrollTo and aim to simplify certain use cases.

### [jQuery.localScroll](https://github.com/flesler/jquery.localScroll)

This plugin makes it very easy to implement anchor navigation.
If you don't want to include another plugin, you can try using something like [this minimalistic gist](https://gist.github.com/flesler/3f3e1166690108abf747).

### [jQuery.serialScroll](https://github.com/flesler/jquery.serialScroll)

This plugin simplifies the creation of scrolling slideshows.

## License

(The MIT License)

Copyright (c) 2007-2015 Ariel Flesler <aflesler@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
