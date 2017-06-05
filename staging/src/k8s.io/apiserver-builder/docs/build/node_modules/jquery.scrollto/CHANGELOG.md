# Changelog

## 2.1.2
### Fix
- Plugin won't break if an empty jQuery object is passed, it's now consistent with selector target #121
### Docs
- Converted the CHANGELOG to Markdown

## 2.1.1
### Fix
- Slight change so define function is not minified (#91)

## 2.1.0
### Enhancement
- Avoid animating a needless axis
### Feature
- Implemented interrupt setting, if true will stop animating on user (manual) scroll (#67)

## 2.0.1
### Fix
- Fixed "queue" setting conflicts with $().animate(), forced to always get there as true

## 2.0.0
### Feature
- All settings are passed to jQuery.animate() meaning it now supports even more settings
### Enhancement
- $(window)._scrollable() is no longer needed, the element is always the window
- Delegating to jQuery the get/set of element/window scroll positions.
### Compat
- Dropped support for $.scrollTo.window() and $(window)._scrollable()
### Fix
- Now works consistenly on Chrome 40
- Now works correctly on Windows Phone
- Now works correctly on Android Browsers
- Now works correctly on iOS Browsers

## 1.4.14
###Misc
- Internal both() function will handle nulls correctly

## 1.4.13
###Misc
- Support for CommonJS / NPM added by durango

## 1.4.12
### Fix
- Fixed selector matching body fails on window scrolling

## 1.4.11
###Misc
- Reverted changes from 1.4.10

## 1.4.10
### Enhancement
- Giving the plugin an AMD module id so it can be required (f.e by localScroll)

## 1.4.9
### Enhancement
- "offset" setting can now be a function as well (#60)

## 1.4.8
### Enhancement
- Added support for AMD

## 1.4.7
###Misc
- Changed spacing
- Changed licensing to MIT
- Repo is compliant with official jquery plugins repository

## 1.4.6
### Fix
- Fixed first argument of onAfter and onAfterFirst was original target and should be "parsed" target

## 1.4.5
### Fix
- Fixed passing a negative scroll value crashes

## 1.4.4
###Change
- Re-released as 1.4.4 to avoid issues with bower

## 1.4.3.1
### Fix
- Fixed $.scrollTo(0) broken on 1.4.3

## 1.4.3
### Enhancement
- Limit calculations can be disabled by setting the option 'limit' to false.
- Null target or unmatching selector don't break and fail silently
###Misc
- Removed support for the deprecated setting 'speed'
### Fix
- Removed $.browser.webkit so the plugin works with jQuery +1.8

## 1.4.2
### Feature
- The plugin support percentages as target ('50%' or {top:'50%', left:'45%'})
- Exposed the max() calculation as $.scrollTo.max
### Enhancement
- Renamed $.fn.scrollable to $.fn._scrollable to avoid conflicts with other plugins
### Fix
- Fixing max calculations for regular DOM elements

## 1.4.1
### Feature
- The target can be 'max' to scroll to the end while keeping it elegant.
### Enhancement
- Default duration is 0 for jquery +1.3. Means sync animation
- The plugin works on all major browsers, on compat & quirks modes, including iframes.
- In addition to window/document, if html or body are received, the plugin will choose the right one.
### Fix
- The plugin accepts floating numbers, Thanks Ramin
- Using jQuery.nodeName where neccessary so that this works on xml+xhtml
- The max() internal function wasn't completely accurrate, now it is 98% (except for IE on quirks mode and it's not too noticeable).

## 1.4
### Fix
- Fixed the problem when scrolling the window to absolute positioned elements on Safari.
- Fixed the problem on Opera 9.5 when scrolling the window. That it always scrolls to 0.
### Feature
- Added the settings object as 2nd argument to the onAfter callback.
- The 3rd argument of scrollTo can be just a function and it's used as the onAfter.
- Added full support for iframes (even max scroll calculation).
- Instead of $.scrollTo, $(window).scrollTo() and $(document).scrollTo() can be used.
- Added $().scrollable() that returns the real element to scroll, f.e: $(window).scrollable() == ###body|html], works for iframes
### Enhancement
- Cleaned the code a bit, specially the comments

## 1.3.3
###Change
- Changed the licensing from GPL to GPL+MIT.

## 1.3.2
### Enhancement
- Small improvements to make the code shorter.
###Change
- Removed the last argument received by onAfter as it was the same as the 'this' but jqueryfied.

## 1.3.1
### Feature
- Exposed $.scrollTo.window() to get the element that needs to be animated, to scroll the window.
- Added option 'over'.
### Enhancement
- Made the code as short as possible.
###Change
- Changed the arguments received by onAfter

## 1.3
### Enhancement
- Added semicolon to the start, for safe file concatenation
- Added a limit check, values below 0 or over the maximum are fixed.
- Now it should work faster, only one of html or body go through all the processing, instead of both for all browsers.
### Fix
- Fixed the behavior for Opera, which seemed to react to both changes on <html> and <body>.
- The border is also reduced, when 'margin' is set to true.
###Change
- The option speed has been renamed to duration.
### Feature
- The duration can be specified with a number as 2nd argument, and the rest of the settings as the third ( like $().animate )
- Remade the demo

#### 1.2.4
### Enhancement
- The target can be in the form of { top:x, left:y } allowing different position for each axis.
### Feature
- The option 'offset' has been added, to scroll behind or past the target. Can be a number(both axes) or { top:x, left:y }.

#### 1.2.3
### Feature
- Exposed the defaults.
### Enhancement
- Made the callback functions receive more parameters.

#### 1.2.2
### Fix
- Fixed a bug, I didn't have to add the scrolled amount if it was body or html.

## 1.2
###Change
- The option 'onafter' is now called 'onAfter'.
### Feature
- Two axes can be scrolled together, this is set with the option 'axis'.
- In case 2 axes are chosen, the scrolling can be queued: one scrolls, and then the other.
- There's an intermediary event, 'onAfterFirst' called in case the axes are queued, after the first ends.
- If the option 'margin' is set to true, the plugin will take in account, the margin of the target(no use if target is a value).