# Angular Matchers

angular matchers provides a set of matchers for the [Jasmine](http://pivotal.github.com/jasmine/) JavaScript Testing Framework:
  
- a set of custom matchers for Angular framework that are meant to make your testing a little bit easier


## Angular matchers

jasmine-jquery provides following custom matchers (in alphabetical order):

- `toBe(jQuerySelector)`
  - e.g. `expect($('<div id="some-id"></div>')).toBe('div')`
  - e.g. `expect($('<div id="some-id"></div>')).toBe('div#some-id')`
- `toBeChecked()`
  - only for tags that have checked attribute
  - e.g. `expect($('<input type="checkbox" checked="checked"/>')).toBeChecked()` 
- `toBeEmpty()`
  - Checks for child DOM elements or text.
- `toBeHidden()`
  
  Elements can be considered hidden for several reasons:
    - They have a CSS `display` value of `none`.
    - They are form elements with `type` equal to `hidden`.
    - Their `width` and `height` are explicitly set to `0`.
    - An ancestor element is hidden, so the element is not shown on the page.
- `toHaveCss(css)`
  - e.g. `expect($('<div style="display: none; margin: 10px;"></div>')).toHaveCss({display: "none", margin: "10px"})`
  - e.g. `expect($('<div style="display: none; margin: 10px;"></div>')).toHaveCss({margin: "10px"})`
- `toBeSelected()`
  - only for tags that have selected attribute
  - e.g. `expect($('<option selected="selected"></option>')).toBeSelected()`
- `toBeVisible()`
  - Elements are considered visible if they consume space in the document. Visible elements have a width or height that is greater than zero.
- `toContain(jQuerySelector)`
  - e.g. `expect($('<div><span class="some-class"></span></div>')).toContain('span.some-class')`
- `toExist()`
- `toHaveAttr(attributeName, attributeValue)`
  - attribute value is optional, if omitted it will check only if attribute exists
- `toHaveProp(propertyName, propertyValue)`
  - property value is optional, if omitted it will check only if property exists
- `toHaveClass(className)`
  - e.g. `expect($('<div class="some-class"></div>')).toHaveClass("some-class")`  
- `toHaveData(key, value)`
  - value is optional, if omitted it will check only if an entry for that key exists
- `toHaveHtml(string)`
  - e.g. `expect($('<div><span></span></div>')).toHaveHtml('<span></span>')`
- `toContainHtml(string)`
  - e.g. `expect($('<div><ul></ul><h1>header</h1></div>')).toContainHtml('<ul></ul>')`
- `toHaveId(id)`
  - e.g. `expect($('<div id="some-id"></div>')).toHaveId("some-id")`
- `toHaveText(string)`
  - accepts a String or regular expression
  - e.g. `expect($('<div>some text</div>')).toHaveText('some text')`
- `toHaveValue(value)`
  - only for tags that have value attribute
  - e.g. `expect($('<input type="text" value="some text"/>')).toHaveValue('some text')`
- `toBeDisabled()`
  - e.g. `expect('<input type='submit' disabled='disabled'/>').toBeDisabled()`
- `toBeFocused()`
  - e.g. `expect($('<input type='text' />').focus()).toBeFocused()`
- `toHandle(eventName)`
  - e.g. `expect($form).toHandle("submit")`
- `toHandleWith(eventName, eventHandler)`
  - e.g. `expect($form).toHandleWith("submit", yourSubmitCallback)`
  
The same as with standard Jasmine matchers, all of above custom matchers may be inverted by using `.not` prefix, e.g.:

    expect($('<div>some text</div>')).not.toHaveText(/other/)


## Dependencies

angular matchers was tested with Jasmine 1.2 and jQuery 1.8 on FF, Chrome, and Safari. There is a high chance it will work with older versions and other browsers as well, but I don't typically run test suite against them when adding new features.

## Cross domain policy problems under Chrome

Newer versions of Chrome don't allow file:// URIs read other file:// URIs. In effect, jasmine-jquery cannot properly load fixtures under some versions of Chrome. An override for this is to run Chrome with a switch `--allow-file-access-from-files`. 

Under Windows 7, you have to launch `C:\Users\[UserName]\AppData\Local\Google\Chrome[ SxS]\Application\chrome.exe --allow-file-access-from-files`


### Writing the Code

- Get the code right.
- Include tests that fail without your code, and pass with it.
- Update the (surrounding) documentation, examples elsewhere, and the guides: whatever is affected by your contribution.
- Follow the conventions in the source you see used already, basically [npm coding style](http://npmjs.org/doc/coding-style.html)

If you can, have another developer sanity check your change

### Install via Bower

Now offers bower support. `bower install angularjs-jasmine-matchers --save`


### Build status
[![Build Status](https://travis-ci.org/ferronrsmith/angularjs-jasmine-matchers.png?branch=master)](https://travis-ci.org/ferronrsmith/angularjs-jasmine-matchers)