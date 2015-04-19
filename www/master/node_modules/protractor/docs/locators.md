Using Locators
==============

The heart of end-to-end tests for webpages is finding DOM elements, interacting with them, and getting information about the current state of your application. This doc is an overview of how to locate and perform actions on DOM elements using Protractor.

Overview
--------

Protractor exports a global function `element`, which takes a *Locator* and will return an *ElementFinder*. This function finds a single element - if you need to manipulate multiple elements, use the `element.all` function.

The *ElementFinder* has a set of *action methods*, such as `click()`, `getText()`, and `sendKeys`. These are the core way to interact with an element and get information back from it.

When you find elements in Protractor all actions are asynchronous. Behind the scenes, all actions are sent to the browser being controlled using the JSON Webdriver Wire Protocol. The browser then performs the action as a user natively would.

Locators
--------

A locator tells Protractor how to find a certain DOM element. Protractor exports locator factories on the global `by` object. The most common locators are:

```js
// find an element using a css selector
by.css('.myclass') 

// find an element with the given id
by.id('myid')

// find an element with a certain ng-model
by.model('name')

// find an element bound to the given variable
by.binding('bindingname')
```

For a list of Protractor-specific locators, see the [Protractor API: ProtractorBy](#/api?view=ProtractorBy).

The locators are passed to the `element` function, as below:

```js
element(by.css('some-css'));
element(by.model('item.name'));
element(by.binding('item.name'));
```

When using CSS Selectors as a locator, you can use the shortcut $() notation:

```js
$('my-css');

// Is the same as

element(by.css('my-css'));
```

Actions
-------

The `element()` function returns an ElementFinder object. The ElementFinder knows how to locate the DOM element using the locator you passed in as a parameter, but it has not actually done so yet. It will not contact the browser until an *action* method has been called.

The most common action methods are:

```js
var el = element(locator);

// Click on the element
el.click();

// Send keys to the element (usually an input)
el.sendKeys('my text');

// Clear the text in an element (usually an input)
el.clear();

// Get the value of an attribute, for example, get the value of an input
el.getAttribute('value');
```

Since all actions are asynchronous, all action methods return a [promise](https://code.google.com/p/selenium/wiki/WebDriverJs#Promises). So, to log the text of an element, you would do something like:
```js
var el = element(locator);
el.getText().then(function(text) {
  console.log(text);
});
```

Any action available in WebDriverJS on a WebElement is available on an ElementFinder. [See a full list](#/api?view=ElementFinder).


Finding Multiple Elements
-------------------------

To deal with multiple DOM elements, use the `element.all` function. This also takes a locator as its only parameter.

```js
element.all(by.css('.selector')).then(function(elements) {
  // elements is an array of ElementFinders.
});
```

`element.all()` has several helper functions:

```js
// Number of elements.
element.all(locator).count();

// Get by index (starting at 0).
element.all(locator).get(index);

// First and last.
element.all(locator).first();
element.all(locator).last();
```


Finding Sub-Elements
--------------------

To find sub-elements, simply chain element and element.all functions together as shown below.

Using a single locator to find:

 - an element
    ```js
    element(by.css('some-css'));
    ```

 - a list of elements:
    ```js
    element.all(by.css('some-css'));
    ```

Using chained locators to find:

 - a sub-element:
    ```js
    element(by.css('some-css')).element(by.tagName('tag-within-css'));
    ```

 - to find a list of sub-elements:
    ```js
    element(by.css('some-css')).all(by.tagName('tag-within-css'));
    ```

You can chain with get/first/last as well like so:

```js
element.all(by.css('some-css')).first().element(by.tagName('tag-within-css'));
element.all(by.css('some-css')).get(index).element(by.tagName('tag-within-css'));
element.all(by.css('some-css')).first().all(by.tagName('tag-within-css'));
```

Behind the Scenes: ElementFinders versus WebElements
----------------------------------------------------

If you're familiar with WebDriver and WebElements, or you're just curious about the WebElements mentioned above, you may be wondering how they relate to ElementFinders.

When you call `driver.findElement(locator)`, WebDriver immediately sends a command over to the browser asking it to locate the element. This isn't great for creating page objects, because we want to be able to do things in setup (before a page may have been loaded) like

```js
var myButton = ??;
```

and re-use the variable `myButton` throughout your test. ElementFinders get around this by simply storing the locator information until an action is called.

```js
var myButton = element(locator);
// No command has been sent to the browser yet.
```

The browser will not receive any commands until you call an action.

```js
myButton.click();
// Now two commands are sent to the browser - find the element, and then click it
```

ElementFinders also enable chaining to find subelements, such as `element(locator1).element(locator2)`.

All WebElement actions are wrapped in this way and available on the ElementFinder, in addition to a couple helper methods like `isPresent`. 

You can always access the underlying WebElement using `element(locator).getWebElement()`, but you should not need to.

