/**
 * All scripts to be run on the client via executeAsyncScript or
 * executeScript should be put here.
 *
 * NOTE: These scripts are transmitted over the wire as JavaScript text
 * constructed using their toString representation, and *cannot*
 * reference external variables.
 *
 * Some implementations seem to have issues with // comments, so use star-style
 * inside scripts.  (TODO: add issue number / example implementations
 * that caused the switch to avoid the // comments.)
 */

// jshint browser: true
// jshint shadow: true
/* global angular */
var functions = {};

/**
 * Wait until Angular has finished rendering and has
 * no outstanding $http calls before continuing.
 *
 * Asynchronous.
 *
 * @param {string} rootSelector The selector housing an ng-app
 * @param {function(string)} callback callback. If a failure occurs, it will
 *     be passed as a parameter.
 */
functions.waitForAngular = function(rootSelector, callback) {
  var el = document.querySelector(rootSelector);

  try {
    if (!window.angular) {
      throw new Error('angular could not be found on the window');
    }
    if (angular.getTestability) {
      angular.getTestability(el).whenStable(callback);
    } else {
      if (!angular.element(el).injector()) {
        throw new Error('root element (' + rootSelector + ') has no injector.' +
           ' this may mean it is not inside ng-app.');
      }
      angular.element(el).injector().get('$browser').
          notifyWhenNoOutstandingRequests(callback);
    }
  } catch (err) {
    callback(err.message);
  }
};

/**
 * Find a list of elements in the page by their angular binding.
 *
 * @param {string} binding The binding, e.g. {{cat.name}}.
 * @param {boolean} exactMatch Whether the binding needs to be matched exactly
 * @param {Element} using The scope of the search.
 * @param {string} rootSelector The selector to use for the root app element.
 *
 * @return {Array.<Element>} The elements containing the binding.
 */
functions.findBindings = function(binding, exactMatch, using, rootSelector) {
  var root = document.querySelector(rootSelector || 'body');
  using = using || document;
  if (angular.getTestability) {
    return angular.getTestability(root).
        findBindings(using, binding, exactMatch);
  }
  var bindings = using.getElementsByClassName('ng-binding');
  var matches = [];
  for (var i = 0; i < bindings.length; ++i) {
    var dataBinding = angular.element(bindings[i]).data('$binding');
    if (dataBinding) {
      var bindingName = dataBinding.exp || dataBinding[0].exp || dataBinding;
      if (exactMatch) {
        var matcher = new RegExp('({|\\s|^|\\|)' + binding + '(}|\\s|$|\\|)');
        if (matcher.test(bindingName)) {
          matches.push(bindings[i]);
        }
      } else {
        if (bindingName.indexOf(binding) != -1) {
          matches.push(bindings[i]);
        }
      }

    }
  }
  return matches; /* Return the whole array for webdriver.findElements. */
};

/**
 * Find an array of elements matching a row within an ng-repeat.
 * Always returns an array of only one element for plain old ng-repeat.
 * Returns an array of all the elements in one segment for ng-repeat-start.
 *
 * @param {string} repeater The text of the repeater, e.g. 'cat in cats'.
 * @param {boolean} exact Whether the repeater needs to be matched exactly
 * @param {number} index The row index.
 * @param {Element} using The scope of the search.
 *
 * @return {Array.<Element>} The row of the repeater, or an array of elements
 *     in the first row in the case of ng-repeat-start.
 */
 functions.findRepeaterRows = function(repeater, exact, index, using) {
  function repeaterMatch(ngRepeat, repeater, exact) {
    if (exact) {
      return ngRepeat.split(' track by ')[0].split(' as ')[0].split('|')[0].
          trim() == repeater;
    } else {
      return ngRepeat.indexOf(repeater) != -1;
    }
  }

  using = using || document;

  var prefixes = ['ng-', 'ng_', 'data-ng-', 'x-ng-', 'ng\\:'];
  var rows = [];
  for (var p = 0; p < prefixes.length; ++p) {
    var attr = prefixes[p] + 'repeat';
    var repeatElems = using.querySelectorAll('[' + attr + ']');
    attr = attr.replace(/\\/g, '');
    for (var i = 0; i < repeatElems.length; ++i) {
      if (repeaterMatch(repeatElems[i].getAttribute(attr), repeater, exact)) {
        rows.push(repeatElems[i]);
      }
    }
  }
  /* multiRows is an array of arrays, where each inner array contains
     one row of elements. */
  var multiRows = [];
  for (var p = 0; p < prefixes.length; ++p) {
    var attr = prefixes[p] + 'repeat-start';
    var repeatElems = using.querySelectorAll('[' + attr + ']');
    attr = attr.replace(/\\/g, '');
    for (var i = 0; i < repeatElems.length; ++i) {
      if (repeaterMatch(repeatElems[i].getAttribute(attr), repeater, exact)) {
        var elem = repeatElems[i];
        var row = [];
        while (elem.nodeType != 8 ||
            !repeaterMatch(elem.nodeValue, repeater, exact)) {
          if (elem.nodeType == 1) {
            row.push(elem);
          }
          elem = elem.nextSibling;
        }
        multiRows.push(row);
      }
    }
  }
  var row = rows[index] || [], multiRow = multiRows[index] || [];
  return [].concat(row, multiRow);
 };

 /**
 * Find all rows of an ng-repeat.
 *
 * @param {string} repeater The text of the repeater, e.g. 'cat in cats'.
 * @param {boolean} exact Whether the repeater needs to be matched exactly
 * @param {Element} using The scope of the search.
 *
 * @return {Array.<Element>} All rows of the repeater.
 */
 functions.findAllRepeaterRows = function(repeater, exact, using) {
  function repeaterMatch(ngRepeat, repeater, exact) {
    if (exact) {
      return ngRepeat.split(' track by ')[0].split(' as ')[0].split('|')[0].
          trim() == repeater;
    } else {
      return ngRepeat.indexOf(repeater) != -1;
    }
  }

  using = using || document;

  var rows = [];
  var prefixes = ['ng-', 'ng_', 'data-ng-', 'x-ng-', 'ng\\:'];
  for (var p = 0; p < prefixes.length; ++p) {
    var attr = prefixes[p] + 'repeat';
    var repeatElems = using.querySelectorAll('[' + attr + ']');
    attr = attr.replace(/\\/g, '');
    for (var i = 0; i < repeatElems.length; ++i) {
      if (repeaterMatch(repeatElems[i].getAttribute(attr), repeater, exact)) {
        rows.push(repeatElems[i]);
      }
    }
  }
  for (var p = 0; p < prefixes.length; ++p) {
    var attr = prefixes[p] + 'repeat-start';
    var repeatElems = using.querySelectorAll('[' + attr + ']');
    attr = attr.replace(/\\/g, '');
    for (var i = 0; i < repeatElems.length; ++i) {
      if (repeaterMatch(repeatElems[i].getAttribute(attr), repeater, exact)) {
        var elem = repeatElems[i];
        while (elem.nodeType != 8 ||
            !repeaterMatch(elem.nodeValue, repeater, exact)) {
          if (elem.nodeType == 1) {
            rows.push(elem);
          }
          elem = elem.nextSibling;
        }
      }
    }
  }
  return rows;
 };

/**
 * Find an element within an ng-repeat by its row and column.
 *
 * @param {string} repeater The text of the repeater, e.g. 'cat in cats'.
 * @param {boolean} exact Whether the repeater needs to be matched exactly
 * @param {number} index The row index.
 * @param {string} binding The column binding, e.g. '{{cat.name}}'.
 * @param {Element} using The scope of the search.
 * @param {string} rootSelector The selector to use for the root app element.
 *
 * @return {Array.<Element>} The element in an array.
 */
functions.findRepeaterElement = function(repeater, exact, index, binding, using, rootSelector) {
  function repeaterMatch(ngRepeat, repeater, exact) {
    if (exact) {
      return ngRepeat.split(' track by ')[0].split(' as ')[0].split('|')[0].
          trim() == repeater;
    } else {
      return ngRepeat.indexOf(repeater) != -1;
    }
  }

  var matches = [];
  var root = document.querySelector(rootSelector || 'body');
  using = using || document;

  var rows = [];
  var prefixes = ['ng-', 'ng_', 'data-ng-', 'x-ng-', 'ng\\:'];
  for (var p = 0; p < prefixes.length; ++p) {
    var attr = prefixes[p] + 'repeat';
    var repeatElems = using.querySelectorAll('[' + attr + ']');
    attr = attr.replace(/\\/g, '');
    for (var i = 0; i < repeatElems.length; ++i) {
      if (repeaterMatch(repeatElems[i].getAttribute(attr), repeater, exact)) {
        rows.push(repeatElems[i]);
      }
    }
  }
  /* multiRows is an array of arrays, where each inner array contains
     one row of elements. */
  var multiRows = [];
  for (var p = 0; p < prefixes.length; ++p) {
    var attr = prefixes[p] + 'repeat-start';
    var repeatElems = using.querySelectorAll('[' + attr + ']');
    attr = attr.replace(/\\/g, '');
    for (var i = 0; i < repeatElems.length; ++i) {
      if (repeaterMatch(repeatElems[i].getAttribute(attr), repeater, exact)) {
        var elem = repeatElems[i];
        var row = [];
        while (elem.nodeType != 8 || (elem.nodeValue &&
            !repeaterMatch(elem.nodeValue, repeater, exact))) {
          if (elem.nodeType == 1) {
            row.push(elem);
          }
          elem = elem.nextSibling;
        }
        multiRows.push(row);
      }
    }
  }
  var row = rows[index];
  var multiRow = multiRows[index];
  var bindings = [];
  if (row) {
    if (angular.getTestability) {
      matches.push.apply(
          matches,
          angular.getTestability(root).findBindings(row, binding));
    } else {
      if (row.className.indexOf('ng-binding') != -1) {
        bindings.push(row);
      }
      var childBindings = row.getElementsByClassName('ng-binding');
      for (var i = 0; i < childBindings.length; ++i) {
        bindings.push(childBindings[i]);
      }
    }
  }
  if (multiRow) {
    for (var i = 0; i < multiRow.length; ++i) {
      var rowElem = multiRow[i];
      if (angular.getTestability) {
        matches.push.apply(
            matches,
            angular.getTestability(root).findBindings(rowElem, binding));
      } else {
        if (rowElem.className.indexOf('ng-binding') != -1) {
          bindings.push(rowElem);
        }
        var childBindings = rowElem.getElementsByClassName('ng-binding');
        for (var j = 0; j < childBindings.length; ++j) {
          bindings.push(childBindings[j]);
        }
      }
    }
  }
  for (var i = 0; i < bindings.length; ++i) {
    var dataBinding = angular.element(bindings[i]).data('$binding');
    if (dataBinding) {
      var bindingName = dataBinding.exp || dataBinding[0].exp || dataBinding;
      if (bindingName.indexOf(binding) != -1) {
        matches.push(bindings[i]);
      }
    }
  }
  return matches;
};

/**
 * Find the elements in a column of an ng-repeat.
 *
 * @param {string} repeater The text of the repeater, e.g. 'cat in cats'.
 * @param {boolean} exact Whether the repeater needs to be matched exactly
 * @param {string} binding The column binding, e.g. '{{cat.name}}'.
 * @param {Element} using The scope of the search.
 * @param {string} rootSelector The selector to use for the root app element.
 *
 * @return {Array.<Element>} The elements in the column.
 */
functions.findRepeaterColumn = function(repeater, exact, binding, using, rootSelector) {
  function repeaterMatch(ngRepeat, repeater, exact) {
    if (exact) {
      return ngRepeat.split(' track by ')[0].split(' as ')[0].split('|')[0].
          trim() == repeater;
    } else {
      return ngRepeat.indexOf(repeater) != -1;
    }
  }

  var matches = [];
  var root = document.querySelector(rootSelector || 'body');
  using = using || document;

  var rows = [];
  var prefixes = ['ng-', 'ng_', 'data-ng-', 'x-ng-', 'ng\\:'];
  for (var p = 0; p < prefixes.length; ++p) {
    var attr = prefixes[p] + 'repeat';
    var repeatElems = using.querySelectorAll('[' + attr + ']');
    attr = attr.replace(/\\/g, '');
    for (var i = 0; i < repeatElems.length; ++i) {
      if (repeaterMatch(repeatElems[i].getAttribute(attr), repeater, exact)) {
        rows.push(repeatElems[i]);
      }
    }
  }
  /* multiRows is an array of arrays, where each inner array contains
     one row of elements. */
  var multiRows = [];
  for (var p = 0; p < prefixes.length; ++p) {
    var attr = prefixes[p] + 'repeat-start';
    var repeatElems = using.querySelectorAll('[' + attr + ']');
    attr = attr.replace(/\\/g, '');
    for (var i = 0; i < repeatElems.length; ++i) {
      if (repeaterMatch(repeatElems[i].getAttribute(attr), repeater, exact)) {
        var elem = repeatElems[i];
        var row = [];
        while (elem.nodeType != 8 || (elem.nodeValue &&
            !repeaterMatch(elem.nodeValue, repeater, exact))) {
          if (elem.nodeType == 1) {
            row.push(elem);
          }
          elem = elem.nextSibling;
        }
        multiRows.push(row);
      }
    }
  }
  var bindings = [];
  for (var i = 0; i < rows.length; ++i) {
    if (angular.getTestability) {
      matches.push.apply(
          matches,
          angular.getTestability(root).findBindings(rows[i], binding));
    } else {
      if (rows[i].className.indexOf('ng-binding') != -1) {
        bindings.push(rows[i]);
      }
      var childBindings = rows[i].getElementsByClassName('ng-binding');
      for (var k = 0; k < childBindings.length; ++k) {
        bindings.push(childBindings[k]);
      }
    }
  }
  for (var i = 0; i < multiRows.length; ++i) {
    for (var j = 0; j < multiRows[i].length; ++j) {
      if (angular.getTestability) {
        matches.push.apply(
            matches,
            angular.getTestability(root).findBindings(multiRows[i][j], binding));
      } else {
        var elem = multiRows[i][j];
        if (elem.className.indexOf('ng-binding') != -1) {
          bindings.push(elem);
        }
        var childBindings = elem.getElementsByClassName('ng-binding');
        for (var k = 0; k < childBindings.length; ++k) {
          bindings.push(childBindings[k]);
        }
      }
    }
  }
  for (var j = 0; j < bindings.length; ++j) {
    var dataBinding = angular.element(bindings[j]).data('$binding');
    if (dataBinding) {
      var bindingName = dataBinding.exp || dataBinding[0].exp || dataBinding;
      if (bindingName.indexOf(binding) != -1) {
        matches.push(bindings[j]);
      }
    }
  }
  return matches;
};

/**
 * Find elements by model name.
 *
 * @param {string} model The model name.
 * @param {Element} using The scope of the search.
 * @param {string} rootSelector The selector to use for the root app element.
 *
 * @return {Array.<Element>} The matching elements.
 */
functions.findByModel = function(model, using, rootSelector) {
  var root = document.querySelector(rootSelector || 'body');
  using = using || document;

  if (angular.getTestability) {
    return angular.getTestability(root).
        findModels(using, model, true);
  }
  var prefixes = ['ng-', 'ng_', 'data-ng-', 'x-ng-', 'ng\\:'];
  for (var p = 0; p < prefixes.length; ++p) {
    var selector = '[' + prefixes[p] + 'model="' + model + '"]';
    var elements = using.querySelectorAll(selector);
    if (elements.length) {
      return elements;
    }
  }
};

/**
 * Find elements by options.
 *
 * @param {string} optionsDescriptor The descriptor for the option
 *     (i.e. fruit for fruit in fruits).
 * @param {Element} using The scope of the search.
 *
 * @return {Array.<Element>} The matching elements.
 */
functions.findByOptions = function(optionsDescriptor, using) {
  using = using || document;

  var prefixes = ['ng-', 'ng_', 'data-ng-', 'x-ng-', 'ng\\:'];
  for (var p = 0; p < prefixes.length; ++p) {
    var selector = '[' + prefixes[p] + 'options="' + optionsDescriptor + '"] option';
    var elements = using.querySelectorAll(selector);
    if (elements.length) {
      return elements;
    }
  }
};

/**
 * Find buttons by textual content.
 *
 * @param {string} searchText The exact text to match.
 * @param {Element} using The scope of the search.
 *
 * @return {Array.<Element>} The matching elements.
 */
functions.findByButtonText = function(searchText, using) {
  using = using || document;

  var elements = using.querySelectorAll('button, input[type="button"], input[type="submit"]');
  var matches = [];
  for (var i = 0; i < elements.length; ++i) {
    var element = elements[i];
    var elementText;
    if (element.tagName.toLowerCase() == 'button') {
      elementText = element.innerText || element.textContent;
    } else {
      elementText = element.value;
    }
    if (elementText.trim() === searchText) {
      matches.push(element);
    }
  }

  return matches;
};

/**
 * Find buttons by textual content.
 *
 * @param {string} searchText The exact text to match.
 * @param {Element} using The scope of the search.
 *
 * @return {Array.<Element>} The matching elements.
 */
functions.findByPartialButtonText = function(searchText, using) {
  using = using || document;

  var elements = using.querySelectorAll('button, input[type="button"], input[type="submit"]');
  var matches = [];
  for (var i = 0; i < elements.length; ++i) {
    var element = elements[i];
    var elementText;
    if (element.tagName.toLowerCase() == 'button') {
      elementText = element.innerText || element.textContent;
    } else {
      elementText = element.value;
    }
    if (elementText.indexOf(searchText) > -1) {
      matches.push(element);
    }
  }

  return matches;
};

/**
 * Find elements by css selector and textual content.
 *
 * @param {string} cssSelector The css selector to match.
 * @param {string} searchText The exact text to match.
 * @param {Element} using The scope of the search.
 *
 * @return {Array.<Element>} An array of matching elements.
 */
functions.findByCssContainingText = function(cssSelector, searchText, using) {
  using = using || document;

  var elements = using.querySelectorAll(cssSelector);
  var matches = [];
  for (var i = 0; i < elements.length; ++i) {
    var element = elements[i];
    var elementText = element.textContent || element.innerText || '';
    if (elementText.indexOf(searchText) > -1) {
      matches.push(element);
    }
  }
  return matches;
};

/**
 * Tests whether the angular global variable is present on a page. Retries
 * in case the page is just loading slowly.
 *
 * Asynchronous.
 *
 * @param {number} attempts Number of times to retry.
 * @param {function} asyncCallback callback
 */
functions.testForAngular = function(attempts, asyncCallback) {
  var callback = function(args) {
    setTimeout(function() {
      asyncCallback(args);
    }, 0);
  };
  var check = function(n) {
    try {
      if (window.angular && window.angular.resumeBootstrap) {
        callback([true, null]);
      } else if (n < 1) {
        if (window.angular) {
          callback([false, 'angular never provided resumeBootstrap']);
        } else {
          callback([false, 'retries looking for angular exceeded']);
        }
      } else {
        window.setTimeout(function() {check(n - 1);}, 1000);
      }
    } catch (e) {
      callback([false, e]);
    }
  };
  check(attempts);
};

/**
 * Evalute an Angular expression in the context of a given element.
 *
 * @param {Element} element The element in whose scope to evaluate.
 * @param {string} expression The expression to evaluate.
 *
 * @return {?Object} The result of the evaluation.
 */
functions.evaluate = function(element, expression) {
  return angular.element(element).scope().$eval(expression);
};

functions.allowAnimations = function(element, value) {
  var ngElement = angular.element(element);
  if (ngElement.allowAnimations) {
    // AngularDart: $testability API.
    return ngElement.allowAnimations(value);
  } else {
    // AngularJS
    var enabledFn = ngElement.injector().get('$animate').enabled;
    return (value == null) ? enabledFn() : enabledFn(value);
  }
};

/**
 * Return the current url using $location.absUrl().
 *
 * @param {string} selector The selector housing an ng-app
 */
functions.getLocationAbsUrl = function(selector) {
  var el = document.querySelector(selector);
  if (angular.getTestability) {
    return angular.getTestability(el).
        getLocation();
  }
  return angular.element(el).injector().get('$location').absUrl();
};

/**
 * Browse to another page using in-page navigation.
 *
 * @param {string} selector The selector housing an ng-app
 * @param {string} url In page URL using the same syntax as $location.url(),
 *     /path?search=a&b=c#hash
 */
functions.setLocation = function(selector, url) {
  var el = document.querySelector(selector);
  if (angular.getTestability) {
    return angular.getTestability(el).
        setLocation(url);
  }
  var $injector = angular.element(el).injector();
  var $location = $injector.get('$location');
  var $rootScope = $injector.get('$rootScope');

  if (url !== $location.url()) {
    $location.url(url);
    $rootScope.$digest();
  }
};

/* Publish all the functions as strings to pass to WebDriver's
 * exec[Async]Script.  In addition, also include a script that will
 * install all the functions on window (for debugging.)
 *
 * We also wrap any exceptions thrown by a clientSideScripts function
 * that is not an instance of the Error type into an Error type.  If we
 * don't do so, then the resulting stack trace is completely unhelpful
 * and the exception message is just "unknown error."  These types of
 * exceptions are the common case for dart2js code.  This wrapping gives
 * us the Dart stack trace and exception message.
 */
var util = require('util');
var scriptsList = [];
var scriptFmt = (
    'try { return (%s).apply(this, arguments); }\n' +
    'catch(e) { throw (e instanceof Error) ? e : new Error(e); }');
for (var fnName in functions) {
  if (functions.hasOwnProperty(fnName)) {
    exports[fnName] = util.format(scriptFmt, functions[fnName]);
    scriptsList.push(util.format('%s: %s', fnName, functions[fnName]));
  }
}

exports.installInBrowser = (util.format(
    'window.clientSideScripts = {%s};', scriptsList.join(', ')));
