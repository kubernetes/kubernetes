/*
 (c) Ferron Hanse 2012
 https://github.com/ferronrsmith/anuglarjs-jasmine-matchers
 Released under the MIT license
*/


/*jslint nomen : true*/
/*jslint devel : true*/
/*jslint unparam : true */
/*jslint browser : true */
/*jslint bitwise : true*/
/*global describe, beforeEach, inject, module, angular, document, it, expect, $, jasmine, toJson */

/**
 Provides a comprehensive set of custom matchers for the Jasmine testing framework
 @class matchers
 @main matchers
 **/
beforeEach(function () {
    "use strict";
    var matchers = {},
        hlp = {},
        bjQuery = false;

    hlp.cssMatcher = function (presentClasses, absentClasses) {
        var self = this;
        return function () {
            var element = angular.element(self.actual), present = true, absent = false;

            angular.forEach(presentClasses.split(' '), function (className) {
                present = present && element.hasClass(className);
            });

            angular.forEach(absentClasses.split(' '), function (className) {
                absent = absent || element.hasClass(className);
            });

            self.message = function () {
                return "Expected to {0} have ".t(this.isNot ? "not" : "") + presentClasses +
                    (absentClasses ? (" and not have " + absentClasses + " ") : "") +
                    " but had " + element[0].className + ".";
            };
            return present && !absent;
        };
    };

    /**
     * Returns the index of an object in a given array
     * @method hpl.indexOf
     * @param array :- array object to be checked
     * @param obj :- object (value) to be checked for in the array
     * @return {number} index of the obj in the array
     */
    hlp.indexOf = function (array, obj) {
        var i;
        for (i = 0; i < array.length; i += 1) {
            if (obj === array[i]) {
                return i;
            }
        }
        return -1;
    };

    /**
     * Check if an object has a particular property matches the expected value
     * @method hpl.hasProperty
     * @param actualValue property value
     * @param expectedValue expected value
     * @return {boolean} boolean indicating if the values match
     */
    hlp.hasProperty = function (actualValue, expectedValue) {
        if (expectedValue === undefined) {
            return actualValue !== undefined;
        }
        return actualValue === expectedValue;
    };

    /**
     * Checks if a given element/JavaScript object matches the type
     * @method hpl.typeOf
     * @param actual Object to be checked for type comparison
     * @param type type to be matched
     * @return {boolean} boolean indicating if the type matches the object type
     */
    hlp.typeOf = function (actual, type) {
        return Object.prototype.toString.call(actual) === "[object " + type + "]";
    };

    /**
     * Checks if the a given word/phrase/substring is at the end of a string
     * @method hpl.endsWith
     * @param {String} haystack string to be search
     * @param needle {String} word/phrase/substring
     * @return {boolean} boolean indicating if the word/phrase/substring was found at the end of the string
     */
    hlp.endsWith = function (haystack, needle) {
        return haystack.substr(-needle.length) === needle;
    };

    /**
     * Checks if the a given word/phrase/substring is at the beginning of a string
     * @method hpl.endsWith
     * @param {String} haystack string to be search
     * @param needle {String} word/phrase/substring
     * @return {boolean} boolean indicating if the word/phrase/substring was found at the beginning of the string
     */
    hlp.startsWith = function (haystack, needle) {
        return haystack.substr(0, needle.length) === needle;
    };

    /**
     * Coverts a given object literal to an array
     * @method hlp.objToArray
     * @param obj - object literal
     * @return {Array} array representation of the object
     * @since 0.2 :- Removed $$hashKey check
     */
    hlp.objToArray = function (obj) {
        var arr = [], aDup = {};
        angular.copy(obj, aDup);
        angular.forEach(aDup, function (value, key) {
            arr.push(value);
        });
        return arr;
    };

    /**
     * Coverts a given a list of object literals to a flatten array
     * @method hlp.objListToArray
     * @param obj - object literals
     * @return {Array} flatten array representation of the objects
     */
    hlp.objListToArray = function (obj) {
        var res = [];
        angular.forEach(obj, function (value, key) {
            res = res.concat(hlp.objToArray(value));
        });
        return res;
    };

    hlp.isNumber = function (val) {
        return !isNaN(parseFloat(val)) && !hlp.typeOf(val, 'String');
    };

    /**
     * Message constant for jQuery
     * @type {string}
     */
    hlp.msg = {
        jQuery : "Error: jQuery not found. this matcher has a dependency on jQuery",
        date : {
            invalidType : 'Expected {0} & {1} to be a Date',
            nomatch : {
                Date : 'Expected {0} & {1} to match',
                part : "Invalid part : {0} entered"
            }
        }
    };

    hlp.dp = function (x) {
        return angular.mock.dump(arguments.length > 1 ? arguments : x);
    };

    /**
     * Returns isNot String
     * @param context
     * @param altText
     */
    hlp.isNot = function (context, altText) {
        altText = altText || "";
        return context.isNot ? "not " : altText;
    };

    String.prototype.t = function () {
        var args = arguments;
        return this.replace(/\{(\d+)\}/g, function (match, number) {
            return args[number] !== 'undefined' ? args[number] : match;
        });
    };

    /**
     * Check if jQuery is present
     * @return {boolean} boolean indicating if jQuery is present
     */
    bjQuery = (function () {
        return (window.$ !== undefined || window.jQuery !== undefined);
    }());

    // a check that allows the matchers to work with angular-scenario
    // NB: Not all matchers work with angualar-scenario and i have not done extensive testing on this
    if (this.addMatchers === undefined) {
        this.addMatchers = function (properties) {
            if (angular.scenario !== undefined && angular.isObject(properties)) {
                angular.forEach(properties, function (value, key) {
                    angular.scenario.matcher(key, value);
                });
            }
        };
    }

    matchers.toBeInvalid =  hlp.cssMatcher('ng-invalid', 'ng-valid');
    matchers.toBeValid =  hlp.cssMatcher('ng-valid', 'ng-invalid');
    matchers.toBeDirty =  hlp.cssMatcher('ng-dirty', 'ng-pristine');
    matchers.toBePristine = hlp.cssMatcher('ng-pristine', 'ng-dirty');
    matchers.toEqual = function (expected) {
        if (this.actual && this.actual.$$log) {
            if (typeof expected === 'string') {
                this.actual = this.actual.toString();
            } else {
                this.actual = this.actual.toArray();
            }
        }
        return jasmine.Matchers.prototype.toEqual.call(this, expected);
    };

    matchers.toEqualData = function (expected) {
        this.message = function () {
            return "Expected " + hlp.dp(this.actual) + " data {0} to Equal ".t(this.isNot ? "not" : "") + expected;
        };
        return angular.equals(this.actual, expected);
    };

    matchers.toEqualError = function (message) {
        this.message = function () {
            var expected;
            if (this.actual.message && this.actual.name === 'Error') {
                expected = angular.toJson(this.actual.message);
            } else {
                expected = angular.toJson(this.actual);
            }
            return "Expected " + expected + " to {0} be an Error with message ".t(this.isNot ? "not" : "") + angular.toJson(message);
        };
        return this.actual.name === 'Error' && this.actual.message === message;
    };

    matchers.toMatchError = function (messageRegexp) {
        this.message = function () {
            var expected;
            if (this.actual.message && this.actual.name === 'Error') {
                expected = angular.toJson(this.actual.message);
            } else {
                expected = angular.toJson(this.actual);
            }
            return "Expected " + expected + " to {0} match an Error with message ".t(this.isNot ? "not" : "") + angular.toJson(messageRegexp);
        };
        return this.actual.name === 'Error' && messageRegexp.test(this.actual.message);
    };

    matchers.toHaveBeenCalledOnce = function () {
        if (arguments.length > 0) {
            throw new Error('toHaveBeenCalledOnce does not take arguments, use toHaveBeenCalledWith');
        }

        if (!jasmine.isSpy(this.actual)) {
            throw new Error('Expected a spy, but got ' + jasmine.pp(this.actual) + '.');
        }

        this.message = function () {
            var msg = 'Expected spy ' + this.actual.identity + ' to have been called once, but was ',
                count = this.actual.callCount;
            return [
                count === 0 ? msg + 'never called.' : msg + 'called ' + count + ' times.',
                msg.replace('to have', 'not to have') + 'called once.'
            ];
        };

        return this.actual.callCount === 1;
    };

    matchers.toHaveBeenCalledOnceWith = function () {
        var expectedArgs = jasmine.util.argsToArray(arguments);

        if (!jasmine.isSpy(this.actual)) {
            throw new Error('Expected a spy, but got ' + jasmine.pp(this.actual) + '.');
        }

        this.message = function () {
            var result;
            if (this.actual.callCount !== 1) {
                if (this.actual.callCount === 0) {
                    result = [
                        'Expected spy ' + this.actual.identity + ' to have been called with ' +
                            jasmine.pp(expectedArgs) + ' but it was never called.',
                        'Expected spy ' + this.actual.identity + ' not to have been called with ' +
                            jasmine.pp(expectedArgs) + ' but it was.'
                    ];
                } else {
                    result = [
                        'Expected spy ' + this.actual.identity + ' to have been called with ' +
                            jasmine.pp(expectedArgs) + ' but it was never called.',
                        'Expected spy ' + this.actual.identity + ' not to have been called with ' +
                            jasmine.pp(expectedArgs) + ' but it was.'
                    ];
                }
            } else {
                result = [
                    'Expected spy ' + this.actual.identity + ' to have been called with ' +
                        jasmine.pp(expectedArgs) + ' but was called with ' + jasmine.pp(this.actual.argsForCall),
                    'Expected spy ' + this.actual.identity + ' not to have been called with ' +
                        jasmine.pp(expectedArgs) + ' but was called with ' + jasmine.pp(this.actual.argsForCall)
                ];
            }
            return result;
        };

        return this.actual.callCount === 1 && this.env.contains_(this.actual.argsForCall, expectedArgs);
    };

    matchers.toBeOneOf = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be one of '".t(this.isNot ? "not" : "") + hlp.dp(arguments) + "'.";
        };
        return hlp.indexOf(arguments, this.actual) !== -1;
    };

    matchers.toHaveClass = function (clazz) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} have class '".t(this.isNot ? "not" : "") + clazz + "'.";
        };
        return this.actual.hasClass ? this.actual.hasClass(clazz) : angular.element(this.actual).hasClass(clazz);
    };

    matchers.toHaveCss = function (css) {
        var prop; // css prop
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} have css '".t(this.isNot ? "not" : "") + hlp.dp(css) + "'.";
        };
        for (prop in css) {
            if (css.hasOwnProperty(prop)) {
                if (this.actual.css(prop) !== css[prop]) {
                    return false;
                }
            }
        }
        return true;
    };

    matchers.toMatchRegex = function (regex) {

        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} match '".t(this.isNot ? "not" : "") + regex;
        };

        var reg;
        if (hlp.typeOf(regex, "String")) {
            reg = new RegExp(regex);
        } else if (hlp.typeOf(regex, "RegExp")) {
            reg = regex;
        }
        return reg.test(this.actual);
    };

    matchers.toBeVisible = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be visible '".t(this.isNot ? "not" : "");
        };
        return this.actual.is(':visible');
    };

    matchers.toBeHidden = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be hidden '".t(this.isNot ? "not" : "");
        };
        return this.actual.is(':hidden');
    };

    matchers.toBeSelected = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be selected '".t(this.isNot ? "not" : "");
        };
        return this.actual.is(':selected');
    };

    matchers.toBeChecked = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be checked '".t(this.isNot ? "not" : "");
        };
        return this.actual.is(':checked');
    };

    matchers.toBeSameDate = function (date) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be equal to '".t(this.isNot ? "not" : "") + hlp.dp(date);
        };

        var actualDate = this.actual;
        return actualDate.getDate() === date.getDate() &&
            actualDate.getFullYear() === date.getFullYear() &&
            actualDate.getMonth() === date.getMonth() &&
            actualDate.getHours() === date.getHours() &&
            actualDate.getMinutes() === date.getMinutes() &&
            actualDate.getSeconds() === date.getSeconds();
    };

    matchers.toBeEmpty = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be empty '".t(this.isNot ? "not" : "");
        };
        return this.actual.is(':empty');
    };

    matchers.toBeEmptyString = function () {
        this.message = function () {
            return "Expected string '" + hlp.dp(this.actual) + "' to {0} be empty '".t(this.isNot ? "not" : "");
        };
        return hlp.typeOf(this.actual, 'String') && $.trim(this.actual).length === 0;
    };

    matchers.toExist = function () {
        this.message = function () {
            var msg = "";
            if (bjQuery) {
                msg = "Expected '" + hlp.dp(this.actual) + "' to {0} exists '".t(this.isNot ? "not" : "");
            } else {
                msg = hlp.msg.jQuery;
            }
            return msg;
        };
        return bjQuery ? $(document).find(this.actual).length : false;
    };

    matchers.toHaveAttr = function (attributeName, expectedAttributeValue) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} have attribute '".t(this.isNot ? "not" : "") + attributeName + "' with value "  + expectedAttributeValue + ".";
        };
        return hlp.hasProperty(this.actual.attr(attributeName), expectedAttributeValue);
    };

    matchers.toHaveProp = function (propertyName, expectedPropertyValue) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} have property '".t(this.isNot ? "not" : "") + expectedPropertyValue + "'.";
        };
        return hlp.hasProperty(this.actual.prop(propertyName), expectedPropertyValue);
    };

    matchers.toHaveId = function (id) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} have id '".t(this.isNot ? "not" : "") + id + "'.";
        };
        return this.actual.attr('id') === id;
    };

    matchers.toBeDisabled = function (selector) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be disabled '".t(this.isNot ? "not" : "") + hlp.dp(selector) + "'.";
        };
        return this.actual.is(':disabled');
    };

    matchers.toBeFocused = function (selector) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be focused '".t(this.isNot ? "not" : "") + hlp.dp(selector) + "'.";
        };
        return this.actual.is(':focus');
    };

    matchers.toHaveText = function (text) {
        if (!bjQuery) {
            return false;
        }

        this.message = function () {
            var msg = "";
            if (bjQuery) {
                msg = "Expected '" + hlp.dp(this.actual) + "' to {0} have text '".t(this.isNot ? "not" : "") + text + "'.";
            } else {
                msg = hlp.msg.jQuery;
            }
            return msg;
        };

        var trimmedText = $.trim(this.actual.text()), result;
        if (text && angular.isFunction(text.test)) {
            result = text.test(trimmedText);
        } else {
            result = trimmedText === text;
        }
        return result;
    };

    matchers.toHaveValue = function (value) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} have value '".t(this.isNot ? "not" : "") + value + "'.";
        };
        return this.actual.val() === value;
    };

    matchers.toHaveData = function (key, expectedValue) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} have data '" + expectedValue + "'.".t(this.isNot ? "not" : "");
        };
        return hlp.hasProperty(this.actual.data(key), expectedValue);
    };

    /**
     * Does not return true if subject is null
     * @return {Boolean}
     */
    matchers.toBeObject = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be an [Object]".t(this.isNot ? "not" : "");
        };
        return hlp.typeOf(this.actual, 'Object');
    };


    /**
     * @return {Boolean}
     */
    matchers.toBeArray = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be an [Array]".t(this.isNot ? "not" : "");
        };
        return hlp.typeOf(this.actual, 'Array');
    };

    /**
     * @return {Boolean}
     */
    matchers.toBeDate = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be a [Date]".t(this.isNot ? "not" : "");
        };
        return hlp.typeOf(this.actual, 'Date');
    };

    /**
     * @return {Boolean}
     */
    matchers.toBeBefore = function (date) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be before".t(this.isNot ? "not" : "") + hlp.dp(date);
        };
        return hlp.typeOf(this.actual, 'Date') && this.actual.getTime() < date.getTime();
    };

    /**
     * @return {Boolean}
     */
    matchers.toBeAfter = function (date) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be after".t(this.isNot ? "not" : "") + hlp.dp(date);
        };
        return hlp.typeOf(this.actual, 'Date') && this.actual.getTime() > date.getTime();
    };

    matchers.toBeIso8601Date = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be ISO8601 Date Format".t(this.isNot ? "not" : "");
        };
        return hlp.typeOf(this.actual, 'String')
            && this.actual.length >= 10
            && new Date(this.actual).toString() !== 'Invalid Date'
            && new Date(this.actual).toISOString().slice(0, this.actual.length) === this.actual;
    };

    /**
     * Asserts subject is an Array with a defined number of members
     * @param  {Number} size
     * @return {Boolean}
     */
    matchers.toBeArrayOfSize = function (size) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be an [Array] of size {1}".t(this.isNot ? "not" : "", size);
        };
        return hlp.typeOf(this.actual, 'Array') && this.actual.length === size;
    };

    /**
     * @return {Boolean}
     */
    matchers.toBeString = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be a [String]".t(this.isNot ? "not" : "");
        };
        return hlp.typeOf(this.actual, 'String');
    };

    /**
     * @return {Boolean}
     */
    matchers.toBeBoolean = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to {0} be Boolean".t(this.isNot ? "not" : "");
        };
        return hlp.typeOf(this.actual, 'Boolean');
    };


    /**
     * @return {Boolean}
     */
    matchers.toBeNonEmptyString = function () {
        if (!bjQuery) {
            return false;
        }

        this.message = function () {
            var msg = "";
            if (bjQuery) {
                msg = "Expected '" + hlp.dp(this.actual) + "' to " + hlp.isNot(this, "") + "be a non empty string ";
            } else {
                msg = hlp.msg.jQuery;
            }
            return msg;
        };
        return hlp.typeOf(this.actual, 'String') && $.trim(this.actual).length > 0;
    };

    /**
     * @return {Boolean}
     */
    matchers.toBeNumber = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to " + hlp.isNot(this, "") + "be a [Number]";
        };
        return hlp.isNumber(this.actual);
    };

    matchers.toBeEvenNumber = function () {
        this.message = function () {
            return "Expected " + hlp.dp(this.actual) + " to " + hlp.isNot(this, "") + "be an even number";
        };
        return hlp.isNumber(this.actual) && this.actual % 2 === 0;
    };

    matchers.toBeOddNumber = function () {
        this.message = function () {
            return "Expected " + hlp.dp(this.actual) + " to " + hlp.isNot(this, "") + "be an odd number";
        };
        return hlp.isNumber(this.actual) && this.actual % 2 !== 0;
    };

    matchers.toBeNaN = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to " + hlp.isNot(this, "") + "be a [NaN]";
        };
        return isNaN(this.actual);
    };

    /**
     * @return {Boolean}
     */
    matchers.toBeFunction = function () {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to " + hlp.isNot(this, "") + "be a [Function]";
        };
        return hlp.typeOf(this.actual, 'Function');
    };

    matchers.toHaveLength = function (length) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to " + hlp.isNot(this, "") + "have a length of " + length;
        };
        return this.actual.length === length;
    };

    matchers.toStartWith = function (value) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + " " + hlp.isNot(this, "") + "to start with " + value;
        };
        return hlp.startsWith(this.actual, value);
    };

    matchers.toEndWith = function (value) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + " " + hlp.isNot(this, "") + "' to end with " + value;
        };
        return hlp.endsWith(this.actual, value);
    };

    matchers.toContainOnce = function (value) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to contain only one " + value;
        };
        var actual = this.actual, containsOnce = false, firstFoundAt;
        if (actual) {
            firstFoundAt = actual.indexOf(value);
            containsOnce = firstFoundAt !== -1 && firstFoundAt === actual.lastIndexOf(value);
        }
        return containsOnce;
    };

    matchers.toContainSelector = function (selector) {
        this.message = function () {
            return "Expected '" + hlp.dp(this.actual) + "' to have contain '" + hlp.dp(selector) + "'.";
        };
        return this.actual.find(selector).length;
    };

    /**
     * @return {boolean}
     */
    matchers.toBeUniqueArray = function () {
        // iterate over the array, adding unique elements to o
        var arr = this.actual, i, len = this.actual.length, o = [];
        this.message = function () {
            return "Expected " + hlp.dp(this.actual) + " values {0} to be unique".t(this.isNot ? "not" : "");
        };
        for (i = 0; i < len; i += 1) {
            if (hlp.indexOf(o, arr[i]) === -1) {
                o.push(arr[i]);
            } else {
                return false;
            }
        }
        return true;
    };

    matchers.toHaveMatchingAtrr = function (attr, obj) {
        var arr = hlp.objListToArray(obj),
            result = true,
            temp = this.actual,
            iter = 0,
            len = this.actual.length;

        // can't compare arrays of different lengths
        if (this.actual.length !== arr.length) {
            return false;
        }

        for (iter = 0; iter < len; iter += 1) {
            result &= temp.eq(iter).attr(attr) === arr[iter];
        }

        this.message = function () {
            var message;
            if (this.actual.length === arr.length) {
                message = "Expected '" + hlp.dp(this.actual) + "' elements to have attributes " + hlp.dp(arr) + " " + hlp.dp(arr);
            } else {
                message = "Can't compare obj properties of length " + arr.length + " with element collection of length " + this.actual.length;
            }
            return message;
        };

        return result;
    };

    /**
     *
     * @method matchers.toMatchDatePart
     * @param oDate {Date} Date to be compared
     * @param {String} part specific part/property of the date you want to be compared </br
     *        <br />
     *        <b>Currently supported parts are listed below :</b>
     *        <ul>
     *            <li>date</li>
     *            <li>day</li>
     *            <li>month</li>
     *            <li>year</li>
     *            <li>milliseconds</li>
     *            <li>minutes</li>
     *            <li>seconds</li>
     *            <li>hours</li>
     *            <li>time</li>
     *        </ul>
     *  e.g usages :expect(date).toMatchDatePart(date, 'day');
     * @beta
     */
    matchers.toMatchDatePart = function (oDate, part) {
        var cDate = this.actual,
            msg,
            result;
        if (hlp.typeOf(cDate, 'Date') && hlp.typeOf(oDate, 'Date')) {
            switch (part) {
            case 'date':
                result = cDate.getDate() === oDate.getDate();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getDate()), hlp.dp(oDate.getDate()));
                break;
            case 'day':
                result = cDate.getDay() === oDate.getDay();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getDay()), hlp.dp(oDate.getDay()));
                break;
            case 'month':
                result = cDate.getMonth() === oDate.getMonth();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getMonth()), hlp.dp(oDate.getMonth()));
                break;
            case 'year':
                result = cDate.getFullYear() === oDate.getFullYear();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getFullYear()), hlp.dp(oDate.getFullYear()));
                break;
            case 'milliseconds':
                result = cDate.getMilliseconds() === oDate.getMilliseconds();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getMilliseconds()), hlp.dp(oDate.getMilliseconds()));
                break;
            case 'seconds':
                result = cDate.getSeconds() === oDate.getSeconds();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getSeconds()), hlp.dp(oDate.getSeconds()));
                break;
            case 'minutes':
                result = cDate.getMinutes() === oDate.getMinutes();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getMinutes()), hlp.dp(oDate.getMinutes()));
                break;
            case 'hours':
                result = cDate.getHours() === oDate.getHours();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getHours()), hlp.dp(oDate.getHours()));
                break;
            case 'time':
                result = cDate.getTime() === oDate.getTime();
                msg = hlp.msg.date.nomatch.Date.t(hlp.dp(cDate.getTime()), hlp.dp(oDate.getTime()));
                break;
            default:
                msg = hlp.msg.date.nomatch.part.t(part);
            }

        } else {
            msg = hlp.msg.date.invalidType.t(hlp.dp(cDate), hlp.dp(oDate));
            result = false;
        }
        this.message = function () { return msg; };
        return result;
    };

    // aliases
    this.addMatchers(matchers);

    // Keep a reference to the original matchers, for tests
    jasmine.__angular_jasmine_matchers__ = matchers;
});