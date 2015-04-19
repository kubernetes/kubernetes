'use strict';
module.exports = function (params) {
    params = params || {};
    var customPrefixes = params.customPrefixes || [];

    //find ng-something by default
    var prefix = 'ng-';
    //optionally add custom prefixes
    if (Array.isArray(customPrefixes) && customPrefixes.length) {
        var additions = customPrefixes.join('|');
        prefix += '|';
        prefix += additions;
    }

    //wrap around to insert into replace str later
    prefix = '(' + prefix + '){1}';

    //handle the following:
    //1. ' ng-'
    //2. '<ng-'
    //3. '</ng-'
    var allowedPreChars = '(\\s|<|<\/){1}';
    //build find/replace regex
    //$1 -> allowable pre-chars
    //$2 -> prefix match
    //$3 -> actual directive (partially)
    var replaceRegex = new RegExp(allowedPreChars + prefix + '(\\w+)', 'ig');

    //replace with data-ng-something
    var replaceStr = '$1data-$2$3';

    return {
        test: function (str) {
            if (typeof str !== 'string') {
                throw new Error('Input to test function must be a string');
            }
            return replaceRegex.test(str);
        },
        replace: function (str) {
            if (typeof str !== 'string') {
                throw new Error('Input to replace function must be a string');
            }
            return str.replace(replaceRegex, replaceStr);
        }
    };
};
