var camelCase      = require('camel-case');
var upperCaseFirst = require('upper-case-first');

/**
 * Pascal case a string.
 *
 * @param  {String} string
 * @param  {String} [locale]
 * @return {String}
 */
module.exports = function (string, locale) {
  return upperCaseFirst(camelCase(string, locale), locale);
};
