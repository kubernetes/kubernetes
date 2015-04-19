var upperCase = require('upper-case');
var lowerCase = require('lower-case');

/**
 * Swap the case of a string. Manually iterate over every character and check
 * instead of replacing certain characters for better unicode support.
 *
 * @param  {String} str
 * @param  {String} [locale]
 * @return {String}
 */
module.exports = function (str, locale) {
  if (str == null) {
    return '';
  }

  var result = '';

  for (var i = 0; i < str.length; i++) {
    var c = str[i];
    var u = upperCase(c, locale);

    result += u === c ? lowerCase(c, locale) : u;
  }

  return result;
};
