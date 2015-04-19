var sentenceCase = require('sentence-case');

/**
 * Snake case a string.
 *
 * @param  {String} str
 * @param  {String} [locale]
 * @return {String}
 */
module.exports = function (str, locale) {
  return sentenceCase(str, locale, '_');
};
