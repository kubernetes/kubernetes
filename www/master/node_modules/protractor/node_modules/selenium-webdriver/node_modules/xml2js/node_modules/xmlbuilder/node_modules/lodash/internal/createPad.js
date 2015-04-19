var repeat = require('../string/repeat');

/** Native method references. */
var ceil = Math.ceil;

/* Native method references for those with the same name as other `lodash` methods. */
var nativeIsFinite = global.isFinite;

/**
 * Creates the pad required for `string` based on the given padding length.
 * The `chars` string may be truncated if the number of padding characters
 * exceeds the padding length.
 *
 * @private
 * @param {string} string The string to create padding for.
 * @param {number} [length=0] The padding length.
 * @param {string} [chars=' '] The string used as padding.
 * @returns {string} Returns the pad for `string`.
 */
function createPad(string, length, chars) {
  var strLength = string.length;
  length = +length;

  if (strLength >= length || !nativeIsFinite(length)) {
    return '';
  }
  var padLength = length - strLength;
  chars = chars == null ? ' ' : (chars + '');
  return repeat(chars, ceil(padLength / chars.length)).slice(0, padLength);
}

module.exports = createPad;
