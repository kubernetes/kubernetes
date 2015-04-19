var baseToString = require('../internal/baseToString'),
    createPad = require('../internal/createPad');

/**
 * Pads `string` on the left side if it is shorter then the given padding
 * length. The `chars` string may be truncated if the number of padding
 * characters exceeds the padding length.
 *
 * @static
 * @memberOf _
 * @category String
 * @param {string} [string=''] The string to pad.
 * @param {number} [length=0] The padding length.
 * @param {string} [chars=' '] The string used as padding.
 * @returns {string} Returns the padded string.
 * @example
 *
 * _.padLeft('abc', 6);
 * // => '   abc'
 *
 * _.padLeft('abc', 6, '_-');
 * // => '_-_abc'
 *
 * _.padLeft('abc', 3);
 * // => 'abc'
 */
function padLeft(string, length, chars) {
  string = baseToString(string);
  return string && (createPad(string, length, chars) + string);
}

module.exports = padLeft;
