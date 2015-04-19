/**
 * Used by `_.max` and `_.min` as the default callback for string values.
 *
 * @private
 * @param {string} string The string to inspect.
 * @returns {number} Returns the code unit of the first character of the string.
 */
function charAtCallback(string) {
  return string.charCodeAt(0);
}

module.exports = charAtCallback;
