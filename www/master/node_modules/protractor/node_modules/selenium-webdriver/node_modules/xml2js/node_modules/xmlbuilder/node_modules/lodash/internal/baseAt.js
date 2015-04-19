var isIndex = require('./isIndex'),
    isLength = require('./isLength');

/**
 * The base implementation of `_.at` without support for strings and individual
 * key arguments.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {number[]|string[]} [props] The property names or indexes of elements to pick.
 * @returns {Array} Returns the new array of picked elements.
 */
function baseAt(collection, props) {
  var index = -1,
      length = collection.length,
      isArr = isLength(length),
      propsLength = props.length,
      result = Array(propsLength);

  while(++index < propsLength) {
    var key = props[index];
    if (isArr) {
      key = parseFloat(key);
      result[index] = isIndex(key, length) ? collection[key] : undefined;
    } else {
      result[index] = collection[key];
    }
  }
  return result;
}

module.exports = baseAt;
