var baseEach = require('./baseEach'),
    baseSortBy = require('./baseSortBy'),
    compareMultiple = require('./compareMultiple'),
    isLength = require('./isLength');

/**
 * The base implementation of `_.sortByOrder` without param guards.
 *
 * @private
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {string[]} props The property names to sort by.
 * @param {boolean[]} orders The sort orders of `props`.
 * @returns {Array} Returns the new sorted array.
 */
function baseSortByOrder(collection, props, orders) {
  var index = -1,
      length = collection.length,
      result = isLength(length) ? Array(length) : [];

  baseEach(collection, function(value) {
    var length = props.length,
        criteria = Array(length);

    while (length--) {
      criteria[length] = value == null ? undefined : value[props[length]];
    }
    result[++index] = { 'criteria': criteria, 'index': index, 'value': value };
  });

  return baseSortBy(result, function(object, other) {
    return compareMultiple(object, other, orders);
  });
}

module.exports = baseSortByOrder;
