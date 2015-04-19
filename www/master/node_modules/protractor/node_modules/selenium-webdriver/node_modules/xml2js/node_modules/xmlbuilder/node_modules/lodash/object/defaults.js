var arrayCopy = require('../internal/arrayCopy'),
    assign = require('./assign'),
    assignDefaults = require('../internal/assignDefaults');

/**
 * Assigns own enumerable properties of source object(s) to the destination
 * object for all destination properties that resolve to `undefined`. Once a
 * property is set, additional values of the same property are ignored.
 *
 * @static
 * @memberOf _
 * @category Object
 * @param {Object} object The destination object.
 * @param {...Object} [sources] The source objects.
 * @returns {Object} Returns `object`.
 * @example
 *
 * _.defaults({ 'user': 'barney' }, { 'age': 36 }, { 'user': 'fred' });
 * // => { 'user': 'barney', 'age': 36 }
 */
function defaults(object) {
  if (object == null) {
    return object;
  }
  var args = arrayCopy(arguments);
  args.push(assignDefaults);
  return assign.apply(undefined, args);
}

module.exports = defaults;
