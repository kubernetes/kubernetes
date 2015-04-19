var baseForRight = require('../internal/baseForRight'),
    bindCallback = require('../internal/bindCallback'),
    keys = require('./keys');

/**
 * This method is like `_.forOwn` except that it iterates over properties of
 * `object` in the opposite order.
 *
 * @static
 * @memberOf _
 * @category Object
 * @param {Object} object The object to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @param {*} [thisArg] The `this` binding of `iteratee`.
 * @returns {Object} Returns `object`.
 * @example
 *
 * function Foo() {
 *   this.a = 1;
 *   this.b = 2;
 * }
 *
 * Foo.prototype.c = 3;
 *
 * _.forOwnRight(new Foo, function(value, key) {
 *   console.log(key);
 * });
 * // => logs 'b' and 'a' assuming `_.forOwn` logs 'a' and 'b'
 */
function forOwnRight(object, iteratee, thisArg) {
  iteratee = bindCallback(iteratee, thisArg, 3);
  return baseForRight(object, iteratee, keys);
}

module.exports = forOwnRight;
