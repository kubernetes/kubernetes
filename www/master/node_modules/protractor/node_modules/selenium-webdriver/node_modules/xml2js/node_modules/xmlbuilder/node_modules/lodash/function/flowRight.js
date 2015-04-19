var createComposer = require('../internal/createComposer');

/**
 * This method is like `_.flow` except that it creates a function that
 * invokes the provided functions from right to left.
 *
 * @static
 * @memberOf _
 * @alias backflow, compose
 * @category Function
 * @param {...Function} [funcs] Functions to invoke.
 * @returns {Function} Returns the new function.
 * @example
 *
 * function square(n) {
 *   return n * n;
 * }
 *
 * var addSquare = _.flowRight(square, _.add);
 * addSquare(1, 2);
 * // => 9
 */
var flowRight = createComposer(true);

module.exports = flowRight;
