// Copyright 2007 The Closure Library Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Python style iteration utilities.
 * @author arv@google.com (Erik Arvidsson)
 */


goog.provide('goog.iter');
goog.provide('goog.iter.Iterable');
goog.provide('goog.iter.Iterator');
goog.provide('goog.iter.StopIteration');

goog.require('goog.array');
goog.require('goog.asserts');
goog.require('goog.functions');
goog.require('goog.math');


/**
 * @typedef {goog.iter.Iterator|{length:number}|{__iterator__}}
 */
goog.iter.Iterable;


// For script engines that already support iterators.
if ('StopIteration' in goog.global) {
  /**
   * Singleton Error object that is used to terminate iterations.
   * @type {Error}
   */
  goog.iter.StopIteration = goog.global['StopIteration'];
} else {
  /**
   * Singleton Error object that is used to terminate iterations.
   * @type {Error}
   * @suppress {duplicate}
   */
  goog.iter.StopIteration = Error('StopIteration');
}



/**
 * Class/interface for iterators.  An iterator needs to implement a {@code next}
 * method and it needs to throw a {@code goog.iter.StopIteration} when the
 * iteration passes beyond the end.  Iterators have no {@code hasNext} method.
 * It is recommended to always use the helper functions to iterate over the
 * iterator or in case you are only targeting JavaScript 1.7 for in loops.
 * @constructor
 * @template VALUE
 */
goog.iter.Iterator = function() {};


/**
 * Returns the next value of the iteration.  This will throw the object
 * {@see goog.iter#StopIteration} when the iteration passes the end.
 * @return {VALUE} Any object or value.
 */
goog.iter.Iterator.prototype.next = function() {
  throw goog.iter.StopIteration;
};


/**
 * Returns the {@code Iterator} object itself.  This is used to implement
 * the iterator protocol in JavaScript 1.7
 * @param {boolean=} opt_keys  Whether to return the keys or values. Default is
 *     to only return the values.  This is being used by the for-in loop (true)
 *     and the for-each-in loop (false).  Even though the param gives a hint
 *     about what the iterator will return there is no guarantee that it will
 *     return the keys when true is passed.
 * @return {!goog.iter.Iterator.<VALUE>} The object itself.
 */
goog.iter.Iterator.prototype.__iterator__ = function(opt_keys) {
  return this;
};


/**
 * Returns an iterator that knows how to iterate over the values in the object.
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable  If the
 *     object is an iterator it will be returned as is.  If the object has an
 *     {@code __iterator__} method that will be called to get the value
 *     iterator.  If the object is an array-like object we create an iterator
 *     for that.
 * @return {!goog.iter.Iterator.<VALUE>} An iterator that knows how to iterate
 *     over the values in {@code iterable}.
 * @template VALUE
 */
goog.iter.toIterator = function(iterable) {
  if (iterable instanceof goog.iter.Iterator) {
    return iterable;
  }
  if (typeof iterable.__iterator__ == 'function') {
    return iterable.__iterator__(false);
  }
  if (goog.isArrayLike(iterable)) {
    var i = 0;
    var newIter = new goog.iter.Iterator;
    newIter.next = function() {
      while (true) {
        if (i >= iterable.length) {
          throw goog.iter.StopIteration;
        }
        // Don't include deleted elements.
        if (!(i in iterable)) {
          i++;
          continue;
        }
        return iterable[i++];
      }
    };
    return newIter;
  }


  // TODO(arv): Should we fall back on goog.structs.getValues()?
  throw Error('Not implemented');
};


/**
 * Calls a function for each element in the iterator with the element of the
 * iterator passed as argument.
 *
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable  The iterator
 *     to iterate over. If the iterable is an object {@code toIterator} will be
 *     called on it.
 * @param {function(this:THIS,VALUE,undefined,goog.iter.Iterator.<VALUE>)|
 *         function(this:THIS,number,undefined,goog.iter.Iterator.<VALUE>)} f
 *     The function to call for every element.  This function takes 3 arguments
 *     (the element, undefined, and the iterator) and the return value is
 *     irrelevant.  The reason for passing undefined as the second argument is
 *     so that the same function can be used in {@see goog.array#forEach} as
 *     well as others.
 * @param {THIS=} opt_obj  The object to be used as the value of 'this' within
 *     {@code f}.
 * @template THIS, VALUE
 */
goog.iter.forEach = function(iterable, f, opt_obj) {
  if (goog.isArrayLike(iterable)) {
    /** @preserveTry */
    try {
      // NOTES: this passes the index number to the second parameter
      // of the callback contrary to the documentation above.
      goog.array.forEach(/** @type {goog.array.ArrayLike} */(iterable), f,
                         opt_obj);
    } catch (ex) {
      if (ex !== goog.iter.StopIteration) {
        throw ex;
      }
    }
  } else {
    iterable = goog.iter.toIterator(iterable);
    /** @preserveTry */
    try {
      while (true) {
        f.call(opt_obj, iterable.next(), undefined, iterable);
      }
    } catch (ex) {
      if (ex !== goog.iter.StopIteration) {
        throw ex;
      }
    }
  }
};


/**
 * Calls a function for every element in the iterator, and if the function
 * returns true adds the element to a new iterator.
 *
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     to iterate over.
 * @param {
 *     function(this:THIS,VALUE,undefined,goog.iter.Iterator.<VALUE>):boolean} f
 *     The function to call for every element. This function takes 3 arguments
 *     (the element, undefined, and the iterator) and should return a boolean.
 *     If the return value is true the element will be included in the returned
 *     iterator.  If it is false the element is not included.
 * @param {THIS=} opt_obj The object to be used as the value of 'this' within
 *     {@code f}.
 * @return {!goog.iter.Iterator.<VALUE>} A new iterator in which only elements
 *     that passed the test are present.
 * @template THIS, VALUE
 */
goog.iter.filter = function(iterable, f, opt_obj) {
  var iterator = goog.iter.toIterator(iterable);
  var newIter = new goog.iter.Iterator;
  newIter.next = function() {
    while (true) {
      var val = iterator.next();
      if (f.call(opt_obj, val, undefined, iterator)) {
        return val;
      }
    }
  };
  return newIter;
};


/**
 * Calls a function for every element in the iterator, and if the function
 * returns false adds the element to a new iterator.
 *
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     to iterate over.
 * @param {
 *     function(this:THIS,VALUE,undefined,goog.iter.Iterator.<VALUE>):boolean} f
 *     The function to call for every element. This function takes 3 arguments
 *     (the element, undefined, and the iterator) and should return a boolean.
 *     If the return value is false the element will be included in the returned
 *     iterator.  If it is true the element is not included.
 * @param {THIS=} opt_obj The object to be used as the value of 'this' within
 *     {@code f}.
 * @return {!goog.iter.Iterator.<VALUE>} A new iterator in which only elements
 *     that did not pass the test are present.
 * @template THIS, VALUE
 */
goog.iter.filterFalse = function(iterable, f, opt_obj) {
  return goog.iter.filter(iterable, goog.functions.not(f), opt_obj);
};


/**
 * Creates a new iterator that returns the values in a range.  This function
 * can take 1, 2 or 3 arguments:
 * <pre>
 * range(5) same as range(0, 5, 1)
 * range(2, 5) same as range(2, 5, 1)
 * </pre>
 *
 * @param {number} startOrStop  The stop value if only one argument is provided.
 *     The start value if 2 or more arguments are provided.  If only one
 *     argument is used the start value is 0.
 * @param {number=} opt_stop  The stop value.  If left out then the first
 *     argument is used as the stop value.
 * @param {number=} opt_step  The number to increment with between each call to
 *     next.  This can be negative.
 * @return {!goog.iter.Iterator.<number>} A new iterator that returns the values
 *     in the range.
 */
goog.iter.range = function(startOrStop, opt_stop, opt_step) {
  var start = 0;
  var stop = startOrStop;
  var step = opt_step || 1;
  if (arguments.length > 1) {
    start = startOrStop;
    stop = opt_stop;
  }
  if (step == 0) {
    throw Error('Range step argument must not be zero');
  }

  var newIter = new goog.iter.Iterator;
  newIter.next = function() {
    if (step > 0 && start >= stop || step < 0 && start <= stop) {
      throw goog.iter.StopIteration;
    }
    var rv = start;
    start += step;
    return rv;
  };
  return newIter;
};


/**
 * Joins the values in a iterator with a delimiter.
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     to get the values from.
 * @param {string} deliminator  The text to put between the values.
 * @return {string} The joined value string.
 * @template VALUE
 */
goog.iter.join = function(iterable, deliminator) {
  return goog.iter.toArray(iterable).join(deliminator);
};


/**
 * For every element in the iterator call a function and return a new iterator
 * with that value.
 *
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterator to iterate over.
 * @param {
 *     function(this:THIS,VALUE,undefined,!goog.iter.Iterator.<VALUE>):RESULT} f
 *     The function to call for every element.  This function takes 3 arguments
 *     (the element, undefined, and the iterator) and should return a new value.
 * @param {THIS=} opt_obj The object to be used as the value of 'this' within
 *     {@code f}.
 * @return {!goog.iter.Iterator.<RESULT>} A new iterator that returns the
 *     results of applying the function to each element in the original
 *     iterator.
 * @template THIS, VALUE, RESULT
 */
goog.iter.map = function(iterable, f, opt_obj) {
  var iterator = goog.iter.toIterator(iterable);
  var newIter = new goog.iter.Iterator;
  newIter.next = function() {
    var val = iterator.next();
    return f.call(opt_obj, val, undefined, iterator);
  };
  return newIter;
};


/**
 * Passes every element of an iterator into a function and accumulates the
 * result.
 *
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     to iterate over.
 * @param {function(this:THIS,VALUE,VALUE):VALUE} f The function to call for
 *     every element. This function takes 2 arguments (the function's previous
 *     result or the initial value, and the value of the current element).
 *     function(previousValue, currentElement) : newValue.
 * @param {VALUE} val The initial value to pass into the function on the first
 *     call.
 * @param {THIS=} opt_obj  The object to be used as the value of 'this' within
 *     f.
 * @return {VALUE} Result of evaluating f repeatedly across the values of
 *     the iterator.
 * @template THIS, VALUE
 */
goog.iter.reduce = function(iterable, f, val, opt_obj) {
  var rval = val;
  goog.iter.forEach(iterable, function(val) {
    rval = f.call(opt_obj, rval, val);
  });
  return rval;
};


/**
 * Goes through the values in the iterator. Calls f for each of these, and if
 * any of them returns true, this returns true (without checking the rest). If
 * all return false this will return false.
 *
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     object.
 * @param {
 *     function(this:THIS,VALUE,undefined,goog.iter.Iterator.<VALUE>):boolean} f
 *     The function to call for every value. This function takes 3 arguments
 *     (the value, undefined, and the iterator) and should return a boolean.
 * @param {THIS=} opt_obj The object to be used as the value of 'this' within
 *     {@code f}.
 * @return {boolean} true if any value passes the test.
 * @template THIS, VALUE
 */
goog.iter.some = function(iterable, f, opt_obj) {
  iterable = goog.iter.toIterator(iterable);
  /** @preserveTry */
  try {
    while (true) {
      if (f.call(opt_obj, iterable.next(), undefined, iterable)) {
        return true;
      }
    }
  } catch (ex) {
    if (ex !== goog.iter.StopIteration) {
      throw ex;
    }
  }
  return false;
};


/**
 * Goes through the values in the iterator. Calls f for each of these and if any
 * of them returns false this returns false (without checking the rest). If all
 * return true this will return true.
 *
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     object.
 * @param {
 *     function(this:THIS,VALUE,undefined,goog.iter.Iterator.<VALUE>):boolean} f
 *     The function to call for every value. This function takes 3 arguments
 *     (the value, undefined, and the iterator) and should return a boolean.
 * @param {THIS=} opt_obj The object to be used as the value of 'this' within
 *     {@code f}.
 * @return {boolean} true if every value passes the test.
 * @template THIS, VALUE
 */
goog.iter.every = function(iterable, f, opt_obj) {
  iterable = goog.iter.toIterator(iterable);
  /** @preserveTry */
  try {
    while (true) {
      if (!f.call(opt_obj, iterable.next(), undefined, iterable)) {
        return false;
      }
    }
  } catch (ex) {
    if (ex !== goog.iter.StopIteration) {
      throw ex;
    }
  }
  return true;
};


/**
 * Takes zero or more iterables and returns one iterator that will iterate over
 * them in the order chained.
 * @param {...!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} var_args Any
 *     number of iterable objects.
 * @return {!goog.iter.Iterator.<VALUE>} Returns a new iterator that will
 *     iterate over all the given iterables' contents.
 * @template VALUE
 */
goog.iter.chain = function(var_args) {
  var iterator = goog.iter.toIterator(arguments);
  var iter = new goog.iter.Iterator();
  var current = null;

  iter.next = function() {
    while (true) {
      if (current == null) {
        var it = iterator.next();
        current = goog.iter.toIterator(it);
      }
      try {
        return current.next();
      } catch (ex) {
        if (ex !== goog.iter.StopIteration) {
          throw ex;
        }
        current = null;
      }
    }
  };

  return iter;
};


/**
 * Takes a single iterable containing zero or more iterables and returns one
 * iterator that will iterate over each one in the order given.
 * @see http://docs.python.org/2/library/itertools.html#itertools.chain.from_iterable
 * @param {goog.iter.Iterable} iterable The iterable of iterables to chain.
 * @return {!goog.iter.Iterator.<VALUE>} Returns a new iterator that will
 *     iterate over all the contents of the iterables contained within
 *     {@code iterable}.
 * @template VALUE
 */
goog.iter.chainFromIterable = function(iterable) {
  return goog.iter.chain.apply(undefined, iterable);
};


/**
 * Builds a new iterator that iterates over the original, but skips elements as
 * long as a supplied function returns true.
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     object.
 * @param {
 *     function(this:THIS,VALUE,undefined,goog.iter.Iterator.<VALUE>):boolean} f
 *     The function to call for every value. This function takes 3 arguments
 *     (the value, undefined, and the iterator) and should return a boolean.
 * @param {THIS=} opt_obj The object to be used as the value of 'this' within
 *     {@code f}.
 * @return {!goog.iter.Iterator.<VALUE>} A new iterator that drops elements from
 *     the original iterator as long as {@code f} is true.
 * @template THIS, VALUE
 */
goog.iter.dropWhile = function(iterable, f, opt_obj) {
  var iterator = goog.iter.toIterator(iterable);
  var newIter = new goog.iter.Iterator;
  var dropping = true;
  newIter.next = function() {
    while (true) {
      var val = iterator.next();
      if (dropping && f.call(opt_obj, val, undefined, iterator)) {
        continue;
      } else {
        dropping = false;
      }
      return val;
    }
  };
  return newIter;
};


/**
 * Builds a new iterator that iterates over the original, but only as long as a
 * supplied function returns true.
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     object.
 * @param {
 *     function(this:THIS,VALUE,undefined,goog.iter.Iterator.<VALUE>):boolean} f
 *     The function to call for every value. This function takes 3 arguments
 *     (the value, undefined, and the iterator) and should return a boolean.
 * @param {THIS=} opt_obj This is used as the 'this' object in f when called.
 * @return {!goog.iter.Iterator.<VALUE>} A new iterator that keeps elements in
 *     the original iterator as long as the function is true.
 * @template THIS, VALUE
 */
goog.iter.takeWhile = function(iterable, f, opt_obj) {
  var iterator = goog.iter.toIterator(iterable);
  var newIter = new goog.iter.Iterator;
  var taking = true;
  newIter.next = function() {
    while (true) {
      if (taking) {
        var val = iterator.next();
        if (f.call(opt_obj, val, undefined, iterator)) {
          return val;
        } else {
          taking = false;
        }
      } else {
        throw goog.iter.StopIteration;
      }
    }
  };
  return newIter;
};


/**
 * Converts the iterator to an array
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterator
 *     to convert to an array.
 * @return {!Array.<VALUE>} An array of the elements the iterator iterates over.
 * @template VALUE
 */
goog.iter.toArray = function(iterable) {
  // Fast path for array-like.
  if (goog.isArrayLike(iterable)) {
    return goog.array.toArray(/** @type {!goog.array.ArrayLike} */(iterable));
  }
  iterable = goog.iter.toIterator(iterable);
  var array = [];
  goog.iter.forEach(iterable, function(val) {
    array.push(val);
  });
  return array;
};


/**
 * Iterates over two iterables and returns true if they contain the same
 * sequence of elements and have the same length.
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable1 The first
 *     iterable object.
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable2 The second
 *     iterable object.
 * @return {boolean} true if the iterables contain the same sequence of elements
 *     and have the same length.
 * @template VALUE
 */
goog.iter.equals = function(iterable1, iterable2) {
  var fillValue = {};
  var pairs = goog.iter.zipLongest(fillValue, iterable1, iterable2);
  return goog.iter.every(pairs, function(pair) {
    return pair[0] == pair[1];
  });
};


/**
 * Advances the iterator to the next position, returning the given default value
 * instead of throwing an exception if the iterator has no more entries.
 * @param {goog.iter.Iterator.<VALUE>|goog.iter.Iterable} iterable The iterable
 *     object.
 * @param {VALUE} defaultValue The value to return if the iterator is empty.
 * @return {VALUE} The next item in the iteration, or defaultValue if the
 *     iterator was empty.
 * @template VALUE
 */
goog.iter.nextOrValue = function(iterable, defaultValue) {
  try {
    return goog.iter.toIterator(iterable).next();
  } catch (e) {
    if (e != goog.iter.StopIteration) {
      throw e;
    }
    return defaultValue;
  }
};


/**
 * Cartesian product of zero or more sets.  Gives an iterator that gives every
 * combination of one element chosen from each set.  For example,
 * ([1, 2], [3, 4]) gives ([1, 3], [1, 4], [2, 3], [2, 4]).
 * @see http://docs.python.org/library/itertools.html#itertools.product
 * @param {...!goog.array.ArrayLike.<VALUE>} var_args Zero or more sets, as
 *     arrays.
 * @return {!goog.iter.Iterator.<!Array.<VALUE>>} An iterator that gives each
 *     n-tuple (as an array).
 * @template VALUE
 */
goog.iter.product = function(var_args) {
  var someArrayEmpty = goog.array.some(arguments, function(arr) {
    return !arr.length;
  });

  // An empty set in a cartesian product gives an empty set.
  if (someArrayEmpty || !arguments.length) {
    return new goog.iter.Iterator();
  }

  var iter = new goog.iter.Iterator();
  var arrays = arguments;

  // The first indices are [0, 0, ...]
  var indicies = goog.array.repeat(0, arrays.length);

  iter.next = function() {

    if (indicies) {
      var retVal = goog.array.map(indicies, function(valueIndex, arrayIndex) {
        return arrays[arrayIndex][valueIndex];
      });

      // Generate the next-largest indices for the next call.
      // Increase the rightmost index. If it goes over, increase the next
      // rightmost (like carry-over addition).
      for (var i = indicies.length - 1; i >= 0; i--) {
        // Assertion prevents compiler warning below.
        goog.asserts.assert(indicies);
        if (indicies[i] < arrays[i].length - 1) {
          indicies[i]++;
          break;
        }

        // We're at the last indices (the last element of every array), so
        // the iteration is over on the next call.
        if (i == 0) {
          indicies = null;
          break;
        }
        // Reset the index in this column and loop back to increment the
        // next one.
        indicies[i] = 0;
      }
      return retVal;
    }

    throw goog.iter.StopIteration;
  };

  return iter;
};


/**
 * Create an iterator to cycle over the iterable's elements indefinitely.
 * For example, ([1, 2, 3]) would return : 1, 2, 3, 1, 2, 3, ...
 * @see: http://docs.python.org/library/itertools.html#itertools.cycle.
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable object.
 * @return {!goog.iter.Iterator.<VALUE>} An iterator that iterates indefinitely
 *     over the values in {@code iterable}.
 * @template VALUE
 */
goog.iter.cycle = function(iterable) {
  var baseIterator = goog.iter.toIterator(iterable);

  // We maintain a cache to store the iterable elements as we iterate
  // over them. The cache is used to return elements once we have
  // iterated over the iterable once.
  var cache = [];
  var cacheIndex = 0;

  var iter = new goog.iter.Iterator();

  // This flag is set after the iterable is iterated over once
  var useCache = false;

  iter.next = function() {
    var returnElement = null;

    // Pull elements off the original iterator if not using cache
    if (!useCache) {
      try {
        // Return the element from the iterable
        returnElement = baseIterator.next();
        cache.push(returnElement);
        return returnElement;
      } catch (e) {
        // If an exception other than StopIteration is thrown
        // or if there are no elements to iterate over (the iterable was empty)
        // throw an exception
        if (e != goog.iter.StopIteration || goog.array.isEmpty(cache)) {
          throw e;
        }
        // set useCache to true after we know that a 'StopIteration' exception
        // was thrown and the cache is not empty (to handle the 'empty iterable'
        // use case)
        useCache = true;
      }
    }

    returnElement = cache[cacheIndex];
    cacheIndex = (cacheIndex + 1) % cache.length;

    return returnElement;
  };

  return iter;
};


/**
 * Creates an iterator that counts indefinitely from a starting value.
 * @see http://docs.python.org/2/library/itertools.html#itertools.count
 * @param {number=} opt_start The starting value. Default is 0.
 * @param {number=} opt_step The number to increment with between each call to
 *     next. Negative and floating point numbers are allowed. Default is 1.
 * @return {!goog.iter.Iterator.<number>} A new iterator that returns the values
 *     in the series.
 */
goog.iter.count = function(opt_start, opt_step) {
  var counter = opt_start || 0;
  var step = goog.isDef(opt_step) ? opt_step : 1;
  var iter = new goog.iter.Iterator();

  iter.next = function() {
    var returnValue = counter;
    counter += step;
    return returnValue;
  };

  return iter;
};


/**
 * Creates an iterator that returns the same object or value repeatedly.
 * @param {VALUE} value Any object or value to repeat.
 * @return {!goog.iter.Iterator.<VALUE>} A new iterator that returns the
 *     repeated value.
 * @template VALUE
 */
goog.iter.repeat = function(value) {
  var iter = new goog.iter.Iterator();

  iter.next = goog.functions.constant(value);

  return iter;
};


/**
 * Creates an iterator that returns running totals from the numbers in
 * {@code iterable}. For example, the array {@code [1, 2, 3, 4, 5]} yields
 * {@code 1 -> 3 -> 6 -> 10 -> 15}.
 * @see http://docs.python.org/3.2/library/itertools.html#itertools.accumulate
 * @param {!goog.iter.Iterable.<number>} iterable The iterable of numbers to
 *     accumulate.
 * @return {!goog.iter.Iterator.<number>} A new iterator that returns the
 *     numbers in the series.
 */
goog.iter.accumulate = function(iterable) {
  var iterator = goog.iter.toIterator(iterable);
  var total = 0;
  var iter = new goog.iter.Iterator();

  iter.next = function() {
    total += iterator.next();
    return total;
  };

  return iter;
};


/**
 * Creates an iterator that returns arrays containing the ith elements from the
 * provided iterables. The returned arrays will be the same size as the number
 * of iterables given in {@code var_args}. Once the shortest iterable is
 * exhausted, subsequent calls to {@code next()} will throw
 * {@code goog.iter.StopIteration}.
 * @see http://docs.python.org/2/library/itertools.html#itertools.izip
 * @param {...!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} var_args Any
 *     number of iterable objects.
 * @return {!goog.iter.Iterator.<!Array.<VALUE>>} A new iterator that returns
 *     arrays of elements from the provided iterables.
 * @template VALUE
 */
goog.iter.zip = function(var_args) {
  var args = arguments;
  var iter = new goog.iter.Iterator();

  if (args.length > 0) {
    var iterators = goog.array.map(args, goog.iter.toIterator);
    iter.next = function() {
      var arr = goog.array.map(iterators, function(it) {
        return it.next();
      });
      return arr;
    };
  }

  return iter;
};


/**
 * Creates an iterator that returns arrays containing the ith elements from the
 * provided iterables. The returned arrays will be the same size as the number
 * of iterables given in {@code var_args}. Shorter iterables will be extended
 * with {@code fillValue}. Once the longest iterable is exhausted, subsequent
 * calls to {@code next()} will throw {@code goog.iter.StopIteration}.
 * @see http://docs.python.org/2/library/itertools.html#itertools.izip_longest
 * @param {VALUE} fillValue The object or value used to fill shorter iterables.
 * @param {...!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} var_args Any
 *     number of iterable objects.
 * @return {!goog.iter.Iterator.<!Array.<VALUE>>} A new iterator that returns
 *     arrays of elements from the provided iterables.
 * @template VALUE
 */
goog.iter.zipLongest = function(fillValue, var_args) {
  var args = goog.array.slice(arguments, 1);
  var iter = new goog.iter.Iterator();

  if (args.length > 0) {
    var iterators = goog.array.map(args, goog.iter.toIterator);

    iter.next = function() {
      var iteratorsHaveValues = false;  // false when all iterators are empty.
      var arr = goog.array.map(iterators, function(it) {
        var returnValue;
        try {
          returnValue = it.next();
          // Iterator had a value, so we've not exhausted the iterators.
          // Set flag accordingly.
          iteratorsHaveValues = true;
        } catch (ex) {
          if (ex !== goog.iter.StopIteration) {
            throw ex;
          }
          returnValue = fillValue;
        }
        return returnValue;
      });

      if (!iteratorsHaveValues) {
        throw goog.iter.StopIteration;
      }
      return arr;
    };
  }

  return iter;
};


/**
 * Creates an iterator that filters {@code iterable} based on a series of
 * {@code selectors}. On each call to {@code next()}, one item is taken from
 * both the {@code iterable} and {@code selectors} iterators. If the item from
 * {@code selectors} evaluates to true, the item from {@code iterable} is given.
 * Otherwise, it is skipped. Once either {@code iterable} or {@code selectors}
 * is exhausted, subsequent calls to {@code next()} will throw
 * {@code goog.iter.StopIteration}.
 * @see http://docs.python.org/2/library/itertools.html#itertools.compress
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to filter.
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} selectors An
 *     iterable of items to be evaluated in a boolean context to determine if
 *     the corresponding element in {@code iterable} should be included in the
 *     result.
 * @return {!goog.iter.Iterator.<VALUE>} A new iterator that returns the
 *     filtered values.
 * @template VALUE
 */
goog.iter.compress = function(iterable, selectors) {
  var selectorIterator = goog.iter.toIterator(selectors);

  return goog.iter.filter(iterable, function() {
    return !!selectorIterator.next();
  });
};



/**
 * Implements the {@code goog.iter.groupBy} iterator.
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to group.
 * @param {function(...[VALUE]): KEY=} opt_keyFunc  Optional function for
 *     determining the key value for each group in the {@code iterable}. Default
 *     is the identity function.
 * @constructor
 * @extends {goog.iter.Iterator.<!Array>}
 * @template KEY, VALUE
 * @private
 */
goog.iter.GroupByIterator_ = function(iterable, opt_keyFunc) {

  /**
   * The iterable to group, coerced to an iterator.
   * @type {!goog.iter.Iterator}
   */
  this.iterator = goog.iter.toIterator(iterable);

  /**
   * A function for determining the key value for each element in the iterable.
   * If no function is provided, the identity function is used and returns the
   * element unchanged.
   * @type {function(...[VALUE]): KEY}
   */
  this.keyFunc = opt_keyFunc || goog.functions.identity;

  /**
   * The target key for determining the start of a group.
   * @type {KEY}
   */
  this.targetKey;

  /**
   * The current key visited during iteration.
   * @type {KEY}
   */
  this.currentKey;

  /**
   * The current value being added to the group.
   * @type {VALUE}
   */
  this.currentValue;
};
goog.inherits(goog.iter.GroupByIterator_, goog.iter.Iterator);


/** @override */
goog.iter.GroupByIterator_.prototype.next = function() {
  while (this.currentKey == this.targetKey) {
    this.currentValue = this.iterator.next();  // Exits on StopIteration
    this.currentKey = this.keyFunc(this.currentValue);
  }
  this.targetKey = this.currentKey;
  return [this.currentKey, this.groupItems_(this.targetKey)];
};


/**
 * Performs the grouping of objects using the given key.
 * @param {KEY} targetKey  The target key object for the group.
 * @return {!Array.<VALUE>} An array of grouped objects.
 * @private
 */
goog.iter.GroupByIterator_.prototype.groupItems_ = function(targetKey) {
  var arr = [];
  while (this.currentKey == targetKey) {
    arr.push(this.currentValue);
    try {
      this.currentValue = this.iterator.next();
    } catch (ex) {
      if (ex !== goog.iter.StopIteration) {
        throw ex;
      }
      break;
    }
    this.currentKey = this.keyFunc(this.currentValue);
  }
  return arr;
};


/**
 * Creates an iterator that returns arrays containing elements from the
 * {@code iterable} grouped by a key value. For iterables with repeated
 * elements (i.e. sorted according to a particular key function), this function
 * has a {@code uniq}-like effect. For example, grouping the array:
 * {@code [A, B, B, C, C, A]} produces
 * {@code [A, [A]], [B, [B, B]], [C, [C, C]], [A, [A]]}.
 * @see http://docs.python.org/2/library/itertools.html#itertools.groupby
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to group.
 * @param {function(...[VALUE]): KEY=} opt_keyFunc  Optional function for
 *     determining the key value for each group in the {@code iterable}. Default
 *     is the identity function.
 * @return {!goog.iter.Iterator.<!Array>} A new iterator that returns arrays of
 *     consecutive key and groups.
 * @template KEY, VALUE
 */
goog.iter.groupBy = function(iterable, opt_keyFunc) {
  return new goog.iter.GroupByIterator_(iterable, opt_keyFunc);
};


/**
 * Gives an iterator that gives the result of calling the given function
 * <code>f</code> with the arguments taken from the next element from
 * <code>iterable</code> (the elements are expected to also be iterables).
 *
 * Similar to {@see goog.iter#map} but allows the function to accept multiple
 * arguments from the iterable.
 *
 * @param {!goog.iter.Iterable.<!goog.iter.Iterable>} iterable The iterable of
 *     iterables to iterate over.
 * @param {function(this:THIS,...[*]):RESULT} f The function to call for every
 *     element.  This function takes N+2 arguments, where N represents the
 *     number of items from the next element of the iterable. The two
 *     additional arguments passed to the function are undefined and the
 *     iterator itself. The function should return a new value.
 * @param {THIS=} opt_obj The object to be used as the value of 'this' within
 *     {@code f}.
 * @return {!goog.iter.Iterator.<RESULT>} A new iterator that returns the
 *     results of applying the function to each element in the original
 *     iterator.
 * @template THIS, RESULT
 */
goog.iter.starMap = function(iterable, f, opt_obj) {
  var iterator = goog.iter.toIterator(iterable);
  var iter = new goog.iter.Iterator();

  iter.next = function() {
    var args = goog.iter.toArray(iterator.next());
    return f.apply(opt_obj, goog.array.concat(args, undefined, iterator));
  };

  return iter;
};


/**
 * Returns an array of iterators each of which can iterate over the values in
 * {@code iterable} without advancing the others.
 * @see http://docs.python.org/2/library/itertools.html#itertools.tee
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to tee.
 * @param {number=} opt_num  The number of iterators to create. Default is 2.
 * @return {!Array.<goog.iter.Iterator.<VALUE>>} An array of iterators.
 * @template VALUE
 */
goog.iter.tee = function(iterable, opt_num) {
  var iterator = goog.iter.toIterator(iterable);
  var num = goog.isNumber(opt_num) ? opt_num : 2;
  var buffers = goog.array.map(goog.array.range(num), function() {
    return [];
  });

  var addNextIteratorValueToBuffers = function() {
    var val = iterator.next();
    goog.array.forEach(buffers, function(buffer) {
      buffer.push(val);
    });
  };

  var createIterator = function(buffer) {
    // Each tee'd iterator has an associated buffer (initially empty). When a
    // tee'd iterator's buffer is empty, it calls
    // addNextIteratorValueToBuffers(), adding the next value to all tee'd
    // iterators' buffers, and then returns that value. This allows each
    // iterator to be advanced independently.
    var iter = new goog.iter.Iterator();

    iter.next = function() {
      if (goog.array.isEmpty(buffer)) {
        addNextIteratorValueToBuffers();
      }
      goog.asserts.assert(!goog.array.isEmpty(buffer));
      return buffer.shift();
    };

    return iter;
  };

  return goog.array.map(buffers, createIterator);
};


/**
 * Creates an iterator that returns arrays containing a count and an element
 * obtained from the given {@code iterable}.
 * @see http://docs.python.org/2/library/functions.html#enumerate
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to enumerate.
 * @param {number=} opt_start  Optional starting value. Default is 0.
 * @return {!goog.iter.Iterator.<!Array>} A new iterator containing count/item
 *     pairs.
 * @template VALUE
 */
goog.iter.enumerate = function(iterable, opt_start) {
  return goog.iter.zip(goog.iter.count(opt_start), iterable);
};


/**
 * Creates an iterator that returns the first {@code limitSize} elements from an
 * iterable. If this number is greater than the number of elements in the
 * iterable, all the elements are returned.
 * @see http://goo.gl/V0sihp Inspired by the limit iterator in Guava.
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to limit.
 * @param {number} limitSize  The maximum number of elements to return.
 * @return {!goog.iter.Iterator.<VALUE>} A new iterator containing
 *     {@code limitSize} elements.
 * @template VALUE
 */
goog.iter.limit = function(iterable, limitSize) {
  goog.asserts.assert(goog.math.isInt(limitSize) && limitSize >= 0);

  var iterator = goog.iter.toIterator(iterable);

  var iter = new goog.iter.Iterator();
  var remaining = limitSize;

  iter.next = function() {
    if (remaining-- > 0) {
      return iterator.next();
    }
    throw goog.iter.StopIteration;
  };

  return iter;
};


/**
 * Creates an iterator that is advanced {@code count} steps ahead. Consumed
 * values are silently discarded. If {@code count} is greater than the number
 * of elements in {@code iterable}, an empty iterator is returned. Subsequent
 * calls to {@code next()} will throw {@code goog.iter.StopIteration}.
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to consume.
 * @param {number} count  The number of elements to consume from the iterator.
 * @return {!goog.iter.Iterator.<VALUE>} An iterator advanced zero or more steps
 *     ahead.
 * @template VALUE
 */
goog.iter.consume = function(iterable, count) {
  goog.asserts.assert(goog.math.isInt(count) && count >= 0);

  var iterator = goog.iter.toIterator(iterable);

  while (count-- > 0) {
    goog.iter.nextOrValue(iterator, null);
  }

  return iterator;
};


/**
 * Creates an iterator that returns a range of elements from an iterable.
 * Similar to {@see goog.array#slice} but does not support negative indexes.
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to slice.
 * @param {number} start  The index of the first element to return.
 * @param {number=} opt_end  The index after the last element to return. If
 *     defined, must be greater than or equal to {@code start}.
 * @return {!goog.iter.Iterator.<VALUE>} A new iterator containing a slice of
 *     the original.
 * @template VALUE
 */
goog.iter.slice = function(iterable, start, opt_end) {
  goog.asserts.assert(goog.math.isInt(start) && start >= 0);

  var iterator = goog.iter.consume(iterable, start);

  if (goog.isNumber(opt_end)) {
    goog.asserts.assert(
        goog.math.isInt(/** @type {number} */ (opt_end)) && opt_end >= start);
    iterator = goog.iter.limit(iterator, opt_end - start /* limitSize */);
  }

  return iterator;
};


/**
 * Checks an array for duplicate elements.
 * @param {Array.<VALUE>|goog.array.ArrayLike} arr The array to check for
 *     duplicates.
 * @return {boolean} True, if the array contains duplicates, false otherwise.
 * @private
 * @template VALUE
 */
// TODO(user): Consider moving this into goog.array as a public function.
goog.iter.hasDuplicates_ = function(arr) {
  var deduped = [];
  goog.array.removeDuplicates(arr, deduped);
  return arr.length != deduped.length;
};


/**
 * Creates an iterator that returns permutations of elements in
 * {@code iterable}.
 *
 * Permutations are obtained by taking the Cartesian product of
 * {@code opt_length} iterables and filtering out those with repeated
 * elements. For example, the permutations of {@code [1,2,3]} are
 * {@code [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]}.
 * @see http://docs.python.org/2/library/itertools.html#itertools.permutations
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable from which to generate permutations.
 * @param {number=} opt_length Length of each permutation. If omitted, defaults
 *     to the length of {@code iterable}.
 * @return {!goog.iter.Iterator.<!Array.<VALUE>>} A new iterator containing the
 *     permutations of {@code iterable}.
 * @template VALUE
 */
goog.iter.permutations = function(iterable, opt_length) {
  var elements = goog.iter.toArray(iterable);
  var length = goog.isNumber(opt_length) ? opt_length : elements.length;

  var sets = goog.array.repeat(elements, length);
  var product = goog.iter.product.apply(undefined, sets);

  return goog.iter.filter(product, function(arr) {
    return !goog.iter.hasDuplicates_(arr);
  });
};


/**
 * Creates an iterator that returns combinations of elements from
 * {@code iterable}.
 *
 * Combinations are obtained by taking the {@see goog.iter#permutations} of
 * {@code iterable} and filtering those whose elements appear in the order they
 * are encountered in {@code iterable}. For example, the 3-length combinations
 * of {@code [0,1,2,3]} are {@code [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]}.
 * @see http://docs.python.org/2/library/itertools.html#itertools.combinations
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable from which to generate combinations.
 * @param {number} length The length of each combination.
 * @return {!goog.iter.Iterator.<!Array.<VALUE>>} A new iterator containing
 *     combinations from the {@code iterable}.
 * @template VALUE
 */
goog.iter.combinations = function(iterable, length) {
  var elements = goog.iter.toArray(iterable);
  var indexes = goog.iter.range(elements.length);
  var indexIterator = goog.iter.permutations(indexes, length);
  // sortedIndexIterator will now give arrays of with the given length that
  // indicate what indexes into "elements" should be returned on each iteration.
  var sortedIndexIterator = goog.iter.filter(indexIterator, function(arr) {
    return goog.array.isSorted(arr);
  });

  var iter = new goog.iter.Iterator();

  function getIndexFromElements(index) {
    return elements[index];
  }

  iter.next = function() {
    return goog.array.map(
        /** @type {!Array.<number>} */
        (sortedIndexIterator.next()), getIndexFromElements);
  };

  return iter;
};


/**
 * Creates an iterator that returns combinations of elements from
 * {@code iterable}, with repeated elements possible.
 *
 * Combinations are obtained by taking the Cartesian product of {@code length}
 * iterables and filtering those whose elements appear in the order they are
 * encountered in {@code iterable}. For example, the 2-length combinations of
 * {@code [1,2,3]} are {@code [[1,1], [1,2], [1,3], [2,2], [2,3], [3,3]]}.
 * @see http://docs.python.org/2/library/itertools.html#itertools.combinations_with_replacement
 * @see http://en.wikipedia.org/wiki/Combination#Number_of_combinations_with_repetition
 * @param {!goog.iter.Iterator.<VALUE>|!goog.iter.Iterable} iterable The
 *     iterable to combine.
 * @param {number} length The length of each combination.
 * @return {!goog.iter.Iterator.<!Array.<VALUE>>} A new iterator containing
 *     combinations from the {@code iterable}.
 * @template VALUE
 */
goog.iter.combinationsWithReplacement = function(iterable, length) {
  var elements = goog.iter.toArray(iterable);
  var indexes = goog.array.range(elements.length);
  var sets = goog.array.repeat(indexes, length);
  var indexIterator = goog.iter.product.apply(undefined, sets);
  // sortedIndexIterator will now give arrays of with the given length that
  // indicate what indexes into "elements" should be returned on each iteration.
  var sortedIndexIterator = goog.iter.filter(indexIterator, function(arr) {
    return goog.array.isSorted(arr);
  });

  var iter = new goog.iter.Iterator();

  function getIndexFromElements(index) {
    return elements[index];
  }

  iter.next = function() {
    return goog.array.map(
        /** @type {!Array.<number>} */
        (sortedIndexIterator.next()), getIndexFromElements);
  };

  return iter;
};
