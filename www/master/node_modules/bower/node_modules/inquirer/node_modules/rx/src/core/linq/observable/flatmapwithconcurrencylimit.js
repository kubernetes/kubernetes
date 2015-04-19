  function flatMapWithMaxConcurrent(source, maxConcurrent, selector, thisArg) {
    return source.map(function (x, i) {
      var result = selector.call(thisArg, x, i);
      isPromise(result) && (result = observableFromPromise(result));
      (isArrayLike(result) || isIterable(result)) && (result = observableFrom(result));
      return result;
    }).merge(maxConcurrent);
  }

  /**
   *  One of the Following:
   *  Projects each element of an observable sequence to an observable sequence and merges the resulting observable sequences into one observable sequence.
   *
   * @example
   *  var res = source.flatMapWithMaxConcurrent(5, function (x) { return Rx.Observable.range(0, x); });
   *  Or:
   *  Projects each element of an observable sequence to an observable sequence, invokes the result selector for the source element and each of the corresponding inner sequence's elements, and merges the results into one observable sequence.
   *
   *  var res = source.flatMapWithMaxConcurrent(5, function (x) { return Rx.Observable.range(0, x); }, function (x, y) { return x + y; });
   *  Or:
   *  Projects each element of the source observable sequence to the other observable sequence and merges the resulting observable sequences into one observable sequence.
   *
   *  var res = source.flatMapWithMaxConcurrent(5, Rx.Observable.fromArray([1,2,3]));
   * @param selector A transform function to apply to each element or an observable sequence to project each element from the
   * source sequence onto which could be either an observable or Promise.
   * @param {Function} [resultSelector]  A transform function to apply to each element of the intermediate sequence.
   * @param {Any} [thisArg] Object to use as this when executing callback.
   * @returns {Observable} An observable sequence whose elements are the result of invoking the one-to-many transform function collectionSelector on each element of the input sequence and then mapping each of those sequence elements and their corresponding source element to a result element.
   */
  observableProto.selectManyWithMaxConcurrent = observableProto.flatMapWithMaxConcurrent = function (maxConcurrent, selector, resultSelector, thisArg) {
    if (isFunction(selector) && isFunction(resultSelector)) {
      return this.flatMapWithMaxConcurrent(maxConcurrent, function (x, i) {
        var selectorResult = selector(x, i);
        isPromise(selectorResult) && (selectorResult = observableFromPromise(selectorResult));
        (isArrayLike(selectorResult) || isIterable(selectorResult)) && (selectorResult = observableFrom(selectorResult));

        return selectorResult.map(function (y) {
          return resultSelector(x, y, i);
        });
      }, thisArg);
    }
    return isFunction(selector) ?
      flatMapWithMaxConcurrent(this, maxConcurrent, selector, thisArg) :
      flatMapWithMaxConcurrent(this, maxConcurrent, function () { return selector; });
  };
