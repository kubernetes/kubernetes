  /**
   * Projects each element of an observable sequence into a new form by incorporating the element's index.
   * @param {Function} selector A transform function to apply to each source element; the second parameter of the function represents the index of the source element.
   * @param {Any} [thisArg] Object to use as this when executing callback.
   * @returns {Observable} An observable sequence whose elements are the result of invoking the transform function on each element of source.
   */
  observableProto.select = observableProto.map = function (selector, thisArg) {
    var selectorFn = isFunction(selector) ? bindCallback(selector, thisArg, 3) : function () { return selector; },
        source = this;
    return new AnonymousObservable(function (o) {
      var i = 0;
      return source.subscribe(function (x) {
        try {
          var result = selector(x, i++, source);
        } catch (e) {
          return observer.onError(e);
        }

        o.onNext(result);
      }, function (e) { o.onError(e); }, function () { o.onCompleted(); });
    }, source);
  };
