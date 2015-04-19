  /**
   * Creates an array from an observable sequence.
   * @returns {Observable} An observable sequence containing a single element with a list containing all the elements of the source sequence.
   */
  observableProto.toArray = function () {
    var source = this;
    return new AnonymousObservable(function(observer) {
      var arr = [];
      return source.subscribe(
        function (x) { arr.push(x); },
        function (e) { observer.onError(e); },
        function () {
          observer.onNext(arr);
          observer.onCompleted();
        });
    }, source);
  };
