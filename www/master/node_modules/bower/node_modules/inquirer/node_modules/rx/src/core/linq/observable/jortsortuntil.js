  /**
   * jortSort checks if your inputs are sorted until another Observable sequence fires.
   * See http://jort.technology/ for full details.
   * @returns {Observable} An observable which has a single value of true if sorted, else false.
   */
  observableProto.jortSortUntil = function (other) {
    var source = this;
    return new AnonymousObservable(function (observer) {
      var arr = [];
      return source.takeUntil(other).subscribe(
        arr.push.bind(arr),
        observer.onError.bind(observer),
        function () {
          var sorted = arr.slice(0).sort(defaultSubComparer);
          observer.onNext(isEqual(arr, sorted));
          observer.onCompleted();
        });
    }, source);
  };
