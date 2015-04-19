  /**
   * Merges the specified observable sequences into one observable sequence by emitting a list with the elements of the observable sequences at corresponding indexes.
   * @param arguments Observable sources.
   * @returns {Observable} An observable sequence containing lists of elements at corresponding indexes.
   */
  Observable.zipArray = function () {
    var sources;
    if (Array.isArray(arguments[0])) {
      sources = arguments[0];
    } else {
      var len = arguments.length;
      sources = new Array(len);
      for(var i = 0; i < len; i++) { sources[i] = arguments[i]; }
    }
    return new AnonymousObservable(function (observer) {
      var n = sources.length,
        queues = arrayInitialize(n, function () { return []; }),
        isDone = arrayInitialize(n, function () { return false; });

      function next(i) {
        if (queues.every(function (x) { return x.length > 0; })) {
          var res = queues.map(function (x) { return x.shift(); });
          observer.onNext(res);
        } else if (isDone.filter(function (x, j) { return j !== i; }).every(identity)) {
          observer.onCompleted();
          return;
        }
      };

      function done(i) {
        isDone[i] = true;
        if (isDone.every(identity)) {
          observer.onCompleted();
          return;
        }
      }

      var subscriptions = new Array(n);
      for (var idx = 0; idx < n; idx++) {
        (function (i) {
          subscriptions[i] = new SingleAssignmentDisposable();
          subscriptions[i].setDisposable(sources[i].subscribe(function (x) {
            queues[i].push(x);
            next(i);
          }, function (e) { observer.onError(e); }, function () {
            done(i);
          }));
        })(idx);
      }

      return new CompositeDisposable(subscriptions);
    });
  };
