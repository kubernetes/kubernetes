  /**
   * Merges an observable sequence of observable sequences into an observable sequence.
   * @returns {Observable} The observable sequence that merges the elements of the inner sequences.
   */
  observableProto.mergeAll = observableProto.mergeObservable = function () {
    var sources = this;
    return new AnonymousObservable(function (o) {
      var group = new CompositeDisposable(),
        isStopped = false,
        m = new SingleAssignmentDisposable();

      group.add(m);
      m.setDisposable(sources.subscribe(function (innerSource) {
        var innerSubscription = new SingleAssignmentDisposable();
        group.add(innerSubscription);

        // Check for promises support
        isPromise(innerSource) && (innerSource = observableFromPromise(innerSource));

        innerSubscription.setDisposable(innerSource.subscribe(function (x) { o.onNext(x); }, function (e) { o.onError(e); }, function () {
          group.remove(innerSubscription);
          isStopped && group.length === 1 && o.onCompleted();
        }));
      }, function (e) { o.onError(e); }, function () {
        isStopped = true;
        group.length === 1 && o.onCompleted();
      }));
      return group;
    }, sources);
  };
