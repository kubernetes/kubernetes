  var Observable = Rx.Observable,
    observableProto = Observable.prototype,
    AnonymousObservable = Rx.AnonymousObservable,
    Subject = Rx.Subject,
    AsyncSubject = Rx.AsyncSubject,
    Observer = Rx.Observer,
    ScheduledObserver = Rx.internals.ScheduledObserver,
    disposableCreate = Rx.Disposable.create,
    disposableEmpty = Rx.Disposable.empty,
    CompositeDisposable = Rx.CompositeDisposable,
    currentThreadScheduler = Rx.Scheduler.currentThread,
    isFunction = Rx.helpers.isFunction,
    inherits = Rx.internals.inherits,
    addProperties = Rx.internals.addProperties,
    checkDisposed = Rx.Disposable.checkDisposed;

  // Utilities
  function cloneArray(arr) {
    var len = arr.length, a = new Array(len);
    for(var i = 0; i < len; i++) { a[i] = arr[i]; }
    return a;
  }
