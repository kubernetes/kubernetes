  // Aliases
  var Observable = Rx.Observable,
    observableProto = Observable.prototype,
    observableFromPromise = Observable.fromPromise,
    observableThrow = Observable.throwError,
    AnonymousObservable = Rx.AnonymousObservable,
    AsyncSubject = Rx.AsyncSubject,
    disposableCreate = Rx.Disposable.create,
    CompositeDisposable = Rx.CompositeDisposable,
    immediateScheduler = Rx.Scheduler.immediate,
    timeoutScheduler = Rx.Scheduler['default'],
    isScheduler = Rx.Scheduler.isScheduler,
    slice = Array.prototype.slice;
