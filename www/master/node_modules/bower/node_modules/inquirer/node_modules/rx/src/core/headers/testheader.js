  // Defaults
  var Observer = Rx.Observer,
    Observable = Rx.Observable,
    Notification = Rx.Notification,
    VirtualTimeScheduler = Rx.VirtualTimeScheduler,
    Disposable = Rx.Disposable,
    disposableEmpty = Disposable.empty,
    disposableCreate = Disposable.create,
    CompositeDisposable = Rx.CompositeDisposable,
    SingleAssignmentDisposable = Rx.SingleAssignmentDisposable,
    inherits = Rx.internals.inherits,
    defaultComparer = Rx.internals.isEqual;
