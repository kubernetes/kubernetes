  /**
   *  Checks access to the observer for grammar violations. This includes checking for multiple OnError or OnCompleted calls, as well as reentrancy in any of the observer methods.
   *  If a violation is detected, an Error is thrown from the offending observer method call.
   *
   * @returns An observer that checks callbacks invocations against the observer grammar and, if the checks pass, forwards those to the specified observer.
   */
  Observer.prototype.checked = function () { return new CheckedObserver(this); };

  /**
   * Schedules the invocation of observer methods on the given scheduler.
   * @param {Scheduler} scheduler Scheduler to schedule observer messages on.
   * @returns {Observer} Observer whose messages are scheduled on the given scheduler.
   */
  Observer.notifyOn = function (scheduler) {
    return new ObserveOnObserver(scheduler, this);
  };

  /**
  *  Creates an observer from a notification callback.
  * @param {Function} handler Action that handles a notification.
  * @returns The observer object that invokes the specified handler using a notification corresponding to each message it receives.
  */
  Observer.fromNotifier = function (handler, thisArg) {
    var handlerFunc = bindCallback(handler, thisArg, 1);
    return new AnonymousObserver(function (x) {
      return handlerFunc(notificationCreateOnNext(x));
    }, function (e) {
      return handlerFunc(notificationCreateOnError(e));
    }, function () {
      return handlerFunc(notificationCreateOnCompleted());
    });
  };

  /**
  *  Creates a notification callback from an observer.
  * @returns The action that forwards its input notification to the underlying observer.
  */
  Observer.prototype.toNotifier = function () {
    var observer = this;
    return function (n) { return n.accept(observer); };
  };

  /**
  *  Hides the identity of an observer.
  * @returns An observer that hides the identity of the specified observer.
  */
  Observer.prototype.asObserver = function () {
    var source = this;
    return new AnonymousObserver(
      function (x) { source.onNext(x); },
      function (e) { source.onError(e); },
      function () { source.onCompleted(); }
    );
  };
