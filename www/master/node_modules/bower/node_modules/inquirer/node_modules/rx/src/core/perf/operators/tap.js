  var TapObservable = (function(__super__) {
    inherits(TapObservable,__super__);
    function TapObservable(source, observerOrOnNext, onError, onCompleted) {
      this.source = source;
      this.tapObserver = !observerOrOnNext || isFunction(observerOrOnNext) ?
        observerCreate(observerOrOnNext || noop, onError || noop, onCompleted || noop) :
        observerOrOnNext;;
      __super__.call(this);
    }

    TapObservable.prototype.subscribeCore = function(observer) {
      return this.source.subscribe(new TapObserver(observer, this.tapObserver));
    };

    return TapObservable;
  }(ObservableBase));

  var TapObserver = (function(__super__){
    inherits(TapObserver, __super__);
    function TapObserver(observer, tapObserver) {
      this.observer = observer;
      this.tapObserver = tapObserver;
      __super__.call(this;)
    }
    TapObserver.prototype.next = function(x) {
      try {
        this.tapObserver.onNext(x);
      } catch (e) {
        return this.observer.onError(e);
      }
      this.observer.onNext(x);
    };
    TapObserver.prototype.error = function(err) {
      try {
        this.tapObserver.onError(err);
      } catch (e) {
        return this.observer.onError(e);
      }
      this.observer.onError(err);
    };
    TapObserver.prototype.completed = function() {
      try {
        this.tapObserver.onCompleted();
      } catch (e) {
        return this.observer.onError(e);
      }
      this.observer.onCompleted();
    };

    return TapObserver;
  }(AbstractObserver));

  /**
  *  Invokes an action for each element in the observable sequence and invokes an action upon graceful or exceptional termination of the observable sequence.
  *  This method can be used for debugging, logging, etc. of query behavior by intercepting the message stream to run arbitrary actions for messages on the pipeline.
  * @param {Function | Observer} observerOrOnNext Action to invoke for each element in the observable sequence or an observer.
  * @param {Function} [onError]  Action to invoke upon exceptional termination of the observable sequence. Used if only the observerOrOnNext parameter is also a function.
  * @param {Function} [onCompleted]  Action to invoke upon graceful termination of the observable sequence. Used if only the observerOrOnNext parameter is also a function.
  * @returns {Observable} The source sequence with the side-effecting behavior applied.
  */
  observableProto['do'] = observableProto.tap = function (observerOrOnNext, onError, onCompleted) {

    return new TapObservable(this, observerOrOnNext, onError, onCompleted);
  };

  /** @deprecated use #do or #tap instead. */
  observableProto.doAction = function () {
    //deprecate('doAction', 'do or tap');
    return this.tap.apply(this, arguments);
  };

  /**
  *  Invokes an action for each element in the observable sequence.
  *  This method can be used for debugging, logging, etc. of query behavior by intercepting the message stream to run arbitrary actions for messages on the pipeline.
  * @param {Function} onNext Action to invoke for each element in the observable sequence.
  * @param {Any} [thisArg] Object to use as this when executing callback.
  * @returns {Observable} The source sequence with the side-effecting behavior applied.
  */
  observableProto.doOnNext = observableProto.tapOnNext = function (onNext, thisArg) {
    return this.tap(typeof thisArg !== 'undefined' ? function (x) { onNext.call(thisArg, x); } : onNext);
  };

  /**
  *  Invokes an action upon exceptional termination of the observable sequence.
  *  This method can be used for debugging, logging, etc. of query behavior by intercepting the message stream to run arbitrary actions for messages on the pipeline.
  * @param {Function} onError Action to invoke upon exceptional termination of the observable sequence.
  * @param {Any} [thisArg] Object to use as this when executing callback.
  * @returns {Observable} The source sequence with the side-effecting behavior applied.
  */
  observableProto.doOnError = observableProto.tapOnError = function (onError, thisArg) {
    return this.tap(noop, typeof thisArg !== 'undefined' ? function (e) { onError.call(thisArg, e); } : onError);
  };

  /**
  *  Invokes an action upon graceful termination of the observable sequence.
  *  This method can be used for debugging, logging, etc. of query behavior by intercepting the message stream to run arbitrary actions for messages on the pipeline.
  * @param {Function} onCompleted Action to invoke upon graceful termination of the observable sequence.
  * @param {Any} [thisArg] Object to use as this when executing callback.
  * @returns {Observable} The source sequence with the side-effecting behavior applied.
  */
  observableProto.doOnCompleted = observableProto.tapOnCompleted = function (onCompleted, thisArg) {
    return this.tap(noop, null, typeof thisArg !== 'undefined' ? function () { onCompleted.call(thisArg); } : onCompleted);
  };
