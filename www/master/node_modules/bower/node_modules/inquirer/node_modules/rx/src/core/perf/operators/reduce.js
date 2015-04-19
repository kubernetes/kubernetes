  var ReduceObservable = (function(__super__) {
    inherits(ReduceObservable, __super__);
    function ReduceObservable(source, accumulator, hasSeed, seed) {
      this.source = source;
      this.accumulator = accumulator;
      this.hasSeed = hasSeed;
      this.seed = seed;
      __super__.call(this);
    }

    ReduceObservable.prototype.subscribeCore = function(observer) {
      return this.source.subscribe(new ReduceObserver(observer,this));
    };

    return ReduceObservable;
  }(ObservableBase));

  var ReduceObserver = (function(__super__) {
    inherits(ReduceObserver, __super__);
    function ReduceObserver(observer, parent) {
      this.observer = observer;
      this.accumulator = parent.accumulator;
      this.hasSeed = parent.hasSeed;
      this.seed = parent.seed;
      this.hasAccumulation = false;
      this.accumulation = null;
      this.hasValue = false;
      __super__.call(this);
    }
    ReduceObserver.prototype.next = function (x) {
      !this.hasValue && (this.hasValue = true);
      try {
        if (this.hasAccumulation) {
          this.accumulation = this.accumulator(this.accumulation, x);
        } else {
          this.accumulation = this.hasSeed ? this.accumulator(this.seed, x) : x;
          this.hasAccumulation = true;
        }
      } catch (e) {
        this.observer.onError(e);
        return;
      }
    };
    ReduceObserver.prototype.error = function (e) { this.observer.onError(e); };
    ReduceObserver.prototype.completed = function () {
      this.hasValue && this.observer.onNext(this.accumulation);
      !this.hasValue && this.hasSeed && this.observer.onNext(seed);
      !this.hasValue && !this.hasSeed && this.observer.onError(new Error(sequenceContainsNoElements));
      this.observer.onCompleted();
    };

    return ReduceObserver;
  }(AbstractObserver));


  /**
  * Applies an accumulator function over an observable sequence, returning the result of the aggregation as a single element in the result sequence. The specified seed value is used as the initial accumulator value.
  * For aggregation behavior with incremental intermediate results, see Observable.scan.
  * @param {Function} accumulator An accumulator function to be invoked on each element.
  * @param {Any} [seed] The initial accumulator value.
  * @returns {Observable} An observable sequence containing a single element with the final accumulator value.
  */
  observableProto.reduce = function (accumulator) {
    var hasSeed = false;
    if (arguments.length === 2) {
      hasSeed = true;
      var seed = arguments[1];
    }
    return new ReduceObservable(this, accumulator, hasSeed, seed);
  };
