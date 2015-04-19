  var ScanObservable = (function(__super__) {
    inherits(ScanObservable, __super__);
    function ScanObservable(source, accumulator, hasSeed, seed) {
      this.source = source;
      this.accumulator = accumulator;
      this.hasSeed = hasSeed;
      this.seed = seed;
      __super__.call(this);
    }

    ScanObservable.prototype.subscribeCore = function(observer) {
      return this.source.subscribe(new ScanObserver(observer,this);
    };

    return ScanObservable;
  }(ObservableBase));

  var ScanObserver = (function(__super__) {
    inherits(ScanObserver, __super__);
    function ScanObserver(observer, parent) {
      this.observer = observer;
      this.accumulator = parent.accumulator;
      this.hasSeed = parent.hasSeed;
      this.seed = parent.seed;
      this.hasAccumulation = false;
      this.accumulation = null;
      this.hasValue = false;
      __super__.call(this);
    }
    ScanObserver.prototype.next = function (x) {
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

      this.observer.onNext(accumulation);
    };
    ScanObserver.prototype.error = function (e) { this.observer.onError(e); };
    ScanObserver.prototype.completed = function () {
      !this.hasValue && this.hasSeed && this.observer.onNext(seed);
      this.observer.onCompleted();
    };

    return ScanObserver;
  }(AbstractObserver));


  /**
  *  Applies an accumulator function over an observable sequence and returns each intermediate result. The optional seed value is used as the initial accumulator value.
  *  For aggregation behavior with no intermediate results, see Observable.aggregate.
  * @example
  *  var res = source.scan(function (acc, x) { return acc + x; });
  *  var res = source.scan(0, function (acc, x) { return acc + x; });
  * @param {Mixed} [seed] The initial accumulator value.
  * @param {Function} accumulator An accumulator function to be invoked on each element.
  * @returns {Observable} An observable sequence containing the accumulated values.
  */
  observableProto.scan = function () {
    var hasSeed = false, seed, accumulator, source = this;
    if (arguments.length === 2) {
      hasSeed = true;
      seed = arguments[0];
      accumulator = arguments[1];
    } else {
      accumulator = arguments[0];
    }
    return new ScanObservable(this, accumulator, hasSeed, seed);
  };
