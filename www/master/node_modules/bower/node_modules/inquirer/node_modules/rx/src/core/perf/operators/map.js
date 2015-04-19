  var MapObservable = (function (__super__) {
    inherits(MapObservable, __super__);

    function MapObservable(source, selector, thisArg) {
      this.source = source;
      this.selector = bindCallback(selector, thisArg, 3);
      __super__.call(this);
    }

    MapObservable.prototype.internalMap = function (selector, thisArg) {
      var self = this;
      return new MapObservable(this.source, function (x, i, o) { return selector.call(this, self.selector(x, i, o), i, o); }, thisArg)
    };

    MapObservable.prototype.subscribeCore = function (observer) {
      return this.source.subscribe(new MapObserver(observer, this.selector, this));
    };

    return MapObservable;

  }(ObservableBase));

  function MapObserver(observer, selector, source) {
    this.observer = observer;
    this.selector = selector;
    this.source = source;
    this.i = 0;
    this.isStopped = false;
  }

  MapObserver.prototype.onNext = function(x) {
    if (this.isStopped) { return; }
    var result = tryCatch(this.selector).call(this, x, this.i++, this.source);
    if (result === errorObj) {
      return this.observer.onError(result.e);
    }
    this.observer.onNext(result);
  };
  MapObserver.prototype.onError = function (e) {
    if(!this.isStopped) { this.isStopped = true; this.observer.onError(e); }
  };
  MapObserver.prototype.onCompleted = function () {
    if(!this.isStopped) { this.isStopped = true; this.observer.onCompleted(); }
  };
  MapObserver.prototype.dispose = function() { this.isStopped = true; };
  MapObserver.prototype.fail = function (e) {
    if (!this.isStopped) {
      this.isStopped = true;
      this.observer.onError(e);
      return true;
    }

    return false;
  };

  /**
  * Projects each element of an observable sequence into a new form by incorporating the element's index.
  * @param {Function} selector A transform function to apply to each source element; the second parameter of the function represents the index of the source element.
  * @param {Any} [thisArg] Object to use as this when executing callback.
  * @returns {Observable} An observable sequence whose elements are the result of invoking the transform function on each element of source.
  */
  observableProto.map = observableProto.select = function (selector, thisArg) {
    var selectorFn = typeof selector === 'function' ? selector : function () { return selector; };
    return this instanceof MapObservable ?
      this.internalMap(selectorFn, thisArg) :
      new MapObservable(this, selectorFn, thisArg);
  };
