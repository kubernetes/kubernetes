  var SwitchObservable = (function(__super__) {
    inherits(SwitchObservable, __super__);
    function SwitchObservable(source) {
      this.source = source;
      __super__.call(this);
    }

    SwitchObservable.prototype.subscribeCore = function (observer) {
      var innerSubscription = new SerialDisposable(),
        subscription = this.source.subscribe(new SwitchObserver(observer, innerSubscription));
      return new CompositeDisposable(subscription, innerSubscription);
    };

    return SwitchObservable;
  }(ObservableBase));

  var SwitchObserver = (function(__super__) {
    inherits(SwitchObserver, __super__);
    function SwitchObserver(observer, innerSubscription) {
      this.observer = observer;
      this.innerSubscription = innerSubscription;
      this.stopped = false;
      this.latest = 0;
      this.hasLatest = false;
      __super__.call(this);
    }

    SwitchObserver.prototype.next = function (innerSource) {
      var d = new SingleAssignmentDisposable(), id = ++this.latest;
      this.hasLatest = true;
      this.innerSubscription.setDisposable(d);

      // Check if Promise or Observable
      isPromise(innerSource) && (innerSource = observableFromPromise(innerSource));

      d.setDisposable(innerSource.subscribe(new InnerObserver(this, id)));
    };

    SwitchObserver.prototype.error = function (e) {
      this.observer.onError(e);
    };

    SwitchObserver.prototype.completed = function () {
      this.stopped = true;
      !this.hasLatest && this.observer.onCompleted();
    };

    var InnerObserver = (function(__base__) {
      inherits(InnerObserver, __base__);
      function InnerObserver(parent, id) {
        this.parent = parent;
        this.id = id;
        __base__.call(this);
      }
      InnerObserver.prototype.next = function (x) { this.parent.latest === this.id && this.parent.observer.onNext(x); };
      InnerObserver.prototype.error = function (e) { this.parent.latest === this.id && this.parent.observer.onError(e); };
      InnerObserver.prototype.completed = function () {
        if (this.parent.latest === this.id) {
          this.parent.hasLatest = false;
          this.parent.isStopped && this.parent.observer.onCompleted();
        }
      };

      return InnerObserver;
    }(AbstractObserver));

    return SwitchObserver;
  }(AbstractObserver));

  /**
  * Transforms an observable sequence of observable sequences into an observable sequence producing values only from the most recent observable sequence.
  * @returns {Observable} The observable sequence that at any point in time produces the elements of the most recent inner observable sequence that has been received.
  */
  observableProto['switch'] = observableProto.switchLatest = function () {
    return new SwitchObservable(this);
  };
