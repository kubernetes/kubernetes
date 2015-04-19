  /**
   * Combines emissions from both input streams that happen at approximately
   * the same time, defined by a certain time window duration.
   *
   * first:  ----f--f-----f-------f------f------f---f---|>
   * second: -s---s-s----s---s-------s------s--s----s---|>
   * result: -----R-R-----R---------------------R---R---|>
   *
   * @param first The first source Observable
   * @param second The second source Observable
   * @param windowDuration the length of the time window to consider two items close enough
   * @param unit the time unit of time
   * @param combineFunction a function that computes an item to be emitted by the resulting Observable for any two overlapping items emitted by the sources
   * @param an Observable that emits items correlating to items emitted by the source Observables that have happen at approximately the same time
   */
   Observable.combineSynchronized = function (second, windowDuration, combineFunction, scheduler) {
     return this.join(
       second,
       function () {
         return observableTimer(windowDuration, scheduler);
       },
       function () {
         return observableTimer(windowDuration, scheduler);
       },
       combineFunction);
   };
