  /**
   * jortSort checks if your inputs are sorted.  Note that this is only for a sequence with an end.
   * See http://jort.technology/ for full details.
   * @returns {Observable} An observable which has a single value of true if sorted, else false.
   */
  observableProto.jortSort = function () {
    return this.jortSortUntil(observableNever());
  };
